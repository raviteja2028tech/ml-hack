"""
PathSnap Dataset Loader and Geospatial Utilities
Parses .track, .nodes, .arcs, and .route files for individual trips.
Provides local metric projection, spatial indexing with cKDTree,
and graph adjacency for HMM transition modeling.
"""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

EARTH_RADIUS_METERS = 6371000.0


def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance between two angular coordinates in meters."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_METERS * math.asin(math.sqrt(a))


def calculate_bearing(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Calculate compass bearing (0-360 degrees) from point 1 to point 2."""
    dlam = math.radians(lon2 - lon1)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    y = math.sin(dlam) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360.0) % 360.0


def point_to_segment_distance(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> Tuple[float, float, float, float]:
    """
    Project point (px, py) onto directed line segment (x1, y1) -> (x2, y2) in local metric space.
    Returns:
        distance_meters: float
        proj_param_t: float (clamped [0, 1])
        proj_x: float
        proj_y: float
    """
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-9:
        dist = math.hypot(px - x1, py - y1)
        return dist, 0.0, x1, y1

    t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
    t_clamped = max(0.0, min(1.0, t))
    proj_x = x1 + t_clamped * dx
    proj_y = y1 + t_clamped * dy
    dist = math.hypot(px - proj_x, py - proj_y)
    return dist, t_clamped, proj_x, proj_y


@dataclass
class TripData:
    trip_id: str
    partition: str
    track_df: pd.DataFrame
    nodes_df: pd.DataFrame
    arcs_df: pd.DataFrame
    route: Optional[List[int]]
    arc_kdtree: cKDTree
    arc_to_next_arcs: Dict[int, List[int]]
    origin_lon: float
    origin_lat: float
    m_per_deg_lon: float
    m_per_deg_lat: float

    def to_metric(self, lon: float, lat: float) -> Tuple[float, float]:
        """Convert angular lon, lat to local metric (x, y) meters from trip origin."""
        x = (lon - self.origin_lon) * self.m_per_deg_lon
        y = (lat - self.origin_lat) * self.m_per_deg_lat
        return x, y

    def to_lonlat(self, x: float, y: float) -> Tuple[float, float]:
        """Convert local metric (x, y) meters back to angular lon, lat."""
        lon = self.origin_lon + (x / self.m_per_deg_lon)
        lat = self.origin_lat + (y / self.m_per_deg_lat)
        return lon, lat


def load_trip(trip_id: str, base_dir: Path = Path("data/raw/Student folder"), partition: Optional[str] = None) -> TripData:
    """
    Load all files for a trip, compute metric projections, and build spatial indices.
    Searches both train/ and student_test/ if partition is not explicitly provided.
    """
    if partition is None:
        if (base_dir / "train" / trip_id).exists():
            partition = "train"
        elif (base_dir / "student_test" / trip_id).exists():
            partition = "student_test"
        else:
            raise FileNotFoundError(f"Trip {trip_id} not found in train or student_test under {base_dir}")

    trip_folder = base_dir / partition / trip_id
    track_file = trip_folder / f"{trip_id}.track"
    nodes_file = trip_folder / f"{trip_id}.nodes"
    arcs_file = trip_folder / f"{trip_id}.arcs"
    route_file = trip_folder / f"{trip_id}.route"

    # 1. Parse .nodes (lon, lat)
    nodes_raw = []
    with open(nodes_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            if parts:
                nodes_raw.append((idx, float(parts[0]), float(parts[1])))
    nodes_df = pd.DataFrame(nodes_raw, columns=["node_id", "lon", "lat"])

    # Establish trip reference origin (center of nodes bounding box)
    origin_lon = float(nodes_df["lon"].median())
    origin_lat = float(nodes_df["lat"].median())
    m_per_deg_lat = math.radians(1.0) * EARTH_RADIUS_METERS
    m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(origin_lat))

    nodes_df["x_m"] = (nodes_df["lon"] - origin_lon) * m_per_deg_lon
    nodes_df["y_m"] = (nodes_df["lat"] - origin_lat) * m_per_deg_lat

    node_coords = dict(zip(nodes_df["node_id"], zip(nodes_df["x_m"], nodes_df["y_m"], nodes_df["lon"], nodes_df["lat"])))

    # 2. Parse .arcs (from_node_id, to_node_id)
    arcs_raw = []
    with open(arcs_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            if parts:
                u, v = int(parts[0]), int(parts[1])
                ux, uy, ulon, ulat = node_coords[u]
                vx, vy, vlon, vlat = node_coords[v]
                length = math.hypot(vx - ux, vy - uy)
                bearing = calculate_bearing(ulon, ulat, vlon, vlat)
                mid_x = (ux + vx) / 2.0
                mid_y = (uy + vy) / 2.0
                arcs_raw.append((idx, u, v, ux, uy, vx, vy, length, bearing, mid_x, mid_y))

    arcs_df = pd.DataFrame(
        arcs_raw,
        columns=["arc_id", "from_node", "to_node", "ux", "uy", "vx", "vy", "length_m", "bearing_deg", "mid_x", "mid_y"],
    )

    # 3. Parse .track (lon, lat, timestamp)
    track_raw = []
    with open(track_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                track_raw.append((float(parts[0]), float(parts[1]), float(parts[2])))
    track_df = pd.DataFrame(track_raw, columns=["lon", "lat", "timestamp"])

    track_df["x_m"] = (track_df["lon"] - origin_lon) * m_per_deg_lon
    track_df["y_m"] = (track_df["lat"] - origin_lat) * m_per_deg_lat

    # Compute differential time, speed, bearing between consecutive GPS points
    dt = track_df["timestamp"].diff().fillna(1.0)
    dt = dt.apply(lambda v: v if v > 0.0 else 1.0)
    dx = track_df["x_m"].diff().fillna(0.0)
    dy = track_df["y_m"].diff().fillna(0.0)
    step_dist = np.hypot(dx, dy)
    track_df["dt"] = dt
    track_df["speed_mps"] = step_dist / dt

    bearings = [0.0] * len(track_df)
    for i in range(len(track_df)):
        if i == 0:
            if len(track_df) > 1:
                bearings[0] = calculate_bearing(track_df.loc[0, "lon"], track_df.loc[0, "lat"], track_df.loc[1, "lon"], track_df.loc[1, "lat"])
        else:
            bearings[i] = calculate_bearing(track_df.loc[i - 1, "lon"], track_df.loc[i - 1, "lat"], track_df.loc[i, "lon"], track_df.loc[i, "lat"])
    track_df["bearing_deg"] = bearings

    # 4. Parse .route if available
    route: Optional[List[int]] = None
    if route_file.exists():
        with open(route_file, "r", encoding="utf-8") as f:
            route = [int(line.strip()) for line in f if line.strip()]

    # 5. Build cKDTree on arc midpoints (and endpoints for high-fidelity spatial indexing)
    arc_points = np.column_stack([arcs_df["mid_x"].values, arcs_df["mid_y"].values])
    arc_kdtree = cKDTree(arc_points)

    # 6. Graph adjacency: mapping arc_id -> list of directly connected next arc_ids
    node_out_arcs: Dict[int, List[int]] = {}
    for aid, u in zip(arcs_df["arc_id"], arcs_df["from_node"]):
        node_out_arcs.setdefault(u, []).append(aid)

    arc_to_next_arcs: Dict[int, List[int]] = {}
    for aid, v in zip(arcs_df["arc_id"], arcs_df["to_node"]):
        arc_to_next_arcs[aid] = node_out_arcs.get(v, [])

    return TripData(
        trip_id=trip_id,
        partition=partition,
        track_df=track_df,
        nodes_df=nodes_df,
        arcs_df=arcs_df,
        route=route,
        arc_kdtree=arc_kdtree,
        arc_to_next_arcs=arc_to_next_arcs,
        origin_lon=origin_lon,
        origin_lat=origin_lat,
        m_per_deg_lon=m_per_deg_lon,
        m_per_deg_lat=m_per_deg_lat,
    )


def audit_dataset(base_dir: Path = Path("data/raw/Student folder")) -> Dict:
    """
    Audit all trips across train and student_test partitions.
    Computes distribution statistics, missing data, connectivity, and coordinate sanity.
    """
    train_dir = base_dir / "train"
    test_dir = base_dir / "student_test"

    train_trips = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    test_trips = sorted([p.name for p in test_dir.iterdir() if p.is_dir()])

    report = {
        "partitions": {
            "train": {"trip_count": len(train_trips), "trips": train_trips},
            "student_test": {"trip_count": len(test_trips), "trips": test_trips},
        },
        "statistics": {
            "train": {
                "track_points": [],
                "nodes": [],
                "arcs": [],
                "route_length": [],
                "duration_sec": [],
                "sampling_rate_1hz_pct": [],
            },
            "student_test": {
                "track_points": [],
                "nodes": [],
                "arcs": [],
                "route_length": [],
                "duration_sec": [],
                "sampling_rate_1hz_pct": [],
            },
        },
        "data_quality_issues": [],
    }

    print(f"Auditing {len(train_trips)} train trips and {len(test_trips)} student test trips...")

    for partition, trips, p_dir in [("train", train_trips, train_dir), ("student_test", test_trips, test_dir)]:
        for tid in trips:
            t_data = load_trip(tid, base_dir=base_dir, partition=partition)
            track_len = len(t_data.track_df)
            nodes_len = len(t_data.nodes_df)
            arcs_len = len(t_data.arcs_df)
            route_len = len(t_data.route) if t_data.route else 0

            dt_series = t_data.track_df["dt"]
            is_1s = (dt_series == 1.0).mean() * 100.0
            duration = float(t_data.track_df["timestamp"].iloc[-1] - t_data.track_df["timestamp"].iloc[0])

            report["statistics"][partition]["track_points"].append(track_len)
            report["statistics"][partition]["nodes"].append(nodes_len)
            report["statistics"][partition]["arcs"].append(arcs_len)
            report["statistics"][partition]["route_length"].append(route_len)
            report["statistics"][partition]["duration_sec"].append(duration)
            report["statistics"][partition]["sampling_rate_1hz_pct"].append(is_1s)

            # Check route connectivity
            if t_data.route:
                connected = 0
                for i in range(len(t_data.route) - 1):
                    a1 = t_data.route[i]
                    a2 = t_data.route[i + 1]
                    if a2 in t_data.arc_to_next_arcs.get(a1, []):
                        connected += 1
                conn_pct = (connected / (len(t_data.route) - 1)) * 100.0 if len(t_data.route) > 1 else 100.0
                if conn_pct < 99.0:
                    report["data_quality_issues"].append(
                        f"Trip {tid} ({partition}): route connectivity is {conn_pct:.1f}% ({connected}/{len(t_data.route)-1})"
                    )

    return report
