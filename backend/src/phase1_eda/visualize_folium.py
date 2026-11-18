"""
Folium Geographical Visualizations for PathSnap
Renders multi-layer interactive maps:
- Raw GPS track observations (blue circle markers)
- Ground-truth reference route (green line)
- Predicted reconstructed route (red line)
- Local road-network geometry (gray lines)
- Layer controls, legends, and popups
"""

from pathlib import Path
from typing import List, Optional
import folium
from folium import plugins
from src.pathsnap_loader import TripData, load_trip


def create_trip_map(
    trip_data: TripData,
    predicted_arcs: Optional[List[int]] = None,
    show_road_network: bool = True,
    max_network_arcs: int = 1500,
    out_html: Optional[Path] = None,
) -> folium.Map:
    """
    Generate an interactive multi-layer Folium map for a trip.
    """
    track_df = trip_data.track_df
    nodes_df = trip_data.nodes_df
    arcs_df = trip_data.arcs_df
    route = trip_data.route

    # Center map on trip median coordinates
    center_lat = float(track_df["lat"].median())
    center_lon = float(track_df["lon"].median())

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    node_pos = dict(zip(nodes_df["node_id"], zip(nodes_df["lat"], nodes_df["lon"])))

    # 1. Road Network Background Layer
    if show_road_network:
        network_group = folium.FeatureGroup(name="Road Network (Arcs)", show=False)
        # To avoid lagging the browser, sample arcs near the trajectory bounding box
        min_lon, max_lon = track_df["lon"].min() - 0.005, track_df["lon"].max() + 0.005
        min_lat, max_lat = track_df["lat"].min() - 0.005, track_df["lat"].max() + 0.005

        count = 0
        for _, arc in arcs_df.iterrows():
            u, v = int(arc["from_node"]), int(arc["to_node"])
            ulat, ulon = node_pos[u]
            vlat, vlon = node_pos[v]
            if (min_lat <= ulat <= max_lat and min_lon <= ulon <= max_lon) or (
                min_lat <= vlat <= max_lat and min_lon <= vlon <= max_lon
            ):
                folium.PolyLine(
                    locations=[[ulat, ulon], [vlat, vlon]],
                    color="#9e9e9e",
                    weight=1.5,
                    opacity=0.6,
                ).add_to(network_group)
                count += 1
                if count >= max_network_arcs:
                    break
        network_group.add_to(fmap)

    # 2. Ground Truth Route Layer (Green)
    if route:
        route_group = folium.FeatureGroup(name="Reference Route (Ground Truth)", show=True)
        route_coords = []
        for aid in route:
            arc = arcs_df.loc[aid]
            u, v = int(arc["from_node"]), int(arc["to_node"])
            route_coords.append(node_pos[u])
            route_coords.append(node_pos[v])

        # Remove consecutive duplicate points for clean rendering
        clean_route = [route_coords[0]]
        for pt in route_coords[1:]:
            if pt != clean_route[-1]:
                clean_route.append(pt)

        folium.PolyLine(
            locations=clean_route,
            color="#2e7d32",
            weight=4.5,
            opacity=0.85,
            tooltip=f"Ground Truth Route ({len(route)} arcs)",
        ).add_to(route_group)
        route_group.add_to(fmap)

    # 3. Predicted Route Layer (Red)
    if predicted_arcs:
        pred_group = folium.FeatureGroup(name="Predicted Reconstructed Route", show=True)
        pred_coords = []
        for aid in predicted_arcs:
            if aid < len(arcs_df):
                arc = arcs_df.loc[aid]
                u, v = int(arc["from_node"]), int(arc["to_node"])
                pred_coords.append(node_pos[u])
                pred_coords.append(node_pos[v])

        if pred_coords:
            clean_pred = [pred_coords[0]]
            for pt in pred_coords[1:]:
                if pt != clean_pred[-1]:
                    clean_pred.append(pt)

            folium.PolyLine(
                locations=clean_pred,
                color="#d32f2f",
                weight=3.5,
                dash_array="6, 6",
                opacity=0.9,
                tooltip=f"Predicted Route ({len(predicted_arcs)} arcs)",
            ).add_to(pred_group)
        pred_group.add_to(fmap)

    # 4. Raw GPS Observations Layer (Blue Dots)
    gps_group = folium.FeatureGroup(name="Raw GPS Observations (Noisy)", show=True)
    # Downsample points for visual clarity if track is very dense
    step = 1 if len(track_df) <= 1000 else int(len(track_df) / 1000)
    for idx in range(0, len(track_df), step):
        row = track_df.iloc[idx]
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=3,
            color="#1976d2",
            fill=True,
            fill_color="#2196f3",
            fill_opacity=0.8,
            popup=f"GPS #{idx}<br>Time: {row['timestamp']:.1f}s<br>Speed: {row['speed_mps']:.1f} m/s",
        ).add_to(gps_group)

    # Start and End markers
    start_pt = [track_df.iloc[0]["lat"], track_df.iloc[0]["lon"]]
    end_pt = [track_df.iloc[-1]["lat"], track_df.iloc[-1]["lon"]]
    folium.Marker(
        location=start_pt,
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
        popup="Trip Start",
    ).add_to(gps_group)
    folium.Marker(
        location=end_pt,
        icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa"),
        popup="Trip End",
    ).add_to(gps_group)

    gps_group.add_to(fmap)

    # Layer Control and Legend
    folium.LayerControl(collapsed=False).add_to(fmap)

    if out_html:
        out_html.parent.mkdir(parents=True, exist_ok=True)
        fmap.save(str(out_html))
        print(f"Saved interactive Folium map to: {out_html}")

    return fmap
