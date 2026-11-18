"""
Generate professional REVIEW1_MAP.html using Esri World Basemaps (zero API key, zero 403 blocks).
"""

from pathlib import Path
import sys
import folium

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pathsnap_loader import load_trip

def main():
    print("Loading Trip 00000001...")
    trip = load_trip("00000001", partition="train")

    track_df = trip.track_df
    nodes_df = trip.nodes_df
    arcs_df = trip.arcs_df
    route = trip.route

    node_pos = dict(zip(nodes_df["node_id"], zip(nodes_df["lat"], nodes_df["lon"])))

    # Initialize map without default tile to avoid 403 errors
    center_lat = float(track_df["lat"].median())
    center_lon = float(track_df["lon"].median())

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles=None,
        control_scale=True,
    )

    # 1. Premium Esri Basemaps (No API key, No 403 block from file:// origin)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ",
        name="Light Gray Canvas (Clean)",
        control=True,
    ).add_to(fmap)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Source: Esri, DeLorme, NAVTEQ, USGS",
        name="Esri Streets",
        control=True,
    ).add_to(fmap)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS",
        name="Satellite Imagery",
        control=True,
    ).add_to(fmap)

    # 2. Road Network Layer (Subtle Grey Geometry)
    network_group = folium.FeatureGroup(name="Road Network Graph", show=True)
    min_lon, max_lon = track_df["lon"].min() - 0.006, track_df["lon"].max() + 0.006
    min_lat, max_lat = track_df["lat"].min() - 0.006, track_df["lat"].max() + 0.006

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
                weight=1.8,
                opacity=0.65,
            ).add_to(network_group)
            count += 1
            if count >= 3500:
                break
    network_group.add_to(fmap)

    # 3. Ground-Truth Route Layer (Vibrant Blue Polyline)
    if route:
        route_group = folium.FeatureGroup(name="Ground-Truth Route (Blue)", show=True)
        route_coords = []
        for aid in route:
            arc = arcs_df.loc[aid]
            u, v = int(arc["from_node"]), int(arc["to_node"])
            route_coords.append(node_pos[u])
            route_coords.append(node_pos[v])

        clean_route = [route_coords[0]]
        for pt in route_coords[1:]:
            if pt != clean_route[-1]:
                clean_route.append(pt)

        folium.PolyLine(
            locations=clean_route,
            color="#0d47a1",
            weight=5.5,
            opacity=0.95,
            tooltip=f"True Trajectory ({len(route)} segments)",
        ).add_to(route_group)
        route_group.add_to(fmap)

    # 4. Noisy GPS Observations (Bright Orange with crisp white borders)
    gps_group = folium.FeatureGroup(name="Noisy GPS Observations (Orange)", show=True)
    for idx in range(len(track_df)):
        row = track_df.iloc[idx]
        if idx % 2 == 0 or idx == len(track_df) - 1:
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=4,
                color="#ffffff",
                weight=1.2,
                fill=True,
                fill_color="#ff5722",
                fill_opacity=0.9,
                popup=f"<b>GPS #{idx}</b><br>Time: {row['timestamp']:.0f}s<br>Speed: {row['speed_mps']*3.6:.1f} km/h",
            ).add_to(gps_group)
    gps_group.add_to(fmap)

    # 5. High-visibility Start & End markers
    start_pt = [track_df.iloc[0]["lat"], track_df.iloc[0]["lon"]]
    end_pt = [track_df.iloc[-1]["lat"], track_df.iloc[-1]["lon"]]

    folium.Marker(
        location=start_pt,
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
        popup="<b style='color:#2e7d32;'>TRIP START</b>",
    ).add_to(fmap)

    folium.Marker(
        location=end_pt,
        icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa"),
        popup="<b style='color:#c62828;'>TRIP DESTINATION</b>",
    ).add_to(fmap)

    # 6. Fit bounds tightly to trajectory
    bounds = [
        [track_df["lat"].min() - 0.001, track_df["lon"].min() - 0.001],
        [track_df["lat"].max() + 0.001, track_df["lon"].max() + 0.001],
    ]
    fmap.fit_bounds(bounds, padding=[40, 40])

    # 7. Layer Control (Top Right)
    folium.LayerControl(collapsed=False, position="topright").add_to(fmap)

    # 8. Modern Glassmorphic Info Panel
    info_panel_html = """
    <div style="position: fixed; top: 15px; left: 55px; z-index: 9999;
         background: rgba(255, 255, 255, 0.97); padding: 16px 22px;
         border-radius: 10px; box-shadow: 0 6px 20px rgba(0,0,0,0.18);
         font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         max-width: 320px; border-left: 5px solid #0d47a1;">
      <div style="display:flex; align-items:center; margin-bottom: 6px;">
        <span style="font-size: 16px; font-weight: 800; color: #0d47a1; letter-spacing: -0.3px;">
          PathSnap &mdash; GPS Map Matching
        </span>
      </div>
      <div style="font-size: 12px; color: #555; line-height: 1.5; margin-bottom: 8px;">
        <b>Trip ID:</b> 00000001 (Train Set)<br>
        <b>GPS Observations:</b> 1,315 points (1 Hz nominal)<br>
        <b>Ground Truth Arcs:</b> 525 road segments
      </div>
      <div style="border-top: 1px solid #eee; padding-top: 8px; font-size: 12px; line-height: 1.7;">
        <div style="display:flex; align-items:center;">
          <span style="display:inline-block; width:12px; height:12px; background:#ff5722; border:1px solid #fff; border-radius:50%; margin-right:8px; box-shadow: 0 0 2px rgba(0,0,0,0.4);"></span>
          <b>Orange Dots:</b>&nbsp;Raw Noisy GPS Observations
        </div>
        <div style="display:flex; align-items:center;">
          <span style="display:inline-block; width:16px; height:4px; background:#0d47a1; border-radius:2px; margin-right:8px;"></span>
          <b>Blue Line:</b>&nbsp;Ground-Truth Trajectory
        </div>
        <div style="display:flex; align-items:center;">
          <span style="display:inline-block; width:16px; height:2px; background:#9e9e9e; margin-right:8px;"></span>
          <b>Grey Lines:</b>&nbsp;Road Network Topology
        </div>
        <div style="margin-top: 4px; color: #444; font-size: 11px;">
          <span style="color:#2e7d32; font-weight:bold;">&#9658; Green:</span> Trip Start &nbsp;|&nbsp;
          <span style="color:#c62828; font-weight:bold;">&#9632; Red:</span> Destination
        </div>
      </div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(info_panel_html))

    out_file = PROJECT_ROOT / "outputs" / "figures" / "REVIEW1_MAP.html"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_file))
    print(f"SUCCESS: Generated clean map at {out_file}")

if __name__ == "__main__":
    main()
