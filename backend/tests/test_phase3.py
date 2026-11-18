"""
Phase 3 unit tests (stdlib unittest - pytest is not installed in the offline venv).

Run:  .venv\\Scripts\\python.exe -m unittest tests.test_phase3 -v
"""

import ast
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.candidate_generator import (  # noqa: E402
    ArcGeometry,
    CandidateConfig,
    angle_diff_deg,
    brute_force_candidates,
    generate_candidates,
    point_segment_distance,
)
from src.feature_engineering import (  # noqa: E402
    FEATURE_COLUMNS,
    FEATURE_DTYPES,
    FeatureConfig,
    build_arc_graph,
    build_candidate_features,
    compute_point_kinematics,
)
from src.pathsnap_loader import load_trip, point_to_segment_distance  # noqa: E402
from src.route_labeler import LabelConfig, align_track_to_route, boundary_margin, label_candidates  # noqa: E402

DATA = ROOT / "data" / "raw" / "Student folder"
HAVE_DATA = (DATA / "train" / "00000001").exists()
PROCESSED = ROOT / "data" / "processed" / "phase3"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_geom(segments, n_nodes=None):
    """segments: list of ((x1, y1), (x2, y2)) - each gets its own pair of node ids unless shared."""
    seg = np.asarray(segments, dtype=float)
    a = len(seg)
    return ArcGeometry(
        ux=seg[:, 0, 0], uy=seg[:, 0, 1], vx=seg[:, 1, 0], vy=seg[:, 1, 1],
        from_node=np.arange(0, 2 * a, 2), to_node=np.arange(1, 2 * a, 2), n_nodes=n_nodes or 2 * a,
    )


def make_graph_geom(nodes_xy, arcs):
    nodes = np.asarray(nodes_xy, dtype=float)
    arcs = np.asarray(arcs, dtype=np.int64)
    return ArcGeometry(
        ux=nodes[arcs[:, 0], 0], uy=nodes[arcs[:, 0], 1], vx=nodes[arcs[:, 1], 0], vy=nodes[arcs[:, 1], 1],
        from_node=arcs[:, 0], to_node=arcs[:, 1], n_nodes=len(nodes),
    )


def run_labels(geom, route, x, y, ts, cand_radius=50.0):
    kin = compute_point_kinematics(x, y, ts)
    cands = generate_candidates(geom, x, y, CandidateConfig(radius_m=cand_radius))
    hdg, hv = kin["heading_deg"].to_numpy(), kin["heading_valid"].to_numpy()
    al = align_track_to_route(x, y, hdg, hv, geom, route)
    mg = boundary_margin(x, y, hdg, hv, geom, route, al)
    cl, pl = label_candidates(cands, al, mg, geom)
    return al, cands, cl, pl


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
class TestGeometry(unittest.TestCase):
    def test_vectorised_distance_matches_scalar_reference(self):
        rng = np.random.default_rng(1)
        p = rng.uniform(-100, 100, (500, 2))
        s = rng.uniform(-100, 100, (500, 4))
        d, t, _ = point_segment_distance(p[:, 0], p[:, 1], s[:, 0], s[:, 1], s[:, 2], s[:, 3])
        for i in range(500):
            ref_d, ref_t, _, _ = point_to_segment_distance(p[i, 0], p[i, 1], *s[i])
            self.assertAlmostEqual(d[i], ref_d, places=9)
            self.assertAlmostEqual(t[i], ref_t, places=9)

    def test_known_distances(self):
        d, t, t_raw = point_segment_distance(np.array([5.0, -3.0, 14.0]), np.array([4.0, 0.0, 0.0]), 0.0, 0.0, 10.0, 0.0)
        np.testing.assert_allclose(d, [4.0, 3.0, 4.0])
        np.testing.assert_allclose(t, [0.5, 0.0, 1.0])
        np.testing.assert_allclose(t_raw, [0.5, -0.3, 1.4])

    def test_degenerate_segment(self):
        d, t, _ = point_segment_distance(3.0, 4.0, 0.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(float(d), 5.0)
        self.assertEqual(float(t), 0.0)

    def test_angle_diff_wraps(self):
        np.testing.assert_allclose(angle_diff_deg([350, 10, 180, 0], [10, 350, 0, 270]), [20, 20, 180, 90])

    def test_bearing_convention(self):
        g = make_geom([((0, 0), (0, 10)), ((0, 0), (10, 0)), ((0, 0), (0, -10)), ((0, 0), (-10, 0))])
        np.testing.assert_allclose(g.bearing_deg, [0, 90, 180, 270])

    @unittest.skipUnless(HAVE_DATA, "raw data not present")
    def test_projection_error_vs_haversine_small(self):
        """Local equirectangular metres vs haversine for GPS-to-node pairs within 100 m."""
        from src.pathsnap_loader import haversine_distance

        trip = load_trip("00000013", base_dir=DATA)  # widest trip (~54 km x 33 km)
        tr, nd = trip.track_df, trip.nodes_df
        from scipy.spatial import cKDTree

        tree = cKDTree(nd[["x_m", "y_m"]].to_numpy())
        idx = np.linspace(0, len(tr) - 1, 300).astype(int)
        d_proj, j = tree.query(tr[["x_m", "y_m"]].to_numpy()[idx])
        rel = []
        for k, i in enumerate(idx):
            if 5 < d_proj[k] < 100:
                h = haversine_distance(tr.lon.iat[i], tr.lat.iat[i], nd.lon.iat[j[k]], nd.lat.iat[j[k]])
                rel.append(abs(d_proj[k] - h) / h)
        self.assertGreater(len(rel), 50)
        self.assertLess(max(rel), 0.005)  # < 0.5 % -> < 0.25 m at the 50 m radius


# --------------------------------------------------------------------------- #
# candidate generation
# --------------------------------------------------------------------------- #
class TestCandidateGenerator(unittest.TestCase):
    def _random_network(self, n_arcs=600, seed=0):
        rng = np.random.default_rng(seed)
        start = rng.uniform(0, 3000, (n_arcs, 2))
        length = np.where(rng.random(n_arcs) < 0.1, rng.uniform(300, 1700, n_arcs), rng.uniform(0, 80, n_arcs))
        ang = rng.uniform(0, 2 * np.pi, n_arcs)
        end = start + np.column_stack([np.cos(ang), np.sin(ang)]) * length[:, None]
        return make_geom(np.stack([start, end], axis=1)), rng

    def test_matches_brute_force_on_random_network(self):
        geom, rng = self._random_network()
        px, py = rng.uniform(0, 3000, 400), rng.uniform(0, 3000, 400)
        for r in (30.0, 40.0, 50.0):
            got = generate_candidates(geom, px, py, CandidateConfig(radius_m=r))
            ref = brute_force_candidates(geom, px, py, r)
            self.assertEqual(set(zip(got.point_idx, got.arc_id)), set(zip(ref.point_idx, ref.arc_id)), f"radius {r}")
            m = got.merge(ref, on=["point_idx", "arc_id"], suffixes=("", "_ref"))
            np.testing.assert_allclose(m.dist_m, m.dist_m_ref, atol=1e-3)
            self.assertTrue((got.dist_m <= r).all())

    def test_long_arc_with_far_midpoint_is_found(self):
        # 1.6 km arc; the point is 5 m from its end, 800 m from its midpoint
        geom = make_geom([((0, 0), (1600, 0)), ((0, 300), (10, 300))])
        got = generate_candidates(geom, np.array([1595.0]), np.array([5.0]), CandidateConfig(radius_m=30))
        self.assertEqual(got.arc_id.tolist(), [0])
        self.assertAlmostEqual(float(got.dist_m.iloc[0]), 5.0, places=4)

    def test_sorted_by_point_then_distance_and_cap(self):
        geom = make_geom([((0, y), (100, y)) for y in (2, 5, 9, 14, 20)])
        got = generate_candidates(geom, np.array([50.0, 50.0]), np.array([0.0, 21.0]), CandidateConfig(radius_m=50))
        for _, g in got.groupby("point_idx"):
            self.assertTrue(np.all(np.diff(g.dist_m.to_numpy()) >= 0))
        capped = generate_candidates(geom, np.array([50.0]), np.array([0.0]), CandidateConfig(radius_m=50, max_candidates=2))
        self.assertEqual(capped.arc_id.tolist(), [0, 1])

    def test_no_candidates_and_empty(self):
        geom = make_geom([((0, 0), (10, 0))])
        self.assertEqual(len(generate_candidates(geom, np.array([500.0]), np.array([500.0]))), 0)
        self.assertEqual(len(generate_candidates(geom, np.array([]), np.array([]))), 0)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            CandidateConfig(radius_m=0)
        with self.assertRaises(ValueError):
            CandidateConfig(radius_m=50, corridor_cell_m=40)
        for r in (30, 40, 50):
            self.assertEqual(CandidateConfig(radius_m=r).query_radius_m, r + 5.0)

    @unittest.skipUnless(HAVE_DATA, "raw data not present")
    def test_real_trip_matches_brute_force_and_keeps_local_arc_ids(self):
        trip = load_trip("00000001", base_dir=DATA)
        geom = ArcGeometry.from_trip(trip)
        px, py = trip.track_df.x_m.to_numpy(), trip.track_df.y_m.to_numpy()
        got = generate_candidates(geom, px, py, CandidateConfig(radius_m=50))
        sel = np.arange(0, len(px), 13)
        ref = brute_force_candidates(geom, px[sel], py[sel], 50.0)
        ref["point_idx"] = sel[ref.point_idx.to_numpy()]
        sub = got[got.point_idx.isin(sel)]
        self.assertEqual(set(zip(sub.point_idx, sub.arc_id)), set(zip(ref.point_idx, ref.arc_id)))
        # arc ids are the raw 0-based row numbers of the .arcs file
        raw = np.loadtxt(DATA / "train" / "00000001" / "00000001.arcs", dtype=np.int64)
        a = got.arc_id.to_numpy()
        self.assertTrue(((a >= 0) & (a < len(raw))).all())
        np.testing.assert_array_equal(geom.from_node[a], raw[a, 0])
        np.testing.assert_array_equal(geom.to_node[a], raw[a, 1])


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
class TestFeatures(unittest.TestCase):
    def test_kinematics_straight_line_and_gap_split(self):
        ts = np.r_[np.arange(10.0), np.arange(100.0, 110.0)]  # 90 s gap -> 2 segments
        x = np.r_[np.arange(10.0) * 10.0, 2000 + np.arange(10.0) * 10.0]
        y = np.zeros(20)
        k = compute_point_kinematics(x, y, ts, FeatureConfig(max_gap_s=30))
        self.assertEqual(k.segment_id.tolist(), [0] * 10 + [1] * 10)
        np.testing.assert_allclose(k.speed_mps, 10.0)
        np.testing.assert_allclose(k.heading_deg, 90.0)
        self.assertEqual(int(k.has_prev_point.iloc[10]), 0)  # gap breaks neighbour links
        self.assertTrue(np.isnan(k.dt_prev_s.iloc[10]))

    def test_heading_carried_forward_when_stopped(self):
        x = np.r_[np.arange(5.0) * 10.0, np.full(5, 40.0)]
        y = np.zeros(10)
        k = compute_point_kinematics(x, y, np.arange(10.0))
        self.assertEqual(int(k.heading_valid.iloc[8]), 0)
        self.assertAlmostEqual(float(k.heading_deg.iloc[8]), 90.0)

    def test_directional_and_connectivity_features(self):
        # two-way road along x: nodes 0-1-2 ; arcs 0:0->1, 1:1->0, 2:1->2, 3:2->1
        geom = make_graph_geom([(0, 0), (100, 0), (200, 0)], [(0, 1), (1, 0), (1, 2), (2, 1)])
        graph = build_arc_graph(geom)
        x = np.arange(0.0, 200.0, 10.0) + 5
        y = np.full(x.size, 3.0)
        ts = np.arange(x.size, dtype=float)
        kin = compute_point_kinematics(x, y, ts)
        c = generate_candidates(geom, x, y, CandidateConfig(radius_m=30))
        f = build_candidate_features(c, kin, geom, graph, x, y)
        both = pd.concat([c[["point_idx", "arc_id"]], f], axis=1)
        fwd = both[both.arc_id.isin([0, 2])]
        rev = both[both.arc_id.isin([1, 3])]
        self.assertTrue((fwd.heading_cos > 0.99).all())
        self.assertTrue((rev.heading_cos < -0.99).all())
        self.assertTrue((fwd.along_speed_mps > 9).all() and (rev.along_speed_mps < -9).all())
        self.assertTrue((both.is_two_way == 1).all())
        # point crossing node 1: arc 2 (1->2) is a successor of arc 0 seen at the previous point
        p = int(np.searchsorted(x, 105))
        row = both[(both.point_idx == p) & (both.arc_id == 2)].iloc[0]
        self.assertEqual(int(row.pred_in_prev_cands), 1)
        self.assertLessEqual(int(row.hops_from_prev_anchor), 1)
        self.assertFalse(f.isna().any().any())

    def test_feature_list_is_label_free(self):
        banned = ("label", "route", "true_", "align", "status", "has_positive")
        for c in FEATURE_COLUMNS:
            self.assertFalse(any(b in c for b in banned), c)
        self.assertEqual(set(FEATURE_DTYPES), set(FEATURE_COLUMNS))

    def test_feature_module_never_touches_route(self):
        import src.feature_engineering as live_fe  # the module actually imported (phase-wise location)
        self.assertIn("phase3_features", live_fe.__file__)
        src = Path(live_fe.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + [getattr(node, "module", "") or ""]
                self.assertFalse(any("route_labeler" in n for n in names))
            if isinstance(node, ast.Attribute):
                self.assertNotEqual(node.attr, "route", "feature code must not read trip.route")

    @unittest.skipUnless(HAVE_DATA, "raw data not present")
    def test_features_identical_with_and_without_route(self):
        """Leakage check: removing the route must not change a single feature value."""
        from src.phase3_pipeline import process_trip

        a = process_trip("00000075", with_labels=True)
        trip = load_trip("00000075", base_dir=DATA)
        trip.route = None
        b = process_trip("00000075", with_labels=False, trip=trip)
        pd.testing.assert_frame_equal(a.candidates[FEATURE_COLUMNS], b.candidates[FEATURE_COLUMNS], check_exact=True)
        self.assertNotIn("label", b.candidates.columns)
        self.assertFalse(a.candidates[FEATURE_COLUMNS].isna().any().any())
        for c, dt in FEATURE_DTYPES.items():
            self.assertEqual(a.candidates[c].dtype, np.dtype(dt), c)


# --------------------------------------------------------------------------- #
# labelling
# --------------------------------------------------------------------------- #
class TestLabeling(unittest.TestCase):
    def test_many_points_per_arc_and_skipped_short_arcs(self):
        # chain along x: 50 m, 2 m, 2 m, 50 m, 50 m arcs ; route = all of them in order
        xs = [0, 50, 52, 54, 104, 154]
        geom = make_graph_geom([(v, 0) for v in xs], [(i, i + 1) for i in range(5)])
        route = [0, 1, 2, 3, 4]
        rng = np.random.default_rng(3)
        x = np.r_[np.full(10, 20.0), np.arange(25.0, 150.0, 8.0)]  # 10 stationary points, then 8 m/s
        y = rng.normal(0, 2.0, x.size)
        al, cands, cl, pl = run_labels(geom, route, x + rng.normal(0, 1.0, x.size), y, np.arange(x.size, dtype=float))
        self.assertTrue(np.all(np.diff(al["route_pos"]) >= 0), "alignment must be monotone")
        self.assertEqual(set(al["true_arc_id"][:10]), {0})  # stationary points share one arc
        self.assertGreater(len(x), len(route))  # not 1:1
        exp = np.searchsorted(np.array(xs[1:]), x, side="right").clip(0, 4)
        far = np.min(np.abs(x[:, None] - np.array(xs)[None, :]), axis=1) > 4  # away from junctions
        self.assertGreaterEqual((al["true_arc_id"][far] == exp[far]).mean(), 0.95)
        # exactly one positive per point with a positive
        per_point = pd.concat([cands, cl], axis=1).groupby("point_idx").label.sum()
        self.assertTrue(set(per_point.unique()) <= {0, 1})

    def test_u_turn_on_two_way_road_uses_direction(self):
        # nodes 0-1-2 east; arcs 0:0->1 1:1->2 2:2->1 3:1->0 ; drive east then come back west
        geom = make_graph_geom([(0, 0), (100, 0), (200, 0)], [(0, 1), (1, 2), (2, 1), (1, 0)])
        route = [0, 1, 2, 3]
        x = np.r_[np.arange(5.0, 195.0, 10.0), np.arange(195.0, 5.0, -10.0)]
        y = np.r_[np.full(19, 2.0), np.full(19, -2.0)]
        al, *_ = run_labels(geom, route, x, y, np.arange(x.size, dtype=float))
        out_arcs, back_arcs = al["true_arc_id"][:19], al["true_arc_id"][19:]
        self.assertTrue(set(out_arcs[1:-1]) <= {0, 1})
        self.assertTrue(set(back_arcs[1:-1]) <= {2, 3})
        self.assertTrue(np.all(np.diff(al["route_pos"]) >= 0))

    def test_revisited_arc_gets_two_positions(self):
        # square loop driven twice: route contains every arc twice
        nodes = [(0, 0), (100, 0), (100, 100), (0, 100)]
        geom = make_graph_geom(nodes, [(0, 1), (1, 2), (2, 3), (3, 0)])
        route = [0, 1, 2, 3, 0, 1, 2, 3]
        side = np.arange(5.0, 100.0, 10.0)
        lap = np.r_[np.c_[side, 0 * side], np.c_[100 + 0 * side, side], np.c_[100 - side, 100 + 0 * side], np.c_[0 * side, 100 - side]]
        pts = np.r_[lap, lap]
        al, *_ = run_labels(geom, route, pts[:, 0], pts[:, 1], np.arange(len(pts), dtype=float))
        n = len(lap)
        self.assertTrue(np.all(al["route_pos"][:n] <= 3))
        self.assertTrue(np.all(al["route_pos"][n:] >= 4))

    def test_off_route_tail_is_flagged_not_labelled(self):
        geom = make_graph_geom([(0, 0), (100, 0)], [(0, 1)])
        x = np.r_[np.arange(5.0, 100.0, 10.0), np.full(5, 100.0)]
        y = np.r_[np.zeros(10), np.full(5, 150.0)]  # parked 150 m off the road
        al, cands, cl, pl = run_labels(geom, [0], x, y, np.arange(x.size, dtype=float))
        self.assertTrue((pl.label_status.iloc[10:] == "off_route").all())
        self.assertTrue((pl.has_positive.iloc[10:] == 0).all())
        self.assertTrue((pl.label_status.iloc[:10] != "off_route").all())

    def test_multi_edge_twin_marked_equivalent(self):
        geom = make_graph_geom([(0, 0), (100, 0)], [(0, 1), (0, 1)])  # two arcs, same nodes
        x = np.arange(5.0, 100.0, 10.0)
        al, cands, cl, pl = run_labels(geom, [1], x, np.zeros(x.size), np.arange(x.size, dtype=float))
        both = pd.concat([cands, cl], axis=1)
        self.assertTrue((both[both.arc_id == 1].label == 1).all())
        self.assertTrue((both[both.arc_id == 0].label_equiv == 1).all())
        self.assertTrue((both[both.arc_id == 0].label == 0).all())


# --------------------------------------------------------------------------- #
# partitions / generated dataset
# --------------------------------------------------------------------------- #
class TestPartitions(unittest.TestCase):
    @unittest.skipUnless(HAVE_DATA, "raw data not present")
    def test_train_and_student_test_disjoint(self):
        train = {p.name for p in (DATA / "train").iterdir() if p.is_dir()}
        test = {p.name for p in (DATA / "student_test").iterdir() if p.is_dir()}
        self.assertEqual(len(train), 80)
        self.assertEqual(len(test), 10)
        self.assertEqual(train & test, set())


@unittest.skipUnless((PROCESSED / "train_candidates.parquet").exists(), "phase-3 dataset not built yet")
class TestBuiltDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pyarrow.parquet as pq

        cols = ["trip_id", "point_idx", "arc_id", "label", "label_equiv"]
        cls.train = pq.read_table(PROCESSED / "train_candidates.parquet", columns=cols).to_pandas()
        cls.test = pq.read_table(PROCESSED / "student_test_candidates.parquet", columns=cols).to_pandas()
        cls.schema = pq.read_schema(PROCESSED / "train_candidates.parquet")

    def test_trip_counts_and_disjoint(self):
        self.assertEqual(self.train.trip_id.nunique(), 80)
        self.assertEqual(self.test.trip_id.nunique(), 10)
        self.assertEqual(set(self.train.trip_id) & set(self.test.trip_id), set())

    def test_at_most_one_positive_per_point(self):
        for df in (self.train, self.test):
            s = df.groupby(["trip_id", "point_idx"], observed=True).label.sum()
            self.assertLessEqual(int(s.max()), 1)
            self.assertFalse(((df.label == 1) & (df.label_equiv == 1)).any())

    def test_no_duplicate_rows_and_all_features_present(self):
        for df in (self.train, self.test):
            self.assertFalse(df.duplicated(["trip_id", "point_idx", "arc_id"]).any())
        for c in FEATURE_COLUMNS:
            self.assertIn(c, self.schema.names)

    def test_loader_labelled_only_has_one_positive_per_point(self):
        from src.phase3_dataset import load_candidates

        df = load_candidates("student_test", columns=["trip_id", "point_idx", "label", "label_status"])
        self.assertTrue(set(df.label_status.astype(str).unique()) <= {"ok", "boundary"})
        s = df.groupby(["trip_id", "point_idx"], observed=True).label.sum()
        self.assertTrue((s == 1).all())

    def test_no_nulls_in_features(self):
        import pyarrow.parquet as pq

        meta = pq.ParquetFile(PROCESSED / "train_candidates.parquet").metadata
        names = self.schema.names
        for rg in range(meta.num_row_groups):
            for ci in range(meta.num_columns):
                if names[ci] in FEATURE_COLUMNS:
                    st = meta.row_group(rg).column(ci).statistics
                    self.assertTrue(st is None or st.null_count == 0, names[ci])


if __name__ == "__main__":
    unittest.main(verbosity=2)
