"""Tests for position_config_gen.py — pure-function unit tests.

Covers: _mean, _stdev, _statistical_aggregate, _mode_count,
        _simple_kmeans, _cluster_multi_instance.
Does NOT require ultralytics (PositionConfigGenerator.generate is skipped
when YOLO is None, which is always the case during pytest).
"""

import logging
import math

import pytest

from picture_tool.position.position_config_gen import (
    _cluster_multi_instance,
    _mean,
    _mode_count,
    _simple_kmeans,
    _statistical_aggregate,
    _stdev,
)


# ---------------------------------------------------------------------------
# _mean / _stdev
# ---------------------------------------------------------------------------

class TestMean:
    def test_single_value(self):
        assert _mean([5.0]) == 5.0

    def test_multiple_values(self):
        assert _mean([2.0, 4.0, 6.0]) == pytest.approx(4.0)

    def test_negative_values(self):
        assert _mean([-10.0, 10.0]) == pytest.approx(0.0)


class TestStdev:
    def test_single_value_returns_zero(self):
        assert _stdev([42.0]) == 0.0

    def test_identical_values(self):
        assert _stdev([5.0, 5.0, 5.0]) == 0.0

    def test_known_values(self):
        # Sample stdev of [2, 4, 4, 4, 5, 5, 7, 9] = 2.138...
        vals = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        expected = math.sqrt(sum((v - 5.0) ** 2 for v in vals) / 7)
        assert _stdev(vals) == pytest.approx(expected, rel=1e-6)

    def test_two_values(self):
        # stdev([0, 10]) = sqrt((25+25)/1) = sqrt(50) ≈ 7.071
        assert _stdev([0.0, 10.0]) == pytest.approx(math.sqrt(50), rel=1e-6)


# ---------------------------------------------------------------------------
# _statistical_aggregate
# ---------------------------------------------------------------------------

class TestStatisticalAggregate:
    def test_single_box(self):
        boxes = [[100, 200, 300, 400]]
        result = _statistical_aggregate(boxes)
        # center = (200, 300), w=200, h=200
        assert result["cx"] == 200.0
        assert result["cy"] == 300.0
        assert result["x1"] == 100
        assert result["y1"] == 200
        assert result["x2"] == 300
        assert result["y2"] == 400
        assert result["sigma_cx"] == 0.0
        assert result["sigma_cy"] == 0.0
        assert result["count"] == 1

    def test_two_identical_boxes(self):
        boxes = [[10, 20, 30, 40], [10, 20, 30, 40]]
        result = _statistical_aggregate(boxes)
        assert result["cx"] == 20.0
        assert result["cy"] == 30.0
        assert result["sigma_cx"] == 0.0
        assert result["sigma_cy"] == 0.0
        assert result["count"] == 2

    def test_multiple_boxes_mean_center(self):
        boxes = [
            [0, 0, 100, 100],   # center (50, 50), w=100, h=100
            [20, 20, 120, 120], # center (70, 70), w=100, h=100
        ]
        result = _statistical_aggregate(boxes)
        assert result["cx"] == pytest.approx(60.0)
        assert result["cy"] == pytest.approx(60.0)
        # mean_w=100, mean_h=100 → x1=60-50=10, y1=10, x2=110, y2=110
        assert result["x1"] == 10
        assert result["y1"] == 10
        assert result["x2"] == 110
        assert result["y2"] == 110
        assert result["count"] == 2

    def test_sigma_nonzero(self):
        boxes = [
            [0, 0, 10, 10],     # center (5, 5)
            [100, 100, 110, 110], # center (105, 105)
        ]
        result = _statistical_aggregate(boxes)
        # σ of [5, 105] = sqrt((50^2+50^2)/1) = 70.71
        assert result["sigma_cx"] > 0
        assert result["sigma_cy"] > 0

    def test_varying_sizes(self):
        boxes = [
            [0, 0, 100, 50],   # w=100, h=50, center (50, 25)
            [0, 0, 200, 100],  # w=200, h=100, center (100, 50)
        ]
        result = _statistical_aggregate(boxes)
        # mean center = (75, 37.5), mean_w=150, mean_h=75
        assert result["cx"] == 75.0
        assert result["cy"] == 37.5
        assert result["x1"] == 0   # 75 - 75 = 0
        assert result["y1"] == 0   # 37.5 - 37.5 = 0
        assert result["x2"] == 150  # 75 + 75 = 150
        assert result["y2"] == 75   # 37.5 + 37.5 = 75


# ---------------------------------------------------------------------------
# _mode_count
# ---------------------------------------------------------------------------

class TestModeCount:
    def test_empty_returns_one(self):
        assert _mode_count([]) == 1

    def test_single_value(self):
        assert _mode_count([3]) == 3

    def test_uniform(self):
        assert _mode_count([2, 2, 2]) == 2

    def test_mixed_picks_most_common(self):
        assert _mode_count([1, 2, 2, 3]) == 2

    def test_tie_returns_first_most_common(self):
        # Counter.most_common picks arbitrary on tie; just verify it's one of the tied values
        result = _mode_count([1, 1, 2, 2])
        assert result in (1, 2)


# ---------------------------------------------------------------------------
# _simple_kmeans
# ---------------------------------------------------------------------------

class TestSimpleKmeans:
    def test_k_zero_returns_single_group(self):
        points = [(0.0, 0.0), (10.0, 10.0)]
        groups = _simple_kmeans(points, 0)
        assert len(groups) == 1
        assert sorted(groups[0]) == [0, 1]

    def test_k_greater_than_n_returns_single_group(self):
        points = [(1.0, 1.0)]
        groups = _simple_kmeans(points, 5)
        assert len(groups) == 1

    def test_k_one(self):
        points = [(0.0, 0.0), (100.0, 100.0)]
        groups = _simple_kmeans(points, 1)
        assert len(groups) == 1
        assert sorted(groups[0]) == [0, 1]

    def test_two_clear_clusters(self):
        points = [
            (0.0, 0.0), (1.0, 1.0), (2.0, 0.0),    # cluster A near origin
            (100.0, 100.0), (101.0, 101.0), (99.0, 100.0),  # cluster B far away
        ]
        groups = _simple_kmeans(points, 2)
        assert len(groups) == 2
        # Each group should have exactly 3 points
        sizes = sorted(len(g) for g in groups)
        assert sizes == [3, 3]
        # Verify cluster separation
        for group in groups:
            xs = [points[i][0] for i in group]
            # All points in a group should be within same region
            assert max(xs) - min(xs) < 50

    def test_three_clusters(self):
        points = [
            (0.0, 0.0), (1.0, 0.0),
            (50.0, 50.0), (51.0, 50.0),
            (100.0, 0.0), (101.0, 0.0),
        ]
        groups = _simple_kmeans(points, 3)
        assert len(groups) == 3
        all_indices = sorted(idx for g in groups for idx in g)
        assert all_indices == [0, 1, 2, 3, 4, 5]

    def test_convergence_identical_points(self):
        """All identical points → all in same cluster."""
        points = [(5.0, 5.0)] * 10
        groups = _simple_kmeans(points, 2)
        # One group should have all 10, the other should be empty
        sizes = sorted(len(g) for g in groups)
        assert sizes[0] == 0
        assert sizes[1] == 10

    def test_single_point(self):
        groups = _simple_kmeans([(42.0, 42.0)], 1)
        assert groups == [[0]]


# ---------------------------------------------------------------------------
# _cluster_multi_instance
# ---------------------------------------------------------------------------

class TestClusterMultiInstance:
    def _logger(self):
        return logging.getLogger("test_cluster")

    def test_mode_one_returns_plain_key(self):
        """When mode count is 1, returns plain class name (no #N)."""
        boxes = [[10, 10, 20, 20], [12, 12, 22, 22]]
        counts = [1, 1, 1]  # mode = 1
        result = _cluster_multi_instance(boxes, counts, self._logger(), "LED")
        assert "LED" in result
        assert "LED#0" not in result

    def test_two_clusters_produces_indexed_keys(self):
        boxes = [
            # Cluster near (15, 15)
            [10, 10, 20, 20], [11, 11, 21, 21], [12, 12, 22, 22],
            # Cluster near (105, 105)
            [100, 100, 110, 110], [101, 101, 111, 111], [102, 102, 112, 112],
        ]
        counts = [2, 2, 2]  # mode = 2
        result = _cluster_multi_instance(boxes, counts, self._logger(), "Black")
        assert len(result) == 2
        assert "Black#0" in result
        assert "Black#1" in result

    def test_indexed_keys_sorted_by_x_then_y(self):
        """#0 should be the leftmost cluster."""
        boxes = [
            # Cluster B (right)
            [200, 200, 210, 210], [201, 201, 211, 211],
            # Cluster A (left)
            [10, 10, 20, 20], [11, 11, 21, 21],
        ]
        counts = [2, 2]
        result = _cluster_multi_instance(boxes, counts, self._logger(), "X")
        keys = list(result.keys())
        # X#0 should be the left cluster (cx ≈ 15)
        assert result["X#0"]["cx"] < result["X#1"]["cx"]

    def test_each_cluster_has_statistical_fields(self):
        boxes = [
            [10, 10, 20, 20], [12, 12, 22, 22],
            [100, 100, 110, 110], [102, 102, 112, 112],
        ]
        counts = [2, 2]
        result = _cluster_multi_instance(boxes, counts, self._logger(), "Z")
        for key in result:
            stats = result[key]
            assert "cx" in stats
            assert "cy" in stats
            assert "sigma_cx" in stats
            assert "sigma_cy" in stats
            assert "count" in stats
            assert "x1" in stats and "x2" in stats

    def test_high_variance_counts_warns(self, caplog):
        """When per-image counts vary a lot, should log a warning."""
        boxes = [[10, 10, 20, 20]] * 6
        counts = [2, 2, 5, 1, 2]  # σ > 1.0, len > 2
        with caplog.at_level(logging.WARNING, logger="test_cluster"):
            _cluster_multi_instance(boxes, counts, self._logger(), "Noisy")
        assert any("varies significantly" in r.message for r in caplog.records)

    def test_low_variance_counts_no_warning(self, caplog):
        boxes = [[10, 10, 20, 20]] * 4 + [[100, 100, 110, 110]] * 4
        counts = [2, 2, 2, 2]  # σ = 0
        with caplog.at_level(logging.WARNING, logger="test_cluster"):
            _cluster_multi_instance(boxes, counts, self._logger(), "Stable")
        assert not any("varies significantly" in r.message for r in caplog.records)

    def test_three_instances(self):
        boxes = [
            [10, 10, 20, 20], [11, 11, 21, 21],
            [100, 100, 110, 110], [101, 101, 111, 111],
            [200, 200, 210, 210], [201, 201, 211, 211],
        ]
        counts = [3, 3]
        result = _cluster_multi_instance(boxes, counts, self._logger(), "T")
        assert len(result) == 3
        assert "T#0" in result
        assert "T#1" in result
        assert "T#2" in result
