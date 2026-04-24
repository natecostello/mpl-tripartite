"""Tests for mpl-tripartite."""

from __future__ import annotations

import io
import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mpl_tripartite import TripartiteAxes, TripartiteProjection
from mpl_tripartite._helpers import (
    G0_IN,
    G0_SI,
    _base_len_from_v_unit,
    _clip_segment_to_rect,
    _decade_level_sets,
)

# ── helpers ───────────────────────────────────────────────────────────


class TestBaseLenFromVUnit:
    def test_metric(self):
        for u in ("m/s", "ms", "mps", "M/S"):
            assert _base_len_from_v_unit(u) == "m"

    def test_imperial(self):
        for u in ("in/s", "ips", "IN/S"):
            assert _base_len_from_v_unit(u) == "in"

    def test_invalid(self):
        with pytest.raises(ValueError):
            _base_len_from_v_unit("ft/s")


class TestDecadeLevelSets:
    def test_single_decade(self):
        maj, all_ = _decade_level_sets(1.0, 10.0)
        assert 1.0 in maj
        assert 10.0 in maj
        assert len(all_) == 9 * 2  # k=0 and k=1

    def test_range_10_1000(self):
        maj, all_ = _decade_level_sets(10.0, 1000.0)
        assert 10.0 in maj
        assert 100.0 in maj
        assert 1000.0 in maj

    def test_empty_on_nan(self):
        maj, all_ = _decade_level_sets(float("nan"), 10.0)
        assert maj == []
        assert all_ == []

    def test_cap_extreme_range(self):
        maj, _ = _decade_level_sets(1e-50, 1e50)
        # Should not explode; capped
        assert len(maj) <= 100


class TestClipSegment:
    def test_fully_inside(self):
        result = _clip_segment_to_rect(1, 1, 2, 2, 0, 3, 0, 3)
        assert result == (1, 1, 2, 2)

    def test_fully_outside(self):
        result = _clip_segment_to_rect(4, 4, 5, 5, 0, 3, 0, 3)
        assert result is None

    def test_partial_clip(self):
        # Horizontal line y=1.5 from x=-1 to x=4, rect [0,3]×[0,3]
        result = _clip_segment_to_rect(-1, 1.5, 4, 1.5, 0, 3, 0, 3)
        assert result is not None
        x0, y0, x1, y1 = result
        assert abs(x0 - 0) < 1e-10
        assert abs(x1 - 3) < 1e-10
        assert abs(y0 - 1.5) < 1e-10

    def test_diagonal_clip(self):
        # Line from (0,0) to (4,4), rect [1,3]×[1,3]
        result = _clip_segment_to_rect(0, 0, 4, 4, 1, 3, 1, 3)
        assert result is not None
        x0, y0, x1, y1 = result
        assert abs(x0 - 1) < 1e-10
        assert abs(y0 - 1) < 1e-10
        assert abs(x1 - 3) < 1e-10
        assert abs(y1 - 3) < 1e-10


# ── registration & construction ───────────────────────────────────────


class TestRegistration:
    def test_projection_registered(self):
        fig, ax = plt.subplots(subplot_kw=dict(projection="tripartite"))
        assert isinstance(ax, TripartiteAxes)
        plt.close(fig)

    def test_log_scales(self):
        fig, ax = plt.subplots(subplot_kw=dict(projection="tripartite"))
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"
        plt.close(fig)

    def test_v_unit_in_s(self):
        fig, ax = plt.subplots(subplot_kw=dict(projection="tripartite"))
        assert ax._tri_v_unit == "in/s"
        assert ax._tri_g0 == G0_IN
        plt.close(fig)

    def test_v_unit_m_s(self):
        fig, ax = plt.subplots(subplot_kw=dict(projection=TripartiteProjection(v_unit="m/s")))
        assert isinstance(ax, TripartiteAxes)
        assert ax._tri_v_unit == "m/s"
        assert ax._tri_g0 == G0_SI
        plt.close(fig)

    def test_tripartite_projection_helper(self):
        proj = TripartiteProjection(v_unit="m/s", diag_grid=False)
        cls, kw = proj._as_mpl_axes()
        assert cls is TripartiteAxes
        assert kw["v_unit"] == "m/s"
        assert kw["diag_grid"] is False


# ── rendering smoke tests ─────────────────────────────────────────────


class TestRendering:
    def test_empty_draw(self):
        fig, ax = plt.subplots(subplot_kw=dict(projection="tripartite"))
        ax.set_xlim(10, 10000)
        ax.set_ylim(0.1, 100)
        fig.canvas.draw()  # should not raise
        plt.close(fig)

    def test_draw_with_data(self):
        fig, ax = plt.subplots(subplot_kw=dict(projection="tripartite"))
        freq = np.logspace(1, 4, 100)
        pv = 20 / (2 * np.pi * freq)
        ax.loglog(freq, pv)
        ax.set_xlim(10, 10000)
        ax.set_ylim(0.1, 100)
        fig.canvas.draw()
        plt.close(fig)

    def test_savefig_png(self):
        fig, ax = plt.subplots(subplot_kw=dict(projection="tripartite"))
        freq = np.logspace(1, 4, 100)
        pv = 20 / (2 * np.pi * freq)
        ax.loglog(freq, pv)
        ax.set_xlim(10, 10000)
        ax.set_ylim(0.1, 100)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        assert buf.tell() > 0  # something was written
        plt.close(fig)

    def test_diag_grid_false(self):
        fig, ax = plt.subplots(subplot_kw=dict(projection=TripartiteProjection(diag_grid=False)))
        ax.set_xlim(10, 10000)
        ax.set_ylim(0.1, 100)
        fig.canvas.draw()
        plt.close(fig)

    def test_metric_units_render(self):
        fig, ax = plt.subplots(subplot_kw=dict(projection=TripartiteProjection(v_unit="m/s")))
        freq = np.logspace(0, 3, 50)
        pv = 0.5 / (2 * np.pi * freq)
        ax.loglog(freq, pv)
        ax.set_xlim(1, 1000)
        ax.set_ylim(1e-4, 1)
        fig.canvas.draw()
        plt.close(fig)


# ── math validation ──────────────────────────────────────────────────


class TestMath:
    def test_display_angle_square(self):
        """For a square figure with equal log ranges, slope ±1 → ~±45°."""
        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(projection="tripartite"))
        ax.set_xlim(1, 1000)  # 3 decades
        ax.set_ylim(1, 1000)  # 3 decades
        fig.canvas.draw()
        ang = ax._display_angle(1.0)
        # With equal log ranges and roughly square bbox, expect close to 45
        assert 30 < abs(ang) < 60, f"angle was {ang}"
        plt.close(fig)

    def test_display_angle_neg(self):
        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(projection="tripartite"))
        ax.set_xlim(1, 1000)
        ax.set_ylim(1, 1000)
        fig.canvas.draw()
        ang_pos = ax._display_angle(+1.0)
        ang_neg = ax._display_angle(-1.0)
        assert abs(ang_pos + ang_neg) < 1e-6  # symmetric
        plt.close(fig)

    def test_format_coord(self):
        fig, ax = plt.subplots(subplot_kw=dict(projection="tripartite"))
        # At f=100 Hz, PV=10 in/s:
        # d = v/(2πf) = 10/(200π) ≈ 0.01592 in
        # a = 2πf·v/g0 = 2π·100·10/386.089 ≈ 16.27 g
        s = ax.format_coord(100, 10)
        assert "f=" in s
        assert "PV=" in s
        assert "d=" in s
        assert "a=" in s
        assert "in/s" in s

        # Check numeric accuracy
        TWO_PI = 2 * math.pi
        expected_d = 10 / (TWO_PI * 100)
        expected_a = TWO_PI * 100 * 10 / G0_IN
        assert f"d={expected_d:.4g}" in s
        assert f"a={expected_a:.4g}" in s
        plt.close(fig)

    def test_format_coord_zero(self):
        fig, ax = plt.subplots(subplot_kw=dict(projection="tripartite"))
        assert ax.format_coord(0, 10) == ""
        assert ax.format_coord(10, 0) == ""
        plt.close(fig)


# ── edge cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    def test_narrow_range_1_decade(self):
        fig, ax = plt.subplots(subplot_kw=dict(projection="tripartite"))
        ax.set_xlim(100, 1000)
        ax.set_ylim(1, 10)
        fig.canvas.draw()
        plt.close(fig)

    def test_wide_range_6_decades(self):
        fig, ax = plt.subplots(subplot_kw=dict(projection="tripartite"))
        ax.set_xlim(0.1, 100000)
        ax.set_ylim(0.001, 1000)
        fig.canvas.draw()
        plt.close(fig)

    def test_set_v_unit_relabels(self):
        fig, ax = plt.subplots(
            subplot_kw=dict(
                projection=TripartiteProjection(
                    v_unit="in/s",
                    ylabel="Velocity (in/s)",
                )
            )
        )
        assert "in/s" in ax.get_ylabel()
        ax.set_v_unit("m/s")
        assert ax._tri_base_len == "m"
        # Custom ylabel is preserved by set_v_unit
        assert ax.get_ylabel() == "Velocity (in/s)"
        plt.close(fig)

    def test_set_diag_grid_toggle(self):
        fig, ax = plt.subplots(subplot_kw=dict(projection="tripartite"))
        ax.set_xlim(10, 10000)
        ax.set_ylim(0.1, 100)
        ax.set_diag_grid(False)
        fig.canvas.draw()
        ax.set_diag_grid(True)
        fig.canvas.draw()
        plt.close(fig)

    def test_set_diag_style(self):
        fig, ax = plt.subplots(subplot_kw=dict(projection="tripartite"))
        ax.set_xlim(10, 10000)
        ax.set_ylim(0.1, 100)
        ax.set_diag_style(colors="red", linewidths=1.0)
        fig.canvas.draw()
        plt.close(fig)

    def test_multiple_savefigs(self):
        """Transient artists don't accumulate across draws."""
        fig, ax = plt.subplots(subplot_kw=dict(projection="tripartite"))
        ax.set_xlim(10, 10000)
        ax.set_ylim(0.1, 100)
        n_before = len(ax._children)
        for _ in range(3):
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
        n_after = len(ax._children)
        assert n_after == n_before, "Transient artists leaked into _children"
        plt.close(fig)


# ── configurable labels ──────────────────────────────────────────────


class TestCustomLabels:
    def test_custom_xlabel_ylabel(self):
        """Custom xlabel/ylabel appear on axes."""
        fig, ax = plt.subplots(
            subplot_kw=dict(
                projection=TripartiteProjection(
                    v_unit="m/s",
                    xlabel="Period (s)",
                    ylabel="Sv (m/s)",
                )
            )
        )
        assert ax.get_xlabel() == "Period (s)"
        assert ax.get_ylabel() == "Sv (m/s)"
        plt.close(fig)

    def test_custom_diag_labels_render(self):
        """Custom neg/pos diagonal labels render without error."""
        fig, ax = plt.subplots(
            subplot_kw=dict(
                projection=TripartiteProjection(
                    v_unit="m/s",
                    neg_diag_label="m/s²",
                    pos_diag_label="m",
                    g_normalize=False,
                )
            )
        )
        ax.set_xlim(1, 1000)
        ax.set_ylim(1e-4, 10)
        fig.canvas.draw()  # must not raise
        plt.close(fig)

    def test_default_no_labels(self):
        """Default projection does not set axis labels."""
        fig, ax = plt.subplots(subplot_kw=dict(projection="tripartite"))
        assert ax.get_xlabel() == ""
        assert ax.get_ylabel() == ""
        plt.close(fig)

    def test_format_coord_custom_labels(self):
        """format_coord uses custom ylabel stem and neg/pos labels."""
        fig, ax = plt.subplots(
            subplot_kw=dict(
                projection=TripartiteProjection(
                    v_unit="m/s",
                    ylabel="Sv (m/s)",
                    neg_diag_label="m/s²",
                    pos_diag_label="m",
                    g_normalize=False,
                )
            )
        )
        s = ax.format_coord(100, 1.0)
        assert "Sv=" in s
        assert "m/s²" in s
        assert "m/s" in s  # v_unit
        assert "d=" in s
        plt.close(fig)

    def test_set_v_unit_preserves_custom_ylabel(self):
        """set_v_unit keeps custom ylabel when one was provided."""
        fig, ax = plt.subplots(
            subplot_kw=dict(
                projection=TripartiteProjection(
                    v_unit="in/s",
                    ylabel="Velocity (in/s)",
                )
            )
        )
        assert ax.get_ylabel() == "Velocity (in/s)"
        ax.set_v_unit("m/s")
        # Custom ylabel is preserved (not overwritten with "Pseudo-Velocity")
        assert ax.get_ylabel() == "Velocity (in/s)"
        plt.close(fig)


class TestGNormalize:
    def test_g_normalize_false_format_coord(self):
        """With g_normalize=False, accel is raw (not divided by g0)."""
        fig, ax = plt.subplots(
            subplot_kw=dict(
                projection=TripartiteProjection(
                    v_unit="m/s",
                    neg_diag_label="m/s²",
                    g_normalize=False,
                )
            )
        )
        TWO_PI = 2 * math.pi
        # At f=100 Hz, v=1.0 m/s: raw accel = 2π·100·1.0 = 628.32 m/s²
        s = ax.format_coord(100, 1.0)
        expected_a = TWO_PI * 100 * 1.0
        assert f"a={expected_a:.4g}" in s
        assert "m/s²" in s
        plt.close(fig)

    def test_g_normalize_true_format_coord(self):
        """With g_normalize=True (default), accel is divided by g0."""
        fig, ax = plt.subplots(subplot_kw=dict(projection=TripartiteProjection(v_unit="m/s")))
        TWO_PI = 2 * math.pi
        s = ax.format_coord(100, 1.0)
        expected_a = TWO_PI * 100 * 1.0 / G0_SI
        assert f"a={expected_a:.4g}" in s
        plt.close(fig)

    def test_g_normalize_false_renders(self):
        """g_normalize=False renders diagonal grid without error."""
        fig, ax = plt.subplots(
            subplot_kw=dict(
                projection=TripartiteProjection(
                    v_unit="m/s",
                    g_normalize=False,
                )
            )
        )
        ax.set_xlim(1, 1000)
        ax.set_ylim(1e-4, 10)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        assert buf.tell() > 0
        plt.close(fig)


class TestProjectionPassthrough:
    def test_all_kwargs_reach_axes(self):
        """TripartiteProjection forwards all custom kwargs to TripartiteAxes."""
        proj = TripartiteProjection(
            v_unit="m/s",
            diag_grid=False,
            xlabel="Period (s)",
            ylabel="Sv (m/s)",
            neg_diag_label="m/s²",
            pos_diag_label="m",
            g_normalize=False,
        )
        cls, kw = proj._as_mpl_axes()
        assert cls is TripartiteAxes
        assert kw["v_unit"] == "m/s"
        assert kw["diag_grid"] is False
        assert kw["xlabel"] == "Period (s)"
        assert kw["ylabel"] == "Sv (m/s)"
        assert kw["neg_diag_label"] == "m/s²"
        assert kw["pos_diag_label"] == "m"
        assert kw["g_normalize"] is False

    def test_projection_creates_axes_with_custom_labels(self):
        """Full round-trip: projection → subplots → axes with correct labels."""
        fig, ax = plt.subplots(
            subplot_kw=dict(
                projection=TripartiteProjection(
                    v_unit="m/s",
                    xlabel="Period (s)",
                    ylabel="Sv (m/s)",
                    neg_diag_label="m/s²",
                    pos_diag_label="m",
                    g_normalize=False,
                )
            )
        )
        assert isinstance(ax, TripartiteAxes)
        assert ax.get_xlabel() == "Period (s)"
        assert ax.get_ylabel() == "Sv (m/s)"
        assert ax._tri_neg_diag_label == "m/s²"
        assert ax._tri_pos_diag_label == "m"
        assert ax._tri_g_normalize is False
        plt.close(fig)
