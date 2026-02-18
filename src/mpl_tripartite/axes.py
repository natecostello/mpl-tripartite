"""TripartiteAxes — a matplotlib projection for four-coordinate PV SRS plots."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.projections import register_projection
from matplotlib.text import Text

from ._helpers import (
    G0_IN,
    G0_SI,
    _base_len_from_v_unit,
    _clip_segment_to_rect,
    _decade_level_sets,
)

# ── Default styling ──────────────────────────────────────────────────

_DIAG_LINE_KW: Dict[str, Any] = dict(
    colors="0.8", linewidths=0.6, linestyles=":", zorder=0.5
)
_DIAG_LABEL_KW: Dict[str, Any] = dict(color="0.35", fontsize=8, va="center", zorder=2.5)
_LABEL_NUDGE_PX = 10.0  # pixels above the line


class TripartiteAxes(Axes):
    """Matplotlib axes with native diagonal gridlines for tripartite SRS plots.

    Register with ``import mpl_tripartite`` then use
    ``projection='tripartite'`` in ``plt.subplots``.
    """

    name = "tripartite"

    def __init__(
        self,
        fig,
        *args,
        v_unit: str = "in/s",
        diag_grid: bool = True,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        neg_diag_label: str = "g",
        pos_diag_label: Optional[str] = None,
        g_normalize: bool = True,
        **kwargs,
    ):
        self._tri_v_unit = v_unit
        self._tri_diag_grid = diag_grid
        self._tri_diag_style: Dict[str, Any] = {}
        self._tri_label_style: Dict[str, Any] = {}
        # Validate unit early
        base_len = _base_len_from_v_unit(v_unit)
        self._tri_g0 = G0_SI if base_len == "m" else G0_IN
        self._tri_base_len = base_len
        # Configurable labels (defaults match legacy SRS behavior)
        self._tri_xlabel = xlabel
        self._tri_ylabel = ylabel  # None → auto from v_unit
        self._tri_neg_diag_label = neg_diag_label
        self._tri_pos_diag_label = pos_diag_label  # None → auto from base_len
        self._tri_g_normalize = g_normalize
        super().__init__(fig, *args, **kwargs)

    # ── lifecycle ─────────────────────────────────────────────────────

    def clear(self):
        super().clear()
        self.set_xscale("log")
        self.set_yscale("log")
        if self._tri_xlabel is not None:
            self.set_xlabel(self._tri_xlabel)
        if self._tri_ylabel is not None:
            self.set_ylabel(self._tri_ylabel)
        self.grid(True, which="both", ls=":", lw=0.5, alpha=0.4)

    # ── draw override: inject diagonal artists per frame ──────────────

    def draw(self, renderer):
        if self._tri_diag_grid:
            artists = self._make_diagonal_grid_artists()
            for a in artists:
                if isinstance(a, LineCollection):
                    self.add_collection(a)
                else:
                    self.add_artist(a)
            try:
                super().draw(renderer)
            finally:
                for a in artists:
                    try:
                        a.remove()
                    except Exception:
                        pass
        else:
            super().draw(renderer)

    # ── diagonal grid construction ────────────────────────────────────

    def _make_diagonal_grid_artists(self) -> List:
        fmin, fmax = self.get_xlim()
        vmin, vmax = self.get_ylim()
        if fmin <= 0 or fmax <= 0 or vmin <= 0 or vmax <= 0:
            return []

        TWO_PI = 2.0 * math.pi
        g0 = self._tri_g0
        base_len = self._tri_base_len
        disp_unit = "m" if base_len == "m" else "in"

        # Acceleration envelope (a = 2πfv)
        a_corners = [TWO_PI * f * v for f in (fmin, fmax) for v in (vmin, vmax)]
        a_min, a_max = min(a_corners), max(a_corners)
        if self._tri_g_normalize:
            acc_maj_G, acc_all_G = _decade_level_sets(a_min / g0, a_max / g0)
            acc_all = [g * g0 for g in acc_all_G]
            acc_labels = [(g * g0, int(round(math.log10(g)))) for g in acc_maj_G]
        else:
            acc_maj, acc_all = _decade_level_sets(a_min, a_max)
            acc_labels = [(a, int(round(math.log10(a)))) for a in acc_maj]

        # Displacement envelope (d = v / 2πf)
        d_corners = [v / (TWO_PI * f) for f in (fmin, fmax) for v in (vmin, vmax)]
        d_min, d_max = min(d_corners), max(d_corners)
        d_maj, d_all = _decade_level_sets(d_min, d_max)
        disp_labels = [(d, int(round(math.log10(d)))) for d in d_maj]

        # Merge user style overrides
        line_kw = {**_DIAG_LINE_KW, **self._tri_diag_style}
        label_kw = {**_DIAG_LABEL_KW, **self._tri_label_style}

        artists: List = []

        # ── acceleration segments (slope -1 in log-log) ──────────────
        acc_segs = []
        for a in acc_all:
            seg = self._make_diagonal_segment(a, -1, fmin, fmax, vmin, vmax)
            if seg is not None:
                acc_segs.append(seg)
        if acc_segs:
            lc = LineCollection(acc_segs, **line_kw)
            artists.append(lc)

        # ── displacement segments (slope +1 in log-log) ──────────────
        disp_segs = []
        for d in d_all:
            seg = self._make_diagonal_segment(d, +1, fmin, fmax, vmin, vmax)
            if seg is not None:
                disp_segs.append(seg)
        if disp_segs:
            lc = LineCollection(disp_segs, **line_kw)
            artists.append(lc)

        # ── acceleration labels (right edge) ─────────────────────────
        log_frange = math.log10(fmax) - math.log10(fmin)
        log_vrange = math.log10(vmax) - math.log10(vmin)
        f_right = 10 ** (math.log10(fmax) - 0.02 * log_frange)
        f_left = 10 ** (math.log10(fmin) + 0.02 * log_frange)

        ang_neg = self._display_angle(-1.0)
        ang_pos = self._display_angle(+1.0)

        for a, n_exp in acc_labels:
            v_at = a / (TWO_PI * f_right)
            if not (vmin <= v_at <= vmax):
                continue
            v_nudged = self._nudge_up(f_right, v_at, _LABEL_NUDGE_PX)
            if not (vmin <= v_nudged <= vmax):
                continue
            neg_label = self._tri_neg_diag_label
            txt = Text(
                f_right,
                v_nudged,
                rf"$10^{{{n_exp}}}$ {neg_label}",
                rotation=ang_neg,
                ha="right",
                transform=self.transData,
                **label_kw,
            )
            txt.set_transform_rotates_text(False)
            txt.set_rotation_mode("anchor")
            artists.append(txt)

        # ── displacement labels (left edge) ──────────────────────────
        for d, n_exp in disp_labels:
            v_at = TWO_PI * f_left * d
            if not (vmin <= v_at <= vmax):
                continue
            v_nudged = self._nudge_up(f_left, v_at, _LABEL_NUDGE_PX)
            if not (vmin <= v_nudged <= vmax):
                continue
            pos_label = (
                self._tri_pos_diag_label
                if self._tri_pos_diag_label is not None
                else disp_unit
            )
            txt = Text(
                f_left,
                v_nudged,
                rf"$10^{{{n_exp}}}$ {pos_label}",
                rotation=ang_pos,
                ha="left",
                transform=self.transData,
                **label_kw,
            )
            txt.set_transform_rotates_text(False)
            txt.set_rotation_mode("anchor")
            artists.append(txt)

        return artists

    # ── geometry helpers ──────────────────────────────────────────────

    @staticmethod
    def _make_diagonal_segment(
        constant: float,
        slope: int,
        fmin: float,
        fmax: float,
        vmin: float,
        vmax: float,
        n_pts: int = 60,
    ) -> Optional[list]:
        """Return polyline points for one diagonal, clipped to view, or None.

        Uses *n_pts* log-spaced points so the line follows the true curve
        on log-log axes (a 2-point segment would render as a straight line
        in data space, which bows away from the diagonal on a log scale).
        """
        TWO_PI = 2.0 * math.pi
        try:
            lf0, lf1 = math.log10(fmin), math.log10(fmax)
            lv0, lv1 = math.log10(vmin), math.log10(vmax)
        except (ValueError, OverflowError):
            return None

        if slope == -1:
            v_at_fmin = constant / (TWO_PI * fmin)
            v_at_fmax = constant / (TWO_PI * fmax)
        else:
            v_at_fmin = TWO_PI * fmin * constant
            v_at_fmax = TWO_PI * fmax * constant

        # Quick reject: entirely above or below view?
        try:
            lvy0 = math.log10(v_at_fmin)
            lvy1 = math.log10(v_at_fmax)
        except (ValueError, OverflowError):
            return None

        hi = max(lvy0, lvy1)
        lo = min(lvy0, lvy1)
        if lo > lv1 or hi < lv0:
            return None

        # Build log-spaced frequency array covering visible portion
        log_f = np.linspace(lf0, lf1, n_pts)
        f_arr = 10.0**log_f
        if slope == -1:
            v_arr = constant / (TWO_PI * f_arr)
        else:
            v_arr = TWO_PI * f_arr * constant

        # Mask to visible y-range
        mask = (v_arr >= vmin) & (v_arr <= vmax)
        if not mask.any():
            return None

        # Find contiguous visible run
        indices = np.where(mask)[0]
        # Extend by one on each side for smooth entry/exit at boundary
        i0 = max(indices[0] - 1, 0)
        i1 = min(indices[-1] + 1, n_pts - 1)

        pts = list(zip(f_arr[i0 : i1 + 1].tolist(), v_arr[i0 : i1 + 1].tolist()))
        return pts if len(pts) >= 2 else None

    def _display_angle(self, slope_m: float) -> float:
        """Aspect-ratio-corrected rotation angle for diagonal label text."""
        try:
            fmin, fmax = self.get_xlim()
            vmin, vmax = self.get_ylim()
            u0, u1 = math.log10(fmin), math.log10(fmax)
            v0, v1 = math.log10(vmin), math.log10(vmax)
            w = self.bbox.width
            h = self.bbox.height
            if w <= 0 or h <= 0 or (v1 - v0) == 0:
                return 0.0
            pixel_slope = slope_m * (h / w) * ((u1 - u0) / (v1 - v0))
            return math.degrees(math.atan(pixel_slope))
        except Exception:
            return 0.0

    def _nudge_up(self, x: float, y: float, pixels: float) -> float:
        """Shift a data-space point upward by *pixels* in display space."""
        p = self.transData.transform((x, y))
        p[1] += pixels
        return self.transData.inverted().transform(p)[1]

    # ── status bar ────────────────────────────────────────────────────

    def format_coord(self, x, y) -> str:
        TWO_PI = 2.0 * math.pi
        g0 = self._tri_g0
        base_len = self._tri_base_len
        disp_unit = (
            self._tri_pos_diag_label
            if self._tri_pos_diag_label is not None
            else ("m" if base_len == "m" else "in")
        )
        v_unit = self._tri_v_unit
        neg_label = self._tri_neg_diag_label

        if x <= 0 or y <= 0:
            return ""
        disp = y / (TWO_PI * x)
        if self._tri_g_normalize:
            acc = TWO_PI * x * y / g0
        else:
            acc = TWO_PI * x * y
        # Use ylabel stem (before parenthetical) as velocity label, or fallback to "PV"
        ylabel = self._tri_ylabel
        if ylabel is not None:
            v_label = ylabel.split("(")[0].strip().rstrip()
        else:
            v_label = "PV"
        return (
            f"f={x:.4g} Hz, {v_label}={y:.4g} {v_unit}, "
            f"d={disp:.4g} {disp_unit}, a={acc:.4g} {neg_label}"
        )

    # ── public runtime API ────────────────────────────────────────────

    def set_v_unit(self, v_unit: str):
        """Change velocity/displacement unit and relabel axes."""
        base_len = _base_len_from_v_unit(v_unit)
        self._tri_v_unit = v_unit
        self._tri_base_len = base_len
        self._tri_g0 = G0_SI if base_len == "m" else G0_IN
        if self._tri_ylabel is not None:
            self.set_ylabel(self._tri_ylabel)
        self.stale = True

    def set_diag_grid(self, visible: bool):
        """Toggle diagonal gridlines on or off."""
        self._tri_diag_grid = visible
        self.stale = True

    def set_diag_style(self, **kwargs):
        """Override diagonal line styling (passed to LineCollection)."""
        self._tri_diag_style.update(kwargs)
        self.stale = True

    def set_diag_label_style(self, **kwargs):
        """Override diagonal label styling (passed to Text).

        Accepts any keyword accepted by ``matplotlib.text.Text``, e.g.
        ``fontsize``, ``fontfamily``, ``fontstyle``, ``fontweight``,
        ``color``, ``alpha``, ``zorder``, ``bbox``, etc.
        """
        self._tri_label_style.update(kwargs)
        self.stale = True


# ── TripartiteProjection for _as_mpl_axes protocol ───────────────────


class TripartiteProjection:
    """Parameterized projection object for ``subplot_kw=dict(projection=...)``."""

    def __init__(self, v_unit: str = "in/s", diag_grid: bool = True, **kwargs):
        self.v_unit = v_unit
        self.diag_grid = diag_grid
        self.extra_kw = kwargs

    def _as_mpl_axes(self):
        return TripartiteAxes, dict(
            v_unit=self.v_unit, diag_grid=self.diag_grid, **self.extra_kw
        )


# ── Register projection on import ────────────────────────────────────

register_projection(TripartiteAxes)
