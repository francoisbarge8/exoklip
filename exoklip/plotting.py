"""Figures. matplotlib is imported lazily so the package works without it.

Conventions used throughout: residual images get a diverging colormap on a scale
symmetric about zero (a sequential map on signed residuals hides the
over-subtraction that matters); SNR maps get an explicit detection contour; and
contrast curves are plotted with the y axis inverted, because a *deeper* limit
is a *better* result and should go up.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "plot_reduction_summary",
    "plot_snr_map",
    "plot_contrast_curve",
    "plot_adi_principle",
    "plot_throughput",
]


def _plt() -> Any:
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for exoklip.plotting: pip install matplotlib "
            "(or install the package with the [plot] extra)."
        ) from exc
    import os

    if not os.environ.get("DISPLAY") and os.name != "nt":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _symmetric_limits(image: ArrayLike, percentile: float = 99.5) -> tuple[float, float]:
    finite = np.asarray(image)[np.isfinite(image)]
    if finite.size == 0:
        return -1.0, 1.0
    v = float(np.percentile(np.abs(finite), percentile))
    return -v, v


def _mark(ax: Any, positions: Sequence[Sequence[float]], fwhm: float, color: str = "lime") -> None:
    for y, x in positions:
        ax.add_patch(
            _plt().Circle(
                (x, y), 2.0 * fwhm, fill=False, edgecolor=color, linewidth=1.4, alpha=0.9
            )
        )


def plot_reduction_summary(
    images: dict[str, ArrayLike],
    fwhm: float,
    truth_positions: Sequence[Sequence[float]] = (),
    figsize: tuple[float, float] | None = None,
) -> Any:
    """Side-by-side panels of several reductions of the same dataset.

    Each panel is scaled independently and symmetrically about zero, so the
    comparison is about *structure*, not about who has the largest numbers.
    """
    plt = _plt()
    n = len(images)
    fig, axes = plt.subplots(
        1, n, figsize=figsize or (4.2 * n, 4.6), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    for ax, (title, image) in zip(axes, images.items()):
        arr = np.asarray(image, dtype=float)
        lo, hi = _symmetric_limits(arr)
        im = ax.imshow(arr, origin="lower", cmap="RdBu_r", vmin=lo, vmax=hi)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x (pixels)")
        _mark(ax, truth_positions, fwhm)
        fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.85)
    axes[0].set_ylabel("y (pixels)")
    return fig


def plot_snr_map(
    snr: ArrayLike,
    fwhm: float,
    threshold: float = 5.0,
    truth_positions: Sequence[Sequence[float]] = (),
    ax: Any = None,
) -> Any:
    """SNR map with a detection contour."""
    plt = _plt()
    arr = np.asarray(snr, dtype=float)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.6, 5.0), constrained_layout=True)
    else:
        fig = ax.figure

    vmax = float(np.nanpercentile(np.abs(arr), 99.9)) if np.any(np.isfinite(arr)) else 1.0
    im = ax.imshow(arr, origin="lower", cmap="magma", vmin=-vmax / 3, vmax=vmax)
    filled = np.where(np.isfinite(arr), arr, -np.inf)
    ax.contour(filled, levels=[threshold], colors="cyan", linewidths=1.2)
    _mark(ax, truth_positions, fwhm, color="lime")
    ax.set_title(f"Small-sample SNR map (contour at {threshold:g})")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="SNR")
    return fig


def plot_contrast_curve(
    curve: dict[str, Any],
    pixel_scale: float | None = None,
    ax: Any = None,
) -> Any:
    """Contrast curve, with and without the small-sample correction."""
    plt = _plt()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    else:
        fig = ax.figure

    r = np.asarray(curve["radius"], dtype=float)
    ax.semilogy(r, curve["contrast"], "-", lw=2, color="#1f77b4",
                label="5$\\sigma$, Student (Mawet+2014)")
    ax.semilogy(r, curve["contrast_gaussian"], "--", lw=1.5, color="#d62728",
                label="5$\\sigma$, Gaussian (optimistic)")
    ax.invert_yaxis()
    ax.set_xlabel("Separation (pixels)")
    ax.set_ylabel("Contrast ($F_{\\rm planet}/F_\\star$)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False, fontsize=9)

    secondary = ax.secondary_yaxis(
        "right",
        functions=(lambda c: -2.5 * np.log10(np.clip(c, 1e-30, None)),
                   lambda m: 10 ** (-m / 2.5)),
    )
    secondary.set_ylabel("$\\Delta$mag")

    if pixel_scale:
        top = ax.secondary_xaxis(
            "top", functions=(lambda p: p * pixel_scale, lambda a: a / pixel_scale)
        )
        top.set_xlabel("Separation (arcsec)")
    return fig


def plot_adi_principle(
    cube: ArrayLike,
    angles: ArrayLike,
    fwhm: float,
    companion_radius: float,
    companion_pa: float,
    indices: Sequence[int] = (0, -1),
) -> Any:
    """The pedagogical figure: speckles are fixed, the companion rotates.

    Top row shows raw frames — the speckle pattern is identical, the companion
    has moved. Bottom row shows the same frames derotated — now the companion is
    fixed and the speckles are the things that moved. Everything else in the
    package is a way of exploiting that asymmetry.
    """
    plt = _plt()
    from .core import frame_center
    from .inject import companion_position
    from .rotation import cube_derotate

    arr = np.asarray(cube, dtype=float)
    ang = np.asarray(angles, dtype=float)
    idx = [int(i) % arr.shape[0] for i in indices]
    derotated = cube_derotate(arr, ang)
    center = frame_center(arr.shape)

    fig, axes = plt.subplots(2, len(idx), figsize=(4.0 * len(idx), 8.0),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    for col, i in enumerate(idx):
        for row, (data, label) in enumerate(
            ((arr, "raw"), (derotated, "derotated"))
        ):
            ax = axes[row, col]
            frame = data[i]
            positive = frame[np.isfinite(frame) & (frame > 0)]
            floor = np.percentile(positive, 40) if positive.size else 1.0
            ax.imshow(np.log10(np.clip(frame, floor, None)), origin="lower", cmap="inferno")
            pa = companion_pa if row else companion_pa + ang[i]
            y, x = companion_position(companion_radius, pa, center)
            ax.add_patch(plt.Circle((x, y), 2.2 * fwhm, fill=False,
                                    edgecolor="cyan", lw=1.5))
            ax.set_title(f"frame {i}  (PA = {ang[i]:+.1f}$^\\circ$) — {label}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    return fig


def plot_throughput(result: dict[str, Any], ax: Any = None) -> Any:
    """Measured algorithmic throughput against separation."""
    plt = _plt()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 4.2), constrained_layout=True)
    else:
        fig = ax.figure
    r = np.asarray(result["radius"], dtype=float)
    t = np.asarray(result["throughput"], dtype=float)
    err = np.asarray(result.get("throughput_std", np.zeros_like(t)), dtype=float)
    ax.errorbar(r, t, yerr=err, fmt="o-", capsize=3, color="#2ca02c")
    ax.axhline(1.0, ls=":", color="grey", lw=1)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Separation (pixels)")
    ax.set_ylabel("Throughput")
    ax.set_title("Fraction of companion flux surviving the reduction")
    ax.grid(alpha=0.3)
    return fig
