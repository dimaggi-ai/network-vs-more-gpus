"""Python twin of the data-viz palette validator.

The reference implementation ships as JavaScript. Node is unusable in this
environment (its ICU dependency is missing), so the same computations are
implemented here rather than skipped: the palette checks are meant to be
computed, not eyeballed.

Ported faithfully from ``scripts/validate_palette.js``: sRGB to linear, OKLab,
OKLCH, the Machado-Oliveira-Fernandes (2009) severity-1.0 colour-vision-
deficiency transforms, Euclidean OKLab delta-E scaled by 100, and WCAG contrast.
Thresholds are copied verbatim.

Run: ``python figures/palette_check.py``
"""

from __future__ import annotations

import itertools
import math
import sys
from typing import Iterable, List, Sequence, Tuple

BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 8.0, 6.0
NORMAL_FLOOR = 15.0
CONTRAST_MIN = 3.0
DEFAULT_SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}

MACHADO = {
    "protan": (
        (0.152286, 1.052583, -0.204868),
        (0.114503, 0.786281, 0.099216),
        (-0.003882, -0.048116, 1.051998),
    ),
    "deutan": (
        (0.367322, 0.860646, -0.227968),
        (0.280085, 0.672501, 0.047413),
        (-0.011820, 0.042940, 0.968881),
    ),
    "tritan": (
        (1.255528, -0.076749, -0.178779),
        (-0.078411, 0.930809, 0.147602),
        (0.004733, 0.691367, 0.303900),
    ),
}


def hex_to_srgb(value: str) -> Tuple[float, float, float]:
    h = value.strip().lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def _s2lin(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def linear(value: str) -> Tuple[float, float, float]:
    return tuple(_s2lin(c) for c in hex_to_srgb(value))  # type: ignore[return-value]


def relative_luminance(value: str) -> float:
    r, g, b = linear(value)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((relative_luminance(a), relative_luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def oklab_from_linear(rgb: Sequence[float]) -> Tuple[float, float, float]:
    r, g, b = rgb
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    )


def oklch(value: str) -> Tuple[float, float]:
    lightness, a, b = oklab_from_linear(linear(value))
    return lightness, math.hypot(a, b)


def simulate(value: str, kind: str) -> Tuple[float, float, float]:
    r, g, b = linear(value)
    matrix = MACHADO[kind]
    return tuple(  # type: ignore[return-value]
        min(1.0, max(0.0, row[0] * r + row[1] * g + row[2] * b)) for row in matrix
    )


def delta_e(h1: str, h2: str, kind: str | None = None) -> float:
    a = oklab_from_linear(simulate(h1, kind) if kind else linear(h1))
    b = oklab_from_linear(simulate(h2, kind) if kind else linear(h2))
    return 100 * math.dist(a, b)


def validate(
    palette: Sequence[str], mode: str = "light", surface: str | None = None, pairs: str = "adjacent"
) -> bool:
    surface = surface or DEFAULT_SURFACE[mode]
    lo_band, hi_band = BAND[mode]
    ok = True

    print(f"\n=== palette check: mode={mode} surface={surface} pairs={pairs} ===")
    print(f"{'slot':<5}{'hex':<10}{'L':>7}{'C':>8}{'contrast':>10}  status")
    for i, colour in enumerate(palette, 1):
        lightness, chroma = oklch(colour)
        ratio = contrast(colour, surface)
        flags: List[str] = []
        if not (lo_band <= lightness <= hi_band):
            flags.append(f"FAIL lightness band [{lo_band},{hi_band}]")
            ok = False
        if chroma < CHROMA_FLOOR:
            flags.append(f"FAIL chroma floor {CHROMA_FLOOR}")
            ok = False
        if ratio < CONTRAST_MIN:
            flags.append("WARN contrast <3:1 (relief rule: direct labels or table view)")
        print(
            f"{i:<5}{colour:<10}{lightness:>7.3f}{chroma:>8.3f}{ratio:>10.2f}  "
            + ("; ".join(flags) if flags else "ok")
        )

    index_pairs: Iterable[Tuple[int, int]]
    if pairs == "all":
        index_pairs = itertools.combinations(range(len(palette)), 2)
    else:
        index_pairs = zip(range(len(palette) - 1), range(1, len(palette)))
    index_pairs = list(index_pairs)

    worst_cvd, worst_cvd_pair = math.inf, None
    worst_normal, worst_normal_pair = math.inf, None
    print(f"\n{'pair':<12}{'protan':>9}{'deutan':>9}{'tritan':>9}{'normal':>9}  status")
    for i, j in index_pairs:
        a, b = palette[i], palette[j]
        protan, deutan, tritan = (delta_e(a, b, k) for k in ("protan", "deutan", "tritan"))
        normal = delta_e(a, b)
        cvd = min(protan, deutan)
        status = "ok"
        if cvd < CVD_FLOOR:
            status, ok = f"FAIL cvd<{CVD_FLOOR}", False
        elif cvd < CVD_TARGET:
            status = f"WARN cvd<{CVD_TARGET} (secondary encoding required)"
        if normal < NORMAL_FLOOR:
            status, ok = f"FAIL normal<{NORMAL_FLOOR}", False
        if cvd < worst_cvd:
            worst_cvd, worst_cvd_pair = cvd, (i + 1, j + 1)
        if normal < worst_normal:
            worst_normal, worst_normal_pair = normal, (i + 1, j + 1)
        print(
            f"{f'{i+1}-{j+1}':<12}{protan:>9.1f}{deutan:>9.1f}{tritan:>9.1f}{normal:>9.1f}  {status}"
        )

    print(
        f"\nworst CVD dE {worst_cvd:.1f} at {worst_cvd_pair}; "
        f"worst normal-vision dE {worst_normal:.1f} at {worst_normal_pair}"
    )
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


#: Categorical slots 1-5 of the documented reference palette, in documented order.
LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]

#: Sequential ramp used for the decision maps (single hue, light to dark).
SEQUENTIAL_LIGHT = ["#e8f0fb", "#a9c8ee", "#5598e7", "#2a78d6", "#104281"]


def main() -> int:
    ok = validate(LIGHT, mode="light")
    ok = validate(DARK, mode="dark") and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
