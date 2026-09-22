"""Image-sampled pastel theme for the Fig. 2/Fig. 4 preview."""
from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb

THEME_NAME = "PALETTE_B_BLUEGRAY_BLACKFRAMES_PANELHEIGHTFIX"

PALETTE = {
    "white": "#FFFFFF",
    # Dark neutral retained for text, axes and contour labels.
    "ink": "#2F3E4E",
    "text_secondary": "#6C757D",
    "secondary": "#6C757D",
    "spine": "#000000",
    "grid": "#DFE5EF",
    "minor_grid": "#F2F5F8",
    "panel_line": "#B9D6E8",
    "guide": "#B9D6E8",
    "guide_light": "#DFE5EF",
    "infeasible": "#F1F2F2",
    "infeasible_edge": "#666C70",
    "hatch": "#666C70",
    # Temperature mapping: blue / green / light orange for 4.2 / 10 / 20 K.
    "temp_4k": "#8DCEF3",
    "temp_10k": "#D0E1B5",
    "temp_20k": "#FBDABB",
    "scenario_s1": "#8DCEF3",
    "scenario_s2": "#D0E1B5",
    "scenario_s3": "#FBDABB",
    "cross": "#2F3E4E",
    # Remaining sampled swatches are used for heat-source categories.
    "passive": "#B9D6E8",
    "current_lead": "#B0E0E6",
    "joint": "#F9E5E5",
    "nuclear": "#D0E1B5",
    "charging_em": "#8DCEF3",
    "rose": "#F9E5E5",
    "pale_green": "#D0E1B5",
}


def tint_sequence(base: str, steps: int = 6) -> list[str]:
    """Return an exact white-to-sampled-colour sequence."""
    rgb = to_rgb(base)
    return [
        to_hex(
            tuple((1.0 - weight) + weight * component for component in rgb),
            keep_alpha=False,
        ).upper()
        for weight in (index / (steps - 1) for index in range(steps))
    ]


GREEN_SEQ = tint_sequence(PALETTE["temp_10k"])
BLUE_SEQ = tint_sequence(PALETTE["temp_4k"])
# Fig. 4A only: deepen the warm endpoint while retaining the original hue.
PENALTY_BASE_DEEP = "#DFA1A1"
PENALTY_SEQ = tint_sequence(PENALTY_BASE_DEEP)

# Figure-specific aliases: Fig. 2B green-white; Fig. 2C blue-white.
CRYO_SEQ = GREEN_SEQ
CHARGE_SEQ = BLUE_SEQ

TEMPERATURE = {
    4.2: PALETTE["temp_4k"],
    10.0: PALETTE["temp_10k"],
    20.0: PALETTE["temp_20k"],
}

SCENARIO = {
    "S1": "#61B2E3",
    "S2": "#9FC77B",
    "S3": "#F0B176",
    "cross": PALETTE["cross"],
}

SCENARIO_TINT = {
    "S1": "#DCEEF8",
    "S2": "#E7F0DE",
    "S3": "#FBEADC",
}

# Fig. 4A is explicitly blue-to-white; Fig. 3E retains PENALTY_SEQ.
FIG4A_SEQ = BLUE_SEQ

HEAT_SOURCE = {
    "Passive heat leaks": PALETTE["passive"],
    "Current-lead Joule": PALETTE["current_lead"],
    "Joint Joule": PALETTE["joint"],
    "Nuclear": PALETTE["nuclear"],
    "Charging EM": PALETTE["charging_em"],
    "inactive_border": PALETTE["panel_line"],
}


def make_cmap(name: str, colors: list[str]) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, colors)


CRYO_CMAP = make_cmap("palette_b_green", GREEN_SEQ)
CHARGE_CMAP = make_cmap("palette_b_blue", BLUE_SEQ)
PENALTY_CMAP = make_cmap("palette_b_deep_penalty", PENALTY_SEQ)
