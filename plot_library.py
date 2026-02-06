# plot_library.py (unified plotting utilities - English version)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import matplotlib as mpl
from matplotlib import patheffects
from pathlib import Path
import config as cfg
from typing import Optional, Tuple

# =============================================================================
# Helper functions: contour colors based on background luminance
# =============================================================================

def calculate_luminance(rgb: Tuple[float, float, float]) -> float:
    """
    Compute the luminance of an RGB color.
    
    Uses the standard relative‑luminance formula:
        L = 0.299*R + 0.587*G + 0.114*B
    Return range: 0 (darkest) to 1 (brightest).
    
    Args:
        rgb: RGB color tuple with components in [0, 1].
    
    Returns:
        Luminance value in [0, 1].
    """
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b


def get_contour_color_from_background(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    contour_level: float,
    cmap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
    threshold: float = 0.5
) -> str:
    """
    Choose a contour color (black/white) based on the background luminance
    along the contour path.
    
    Method:
    1. Sample points where Z is close to `contour_level`.
    2. Map each sampled Z to a color via `norm` and `cmap`.
    3. Compute the mean luminance of all sampled colors.
    4. If mean luminance < `threshold`, return 'white' (for dark background).
    5. Otherwise return 'black' (for light background).
    
    Args:
        X: X‑coordinate grid.
        Y: Y‑coordinate grid.
        Z: Z‑value grid.
        contour_level: contour level value.
        cmap: colormap.
        norm: normalization instance.
        threshold: luminance threshold (default 0.5).
    
    Returns:
        Contour color string: 'white' or 'black'.
    """
    # Sample points along the contour region
    # Use points where Z is close to contour_level
    mask = np.abs(Z - contour_level) < (np.nanmax(Z) - np.nanmin(Z)) * 0.1
    
    if not mask.any():
        # If no near‑contour points are found, fall back to all finite points
        mask = np.isfinite(Z)
    
    if not mask.any():
        return 'black'  # default to black
    
    # Extract sampled Z values
    sampled_Z = Z[mask]
    
    # Normalize via `norm`
    normalized_values = norm(sampled_Z)
    
    # Clip to [0, 1]
    normalized_values = np.clip(normalized_values, 0, 1)
    
    # Map normalized values to colors via cmap
    colors = cmap(normalized_values)
    
    # Compute mean luminance for all sampled points
    luminances = []
    for color in colors:
        if len(color) >= 3:
            rgb = (color[0], color[1], color[2])
            lum = calculate_luminance(rgb)
            luminances.append(lum)
    
    if not luminances:
        return 'black'
    
    avg_luminance = np.mean(luminances)
    
    # Choose contour color based on average luminance
    if avg_luminance < threshold:
        return 'white'  # white contour on dark background
    else:
        return 'black'  # black contour on light background


def get_contour_colors_for_levels(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    levels: np.ndarray,
    cmap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
    threshold: float = 0.3
) -> list:
    """
    Compute a list of contour colors for multiple contour levels.
    
    Args:
        X: X‑coordinate grid.
        Y: Y‑coordinate grid.
        Z: Z‑value grid.
        levels: array of contour levels.
        cmap: colormap.
        norm: normalization instance.
        threshold: luminance threshold (default 0.3).
    
    Returns:
        List of color strings, one per contour level.
    """
    colors = []
    for level in levels:
        color = get_contour_color_from_background(
            X, Y, Z, level, cmap, norm, threshold
        )
        colors.append(color)
    return colors


# =============================================================================
# Plot function 1: combined chart (cost breakdown + payback period)
# =============================================================================
font_size = 20
font_size_title = 24
font_size_label = 20
font_size_tick = cfg.HEATMAP_FONT_SIZE_TICK
font_size_legend = 20
font_size_legend_title = 20
font_size_legend_label = 20
font_size_legend_tick = 20

def plot_combined_charts(excel_path: str, save_path: str, project_year_to_plot: int):
    """
    Generate a combined one‑column figure containing both cost breakdown
    and simple payback period, sharing the same width and x‑axis grouping.
    """
    setup_plot_style()
    try:
        df_all = pd.read_excel(excel_path)
    except FileNotFoundError:
        print(f"Error: file not found at '{excel_path}'.")
        return
    
    # --- Data filtering ---
    df = df_all[df_all['project_years'] == project_year_to_plot].copy()
    if df.empty:
        print(f"Error: no records found for project_years == {project_year_to_plot} in '{excel_path}'.")
        return
    
    # --- Style definitions ---
    color_map = {
        (4.2, 'He'): '#386192', (10, 'He'): '#00BFFF',
        (20, 'He'): '#ACD8E5', (20, 'H2'): '#A3D99F',
    }
    hatch_map = {'magnet': '/', 'power': '|', 'coolant': '\\'}

    # --- Create figure and two subplots sharing the x‑axis ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    ax_cost, ax_payback = axes[0], axes[1]

    # --- Core plotting logic ---
    bar_width = 0.6
    current_x = 0
    main_group_padding = 1.0
    tech_scenarios = sorted(df['tech_scenario'].unique())
    x_main_ticks, x_tick_labels = [], []

    def sort_coolant_temp(combo_values):
        return (0 if combo_values[1] == 'He' else 1, combo_values[0])

    for scenario in tech_scenarios:
        df_scenario = df[df['tech_scenario'] == scenario].copy()
        combos = sorted(df_scenario[['temperature_K', 'coolant']].drop_duplicates().values, key=sort_coolant_temp)
        group_start_x = current_x

        for temp, coolant in combos:
            row = df_scenario[(df_scenario['temperature_K'] == temp) & (df_scenario['coolant'] == coolant)]
            if not row.empty:
                color = color_map.get((temp, coolant), 'gray')

                # Cost breakdown bars
                magnet_cost = row['tape_cost_$'].iloc[0] / 1e6
                power_cost = row['power_supply_cost_$'].iloc[0] / 1e6
                coolant_cost = row['coolant_lifetime_cost_$'].iloc[0] / 1e6
                
                ax_cost.bar(current_x, magnet_cost, width=bar_width, color=color, hatch=hatch_map['magnet'], edgecolor='black')
                ax_cost.bar(current_x, power_cost, bottom=magnet_cost, width=bar_width, color=color, hatch=hatch_map['power'], edgecolor='black')
                ax_cost.bar(current_x, coolant_cost, bottom=magnet_cost + power_cost, width=bar_width, color=color, hatch=hatch_map['coolant'], edgecolor='black')

                # Simple payback bar
                payback_value = row['simple_payback_years'].iloc[0]
                ax_payback.bar(current_x, payback_value, width=bar_width, color=color, edgecolor='black')
                current_x += bar_width
        
        group_end_x = current_x
        x_main_ticks.append(group_start_x + (group_end_x - group_start_x - bar_width) / 2)
        x_tick_labels.append(scenario.replace('_', ' '))
        current_x += main_group_padding

    # --- Axes labels and layout (English) ---
    ax_cost.set_ylabel('Direct Magnet System Cost\n(Million USD)', labelpad=18)
    ax_payback.set_ylabel('Simple Payback (Years)', labelpad=18)
    ax_payback.set_xlabel('Technology Scenario', labelpad=20)
    plt.xticks(x_main_ticks, x_tick_labels)
    fig.align_ylabels(axes)
    plt.tight_layout(pad=3.0)
    
    plt.savefig(save_path, dpi=cfg.PLOT_DPI)
    plt.close(fig)
    print(f"Combined economic analysis chart saved to: {save_path}")

# =============================================================================
# Plot function 2: parasitic‑power heatmap for a single scenario/temperature
# =============================================================================

def plot_parasitic_heatmap_single(df: pd.DataFrame, output_dir: Path, cmap: str, norm: mpl.colors.Normalize, 
                                   scenario: str, temperature_K: float, filename_suffix: str = "", 
                                   use_stroke: bool = False):
    """
    Generate and save a parasitic‑power heatmap for a single scenario and temperature.
    The main panel excludes colorbar, axis labels, and tick labels.
    
    Parameters
    ----------
    df:
        DataFrame containing data for a single temperature.
    output_dir:
        Directory to save the figure.
    cmap:
        Colormap name or object.
    norm:
        Global normalization object (shared across plots for a consistent color range).
    scenario:
        Scenario name.
    temperature_K:
        Operating temperature in K.
    filename_suffix:
        Optional filename suffix to distinguish colormaps.
    """
    font_size = cfg.HEATMAP_FONT_SIZE
    setup_plot_style()
    
    label = f"{temperature_K}K"
    
    # Create single subplot
    fig, ax = plt.subplots(1, 1, figsize=cfg.HEATMAP_FIGSIZE_SINGLE)
    
    # Prepare data: drop duplicate (Npw, R_joint) combinations if necessary
    df_clean = df.copy()
    if df_clean.duplicated(subset=['Npw', 'R_joint']).any():
        # If duplicates exist, average over each (Npw, R_joint) group
        df_clean = df_clean.groupby(['Npw', 'R_joint'], as_index=False)['r_parasitic_pct'].mean()
    
    # Pivot data to 2D grid
    pivot = df_clean.pivot(index="Npw", columns="R_joint", values="r_parasitic_pct")
    X, Y = np.meshgrid(pivot.columns, pivot.index)
    Z = pivot.values

    X_log = np.log10(X * 1e9)
    
    
    # Use cell‑center coordinates for pcolormesh and contour so they are aligned
    # Resolve colormap (supports custom/truncated names defined in config)
    cmap_resolved = cfg.resolve_cmap(cmap) if hasattr(cfg, "resolve_cmap") else cmap
    if isinstance(cmap_resolved, str):
        try:
            cmap_obj = plt.get_cmap(cmap_resolved)
        except ValueError:
            # If matplotlib does not recognize the name, try loading from seaborn
            try:
                import seaborn as sns
                cmap_obj = sns.color_palette(cmap_resolved, as_cmap=True)
            except (ImportError, ValueError):
                # If all attempts fail, fall back to a default colormap
                cmap_obj = plt.get_cmap("viridis")
    else:
        cmap_obj = cmap_resolved
    
    # Draw heatmap (using cell centers, shading="auto")
    # pcolormesh and contour both share the same centers for full consistency
    pcm = ax.pcolormesh(X_log, Y, Z, cmap=cmap_obj, norm=norm, shading="auto", alpha=cfg.HEATMAP_ALPHA)
    
    # Contour levels (in %) as a function of temperature
    levels = {'4.2K': [7, 8, 10, 15, 20, 50], '10.0K': [3, 4, 5, 6, 8, 10], '20.0K': [1.5, 2, 2.5, 3, 4, 5]}
    contour_levels = levels.get(label, [])
    
    # Draw contours and hide labels that are too close to axes/borders
    def _hide_labels_near_axes(texts, axis, margin_axes: float = 0.05, margin_data_y: float = 5.0):
        """
        Hide contour labels that are too close to the axes or figure borders
        to avoid overlap between numbers and axes/frame.
        
        Parameters
        ----------
        texts:
            List of text objects returned by `clabel`.
        axis:
            Matplotlib Axes object.
        margin_axes:
            Margin (0–1) in normalized axes coordinates used to decide if a
            label is close to the border (default 0.05).
        margin_data_y:
            Margin in data coordinates for the bottom/top region in y
            (used to treat the x‑axis area specially, default 5.0).
        
        Returns
        -------
        bool
            True if all labels were hidden; False otherwise.
        """
        if not texts:
            return False
        
        # Get current data limits
        xlim = axis.get_xlim()
        ylim = axis.get_ylim()
        
        hidden_count = 0
        for txt in texts:
            xdata, ydata = txt.get_position()
            
            # Method 1: use axes coordinates (works for all borders)
            x_disp, y_disp = axis.transData.transform((xdata, ydata))
            x_axes, y_axes = axis.transAxes.inverted().transform((x_disp, y_disp))
            
            # Method 2: use data coordinates (especially for bottom x‑axis region)
            near_bottom = ydata < (ylim[0] + margin_data_y)
            near_top = ydata > (ylim[1] - margin_data_y)
            near_left = xdata < (xlim[0] + (xlim[1] - xlim[0]) * margin_axes)
            near_right = xdata > (xlim[1] - (xlim[1] - xlim[0]) * margin_axes)
            
            # Combined axes‑coordinate check for stricter edge detection
            near_edge_axes = (
                x_axes < margin_axes
                or x_axes > 1.0 - margin_axes
                or y_axes < margin_axes
                or y_axes > 1.0 - margin_axes
            )
            
            # Hide label if it is close to any edge
            if near_bottom or near_top or near_left or near_right or near_edge_axes:
                txt.set_visible(False)
                hidden_count += 1
        
        # Return True if all labels were hidden
        return hidden_count == len(texts)
    
    if len(contour_levels) > 0:
        if use_stroke:
            # Use stroke effect: add white outline to black contour lines
            CS = None
            clabels = []
            for i, level in enumerate(contour_levels):
                # Draw black contour line
                CS_level = ax.contour(X_log, Y, Z, levels=[level], colors='black', linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH)
                if CS is None:
                    CS = CS_level
                # Add white stroke to each contour collection
                for collection in CS_level.collections:
                    collection.set_path_effects([
                        patheffects.withStroke(
                            linewidth=cfg.CONTOUR_STROKE_LINEWIDTH,
                            foreground=cfg.CONTOUR_STROKE_FOREGROUND,
                            alpha=cfg.CONTOUR_STROKE_ALPHA
                        ),
                        patheffects.Normal()
                    ])
                # Add stroke effect to contour labels
                texts_level = ax.clabel(CS_level, inline=True, fontsize=font_size, fmt=cfg.HEATMAP_CONTOUR_FMT_PARASITIC, colors='black')
                if texts_level:
                    # Apply white stroke to each text label
                    for txt in texts_level:
                        txt.set_path_effects([
                            patheffects.withStroke(
                                linewidth=cfg.LABEL_STROKE_LINEWIDTH,
                                foreground=cfg.LABEL_STROKE_FOREGROUND,
                                alpha=cfg.LABEL_STROKE_ALPHA
                            ),
                            patheffects.Normal()
                        ])
                    # Immediately hide labels near axes
                    all_hidden = _hide_labels_near_axes(texts_level, ax)
                    # If all labels for this level are hidden, remove the contour
                    if all_hidden:
                        for collection in CS_level.collections:
                            collection.remove()
                    else:
                        clabels.extend(texts_level)
        else:
            # Use background luminance to choose contour color
            level_colors_list = get_contour_colors_for_levels(X_log, Y, Z, np.array(contour_levels), cmap_obj, norm)
            
            # Draw each contour with its selected color
            CS = None
            clabels = []
            for i, level in enumerate(contour_levels):
                color = level_colors_list[i] if i < len(level_colors_list) else 'black'  # default to black
                CS_level = ax.contour(X_log, Y, Z, levels=[level], colors=color, linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH)
                if CS is None:
                    CS = CS_level
                texts_level = ax.clabel(CS_level, inline=True, fontsize=font_size, fmt=cfg.HEATMAP_CONTOUR_FMT_PARASITIC, colors=color)
                if texts_level:
                    # Immediately hide labels near axes
                    all_hidden = _hide_labels_near_axes(texts_level, ax)
                    # If all labels for this level are hidden, remove the contour
                    if all_hidden:
                        for collection in CS_level.collections:
                            collection.remove()
                    else:
                        clabels.extend(texts_level)
    else:
        CS = None
        clabels = None
    

    # Set axis limits but hide ticks and tick labels

    xlim_min = np.min(df["R_joint"])*1e9
    xlim_max = np.max(df["R_joint"])*1e9
    ax.set_xlim(np.log10(xlim_min), np.log10(xlim_max))
    ylim_min = np.min(df["Npw"])
    ylim_max = np.max(df["Npw"])
    ax.set_ylim(ylim_min, ylim_max)
    
    # Remove all tick values and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Remove axis labels
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    # Set frame style (consistent with AF heatmaps)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(3)
    
    # Preserve title if needed (scenario and temperature)
    # ax.set_title(f"{scenario} - {label}")

    # Tight layout: remove white margins so the panel fills the figure
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    ax.set_position([0, 0, 1, 1]) 
    
    # Save single panel (no colorbar, axis labels, or ticks)
    # Use the same saving style as AF heatmaps: `bbox_inches="tight", pad_inches=0.0`
    suffix_str = f"_{filename_suffix}" if filename_suffix else ""
    plot_path = output_dir / f"parasitic_ratio_heatmap_{scenario}_{label}{suffix_str}.{cfg.PLOT_FORMAT}"
    plt.savefig(plot_path, dpi=cfg.PLOT_DPI, bbox_inches='tight', pad_inches=0.0)
    plt.close(fig)
    print(f"Heatmap saved to: {plot_path}")


def save_parasitic_heatmap_colorbar(output_dir: Path, cmap: str, norm: mpl.colors.Normalize, filename_suffix: str = ""):
    """
    Generate and save a standalone colorbar shared by all parasitic‑ratio heatmaps.
    
    Parameters
    ----------
    output_dir:
        Directory to save the colorbar figure.
    cmap:
        Colormap name or object.
    norm:
        Normalization instance.
    filename_suffix:
        Optional filename suffix to distinguish colormaps.
    """
    setup_plot_style()
    
    # Support custom/truncated colormap names defined in config (e.g. "YlGnBu_trunc_0p8")
    cmap_resolved = cfg.resolve_cmap(cmap) if hasattr(cfg, "resolve_cmap") else cmap
    
    # Create a new ScalarMappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_resolved, norm=norm)
    sm.set_array([])  # empty array placeholder
    
    fig_cbar = plt.figure(figsize=cfg.HEATMAP_FIGSIZE_COLORBAR)
    ax_cbar = fig_cbar.add_axes([0.1, 0.15, 0.3, 0.7])
    cbar = fig_cbar.colorbar(sm, cax=ax_cbar, label="Parasitic Power Fraction (%)")
    
    # Configure colorbar ticks
    cbar.set_ticks([2, 4, 10, 20, 50])
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    cbar.ax.tick_params(labelsize=cfg.HEATMAP_FONT_SIZE_COLORBAR)
    cbar.set_label("Parasitic Power Fraction (%)", fontsize=cfg.HEATMAP_FONT_SIZE_COLORBAR, labelpad=15)
    
    # Build filename based on suffix
    suffix_str = f"_{filename_suffix}" if filename_suffix else ""
    cbar_path = output_dir / f"parasitic_ratio_heatmap_colorbar{suffix_str}.{cfg.PLOT_FORMAT}"
    plt.savefig(cbar_path, dpi=cfg.PLOT_DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig_cbar)


def plot_cryo_power_heatmap_single(
    df: pd.DataFrame,
    output_dir: Path,
    cmap: str,
    norm: mpl.colors.Normalize,
    mode_label: str,
    temperature_K: float,
    value_col: str = "cryo_power_MWe",
    filename_suffix: str = "",
):
    """
    Plot a cryogenic‑power heatmap (in MWe) for a single operating mode
    and temperature.

    X‑axis: joint resistance (log scale), Y‑axis: number of parallel tapes.
    The main panel excludes colorbar, axis labels, and tick labels.

    Parameters
    ----------
    df:
        DataFrame containing columns Npw, R_joint, and `value_col`.
    output_dir:
        Directory to save the figure.
    cmap:
        Colormap name or object.
    norm:
        Normalization object (e.g. `LogNorm`).
    mode_label:
        Mode label for filenames (e.g. "pulse", "dwell", "static").
    temperature_K:
        Temperature (e.g. 4.2, 10.0, 20.0).
    value_col:
        Name of the value column (units: MWe).
    filename_suffix:
        Optional filename suffix.
    """
    font_size = cfg.HEATMAP_FONT_SIZE
    setup_plot_style()

    label = f"{temperature_K}K"
    if temperature_K == 10.0 or temperature_K == 20.0:
        label = f"{int(temperature_K)}K"

    fig, ax = plt.subplots(1, 1, figsize=cfg.HEATMAP_FIGSIZE_SINGLE)

    df_clean = df.copy()
    if df_clean.duplicated(subset=["Npw", "R_joint"]).any():
        df_clean = df_clean.groupby(["Npw", "R_joint"], as_index=False)[value_col].mean()

    pivot = df_clean.pivot(index="Npw", columns="R_joint", values=value_col)
    X, Y = np.meshgrid(pivot.columns, pivot.index)
    Z = pivot.values
    X_log = np.log10(X * 1e9)

    cmap_resolved = cfg.resolve_cmap(cmap) if hasattr(cfg, "resolve_cmap") else cmap
    if isinstance(cmap_resolved, str):
        try:
            cmap_obj = plt.get_cmap(cmap_resolved)
        except ValueError:
            try:
                import seaborn as sns
                cmap_obj = sns.color_palette(cmap_resolved, as_cmap=True)
            except (ImportError, ValueError):
                cmap_obj = plt.get_cmap("viridis")
    else:
        cmap_obj = cmap_resolved

    ax.pcolormesh(X_log, Y, Z, cmap=cmap_obj, norm=norm, shading="auto", alpha=cfg.HEATMAP_ALPHA)

    z_min, z_max = np.nanmin(Z), np.nanmax(Z)
    if np.isfinite(z_min) and np.isfinite(z_max) and z_max > z_min:
        finite_pos = Z[np.isfinite(Z) & (Z > 0)]
        if finite_pos.size > 0 and finite_pos.max() > finite_pos.min():
            n_contour = 6
            contour_levels = np.geomspace(finite_pos.min(), finite_pos.max(), n_contour)
            XX, YY = np.meshgrid(X_log[0, :] if X_log.ndim > 1 else X_log, Y[:, 0] if Y.ndim > 1 else Y)
            if XX.shape != Z.shape:
                XX, YY = np.meshgrid(np.unique(X_log), np.unique(Y))
            CS = ax.contour(XX, YY, Z, levels=contour_levels, colors="black", linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH_CRYO)
            ax.clabel(CS, inline=True, fontsize=font_size, fmt=cfg.HEATMAP_CONTOUR_FMT_CRYO, colors="black")

    xlim_min = np.min(df["R_joint"]) * 1e9
    xlim_max = np.max(df["R_joint"]) * 1e9
    ax.set_xlim(np.log10(xlim_min), np.log10(xlim_max))
    ylim_min = np.min(df["Npw"])
    ylim_max = np.max(df["Npw"])
    ax.set_ylim(ylim_min, ylim_max)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(3)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    ax.set_position([0, 0, 1, 1])

    suffix_str = f"_{filename_suffix}" if filename_suffix else ""
    safe_mode = mode_label.replace(" ", "_").lower()
    plot_path = output_dir / f"cryo_power_MWe_heatmap_{safe_mode}_{label}{suffix_str}.{cfg.PLOT_FORMAT}"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=cfg.PLOT_DPI, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    print(f"Cryo power (MWe) heatmap saved to: {plot_path}")


def plot_cryo_mode_grid(
    df: pd.DataFrame,
    scenarios: list,
    temperatures: list,
    save_path: Path,
    cmap: str,
    norm: mpl.colors.Normalize,
    mode_label: str
):
    """
    Plot a grid of cryogenic‑power heatmaps for a given operating mode.
    
    The grid is arranged as rows = scenarios and columns = temperatures.
    Color represents the cryogenic system electric power.
    """
    setup_plot_style()
    scenarios = list(scenarios)
    temperatures = list(temperatures)
    n_rows = len(scenarios)
    n_cols = len(temperatures)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * cfg.HEATMAP_FIGSIZE_SINGLE[0], n_rows * cfg.HEATMAP_FIGSIZE_SINGLE[1]),
        sharex=True,
        sharey=True
    )
    axes = np.atleast_2d(axes)
    pcm = None
    subplot_records = []
    xlim_range = (np.log10(1), np.log10(100))
    ylim_range = (2, 200)

    for i, scenario in enumerate(scenarios):
        for j, temp in enumerate(temperatures):
            ax = axes[i, j]
            subset = df[
                (df["scenario"] == scenario) &
                (df["temperature_K"] == temp)
            ]
            if subset.empty:
                ax.axis("off")
                continue

            subset = subset.copy()
            subset["cryo_power_MW"] = subset["cryo_power_W"] / 1e6

            pivot = subset.pivot_table(
                index="Npw",
                columns="R_joint",
                values="cryo_power_MW",
                aggfunc="mean"
            ).sort_index().sort_index(axis=1)

            X = np.log10(pivot.columns.to_numpy() * 1e9)
            Y = pivot.index.to_numpy()
            Z = pivot.to_numpy()

            pcm = ax.pcolormesh(
                X, Y, Z,
                cmap=cmap,
                norm=norm,
                shading="auto",
                alpha=cfg.HEATMAP_ALPHA
            )

            z_min = np.nanmin(Z)
            z_max = np.nanmax(Z)
            contour_levels = None
            contour_num = 7
            if np.isfinite(z_min) and np.isfinite(z_max) and z_max > z_min:
                finite_positive = Z[np.isfinite(Z) & (Z > 0)]
                if finite_positive.size > 0:
                    z_min_pos = finite_positive.min()
                    z_max_pos = finite_positive.max()
                    if z_max_pos > z_min_pos:
                        if z_max_pos / z_min_pos >= 10:
                            contour_levels = np.geomspace(z_min_pos, z_max_pos, contour_num)
                        else:
                            contour_levels = np.linspace(z_min_pos, z_max_pos, contour_num)
                    else:
                        contour_levels = np.linspace(z_min, z_max, contour_num)
                else:
                    contour_levels = np.linspace(z_min, z_max, contour_num)
                XX, YY = np.meshgrid(X, Y)
                CS = ax.contour(
                    XX, YY, Z,
                    levels=contour_levels,
                    colors="black",
                    linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH_CRYO
                )
                ax.clabel(CS, inline=True, fontsize=font_size_tick, fmt=cfg.HEATMAP_CONTOUR_FMT_CRYO, colors="black")

            ax.set_xlim(xlim_range)
            ax.set_ylim(ylim_range)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")

            subplot_records.append({
                "scenario": scenario,
                "temperature": temp,
                "X": X,
                "Y": Y,
                "Z": Z,
                "contour_levels": contour_levels
            })

    fig.suptitle(mode_label, fontsize=cfg.HEATMAP_FONT_SIZE_TITLE, y=0.99)
    
    if pcm is not None:
    # If a colorbar is present, use `subplots_adjust` instead of `tight_layout`
    # because manually added `add_axes` is not compatible with `tight_layout`
        fig.subplots_adjust(right=0.88, top=0.96)
        cax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(pcm, cax=cax)
        cbar.ax.tick_params(labelsize=cfg.HEATMAP_FONT_SIZE_TICK)
        cbar.set_label("Cryogenic Power (MW)", fontsize=cfg.HEATMAP_FONT_SIZE_COLORBAR_TITLE)
    else:
        # If there is no colorbar, use `tight_layout`
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=cfg.PLOT_DPI)
    plt.close(fig)
    print(f"Cryogenic power grid saved to: {save_path}")

    # Save individual subplots
    individual_dir = save_path.parent / f"{save_path.stem}_subplots"
    individual_dir.mkdir(parents=True, exist_ok=True)
    for record in subplot_records:
        fig_single, ax_single = plt.subplots(figsize=cfg.HEATMAP_FIGSIZE_SINGLE)
        pcm_single = ax_single.pcolormesh(
            record["X"], record["Y"], record["Z"],
            cmap=cmap,
            alpha=cfg.HEATMAP_ALPHA,
            norm=norm,
            shading="auto"
        )
        if record["contour_levels"] is not None:
            XX, YY = np.meshgrid(record["X"], record["Y"])
            CS_single = ax_single.contour(
                XX, YY, record["Z"],
                levels=record["contour_levels"],
                colors="black",
                linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH_CRYO
            )
            ax_single.clabel(CS_single, inline=True, fontsize=font_size_tick, fmt=cfg.HEATMAP_CONTOUR_FMT_CRYO, colors="black")

        ax_single.set_xlim(xlim_range)
        ax_single.set_ylim(ylim_range)
        ax_single.set_xticks([])
        ax_single.set_xticklabels([])
        ax_single.set_xlabel("")
        ax_single.set_yticks([])
        ax_single.set_ylabel("")
        
        # Set frame style (consistent with AF heatmaps)
        for spine in ax_single.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(3)
            spine.set_visible(True)

        # Tight layout: remove white margins so the panel fully fills the figure
        fig_single.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        ax_single.set_position([0, 0, 1, 1])
        
        single_path = individual_dir / f"{record['scenario']}_{record['temperature']:.1f}K.{cfg.PLOT_FORMAT}"
        # Use the same saving style as AF heatmaps: `bbox_inches="tight", pad_inches=0.0`
        fig_single.savefig(single_path, dpi=cfg.PLOT_DPI, bbox_inches='tight', pad_inches=0.0)
        plt.close(fig_single)
        print(f"  └─ Saved subplot: {single_path}")

    # Save shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    colorbar_fig = plt.figure(figsize=cfg.HEATMAP_FIGSIZE_COLORBAR)
    colorbar_ax = colorbar_fig.add_axes([0.3, 0.05, 0.4, 0.9])
    colorbar = plt.colorbar(sm, cax=colorbar_ax)
    colorbar.set_label("Cryogenic Power (MW)")
    colorbar_path = save_path.parent / f"{save_path.stem}_colorbar.{cfg.PLOT_FORMAT}"
    colorbar_fig.savefig(colorbar_path, dpi=cfg.PLOT_DPI, bbox_inches='tight')
    plt.close(colorbar_fig)

# =============================================================================
# Helper plotting utilities (legends, parasitic‑power‑vs‑temperature, etc.)
# =============================================================================

def setup_plot_style():
    """Configure global Matplotlib plotting style."""
    plt.style.use('default')
    # Font configuration: prefer Arial; fall back to CJK‑capable fonts when needed
    # Windows typical CJK fonts: Microsoft YaHei, SimHei, SimSun
    # macOS typical CJK fonts: PingFang SC, STHeiti
    # Linux typical CJK fonts: WenQuanYi Micro Hei, Noto Sans CJK SC
    import platform
    system = platform.system()
    if system == 'Windows':
        # Windows: prefer Arial, fall back to Microsoft YaHei / SimHei / SimSun
        plt.rcParams['font.sans-serif'] = ['Arial', 'Microsoft YaHei', 'SimHei', 'SimSun', 'DejaVu Sans']
    elif system == 'Darwin':  # macOS
        # macOS: prefer Arial, fall back to PingFang SC / STHeiti
        plt.rcParams['font.sans-serif'] = ['Arial', 'PingFang SC', 'STHeiti', 'DejaVu Sans']
    else:  # Linux
        # Linux: prefer Arial, fall back to common CJK fonts
        plt.rcParams['font.sans-serif'] = ['Arial', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
    # Set default font family (prefer sans‑serif / Arial)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['legend.fontsize'] = 20
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['axes.grid'] = False
    plt.rcParams['savefig.transparent'] = True
    plt.rcParams['legend.frameon'] = False

def plot_parasitic_power_vs_temp(excel_path: str, save_path: str):
    """Plot parasitic‑power fraction as a function of operating temperature."""
    setup_plot_style()
    try:
        df = pd.read_excel(excel_path)
    except FileNotFoundError:
        print(f"Error: File not found at '{excel_path}'.")
        return
    power_stats = df.groupby('temperature_K')['r_parasitic_pct'].agg(['mean', 'std']).reset_index()
    fig, ax = plt.subplots(figsize=(10, 8))
    unique_temps = power_stats['temperature_K'].unique()
    colors = ['#386192', '#00BFFF', '#ACD8E5']
    ax.bar(power_stats['temperature_K'], power_stats['mean'], color=colors, width=0.8, edgecolor='black')
    ax.set_xlabel('Operating Temperature (K)', labelpad=15)
    ax.set_ylabel('Cryo-System Parasitic Power (%)', labelpad=15)
    ax.set_title('Parasitic Power vs. Operating Temperature', fontsize=cfg.HEATMAP_FONT_SIZE_TITLE, y=1.05)
    ax.set_xticks(power_stats['temperature_K'])
    plt.tight_layout()
    plt.savefig(save_path, dpi=cfg.PLOT_DPI)
    plt.close(fig)
    print(f"Parasitic power chart saved to: {save_path}")

def save_color_legend(save_path: str):
    """Generate and save a color legend for operating conditions."""
    setup_plot_style()
    color_map = {
        (4.2, 'He'): '#386192', (10, 'He'): '#00BFFF',
        (20, 'He'): '#ACD8E5', (20, 'H2'): '#A3D99F',
    }
    def sort_coolant_temp(combo_values):
        return (0 if combo_values[1] == 'He' else 1, combo_values[0])
    all_combos = sorted(color_map.keys(), key=sort_coolant_temp)
    handles = [mpatches.Patch(color=color_map[combo], label=f'{combo[0]}K - {combo[1]}') for combo in all_combos]
    fig = plt.figure(figsize=(10, 1.5))
    ax = fig.add_subplot(111)
    ax.axis('off')
    legend = ax.legend(
        handles=handles,
        loc='center',
        ncol=len(handles),
        frameon=False,
        fontsize=cfg.HEATMAP_FONT_SIZE_COLORBAR,
        title='Operating Case (Temperature - Coolant)'
    )
    legend.get_frame().set_facecolor('none')
    legend.get_frame().set_edgecolor('none')
    legend.get_title().set_fontsize(cfg.HEATMAP_FONT_SIZE_COLORBAR)
    plt.tight_layout()
    plt.savefig(save_path, dpi=cfg.PLOT_DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Color legend saved to: {save_path}")

def auto_contour_levels(
    Z_sub: np.ndarray,
    norm: mpl.colors.Normalize,
    n_min: int = 4,
    n_max: int = 6,
    q: float = 0.08,
    margin: float = 0.05,
    include_zero: bool = True,
    verbose: bool = False  # whether to print detailed info (mainly for debugging)
) -> np.ndarray:
    """
    Automatically generate 4–6 contour levels for a single subplot, spaced
    roughly uniformly in visual (normalized) space.
    
    Parameters
    ----------
    Z_sub:
        Subplot data array (2D or 1D).
    norm:
        Matplotlib normalization object (must support `inverse`, e.g. SymLogNorm, LogNorm).
    n_min:
        Minimum number of contour levels (default 4).
    n_max:
        Maximum number of contour levels (default 6).
    q:
        Quantile range to avoid extreme outliers (default 0.08 → use 8%–92% quantiles).
    margin:
        Margin fraction in visual space (default 0.05).
    include_zero:
        If data span zero, optionally force the contour closest to zero to be exactly 0.0.
    verbose:
        If True, print additional debug information.
    
    Returns
    -------
    np.ndarray
        Sorted array of contour levels (excluding zmin/zmax).
    """
    # Flatten and drop NaN/inf
    z_flat = Z_sub.flatten()
    z_valid = z_flat[np.isfinite(z_flat)]
    
    # If there are too few valid points, return an empty array
    if len(z_valid) < 10:
        return np.array([])
    
    z_min = float(np.nanmin(z_valid))
    z_max = float(np.nanmax(z_valid))
    
    # If the data range is invalid, return an empty array
    if not np.isfinite(z_min) or not np.isfinite(z_max) or z_max <= z_min:
        return np.array([])
    
    # Transform data into norm space (visual space)
    # Note: norm typically expects input within \[vmin, vmax], but we can use it directly
    try:
        # Get vmin and vmax from norm
        vmin = norm.vmin if hasattr(norm, 'vmin') and np.isfinite(norm.vmin) else z_min
        vmax = norm.vmax if hasattr(norm, 'vmax') and np.isfinite(norm.vmax) else z_max
        
        # Clip valid data to norm range, then convert to visual space
        z_clipped = np.clip(z_valid, vmin, vmax)
        t_values = norm(z_clipped)
        # Ensure this is a plain array, not a MaskedArray
        if isinstance(t_values, np.ma.MaskedArray):
            t_values = np.array(t_values.data[~t_values.mask] if t_values.mask is not np.ma.nomask else t_values.data)
        else:
            t_values = np.array(t_values)
    except Exception as e:
        # If norm transformation fails, fall back to a linear mapping
        t_values = (z_valid - z_min) / (z_max - z_min) if (z_max > z_min) else np.zeros_like(z_valid)
        t_values = np.array(t_values)
    
    # Use a quantile range to avoid extreme-value compression
    t_lo, t_hi = np.quantile(t_values, [q, 1 - q])
    
    # If the quantile span is too small, fall back to the full range
    if t_hi <= t_lo:
        t_lo = np.min(t_values)
        t_hi = np.max(t_values)
    
    # Compute the span in visual space
    span = t_hi - t_lo
    if span <= 0:
        return np.array([])
    
    # Decide the number of contour levels (adaptive on span, but clamped to \[n_min, n_max])
    # Large span → more contours; small span → fewer contours
    n = int(np.clip(np.round(n_min + (n_max - n_min) * min(span / 0.5, 1.0)), n_min, n_max))
    
    # Leave a margin in visual space and sample evenly (excluding endpoints)
    t_levels = np.linspace(t_lo + margin * span, t_hi - margin * span, n)
    
    # Transform levels back to data space
    try:
        if hasattr(norm, 'inverse'):
            levels = norm.inverse(t_levels)
        else:
            # If there is no `inverse` method, use a linear inverse as a fallback
            levels = t_levels * (z_max - z_min) + z_min
    except Exception as e:
        # If inverse transformation fails, use linear inverse as a fallback
        levels = t_levels * (z_max - z_min) + z_min
    
    # Ensure levels lie within the data range (with a small margin to avoid border overlap)
    eps = 1e-9 * (z_max - z_min) if (z_max > z_min) else 1e-9
    levels = np.clip(levels, z_min + eps, z_max - eps)
    
    # Deduplicate and sort levels
    levels = np.unique(levels)
    levels = np.sort(levels)
    
    # If include_zero=True and the data spans 0, snap the closest contour level to 0.0
    if include_zero and z_min < 0 < z_max and len(levels) > 0:
        # Find the contour level closest to 0
        closest_idx = np.argmin(np.abs(levels))
        closest_level = levels[closest_idx]
        
        # If the closest level is within 1% of the data range around 0, replace it with 0.0
        data_range = z_max - z_min
        if abs(closest_level) < 0.01 * data_range:
            levels[closest_idx] = 0.0
            levels = np.unique(np.sort(levels))
    
    return levels


def auto_contour_levels(
    Z_sub: np.ndarray,
    norm: mpl.colors.Normalize,
    n_min: int = 4,
    n_max: int = 6,
    q: float = 0.08,
    margin: float = 0.05,
    include_zero: bool = True,
    verbose: bool = False
) -> np.ndarray:
    """
    Automatically generate 4–6 contour levels for a single subplot,
    approximately evenly spaced in visual (norm) space.
    
    Parameters
    ----------
    Z_sub:
        2D array of data values for the subplot.
    norm:
        Matplotlib normalization object (should support `inverse`,
        e.g. `SymLogNorm`, `LogNorm`).
    n_min:
        Minimum number of contour levels (default 4).
    n_max:
        Maximum number of contour levels (default 6).
    q:
        Central quantile range used to avoid extreme‑value compression
        (default 0.05 → use 5–95% quantiles).
    margin:
        Margin fraction in visual space (default 0.05 → 5%).
    include_zero:
        If data span 0, snap the contour level closest to 0 to exactly 0.0
        (default True).
    verbose:
        If True, print detailed debug information.
    
    Returns
    -------
    np.ndarray
        Sorted array of contour levels (excluding zmin/zmax).
    """
    # Flatten and drop NaN/inf
    z_flat = Z_sub.flatten()
    z_valid = z_flat[np.isfinite(z_flat)]
    
    # If there are too few valid points, return an empty array
    if len(z_valid) < 10:
        return np.array([])
    
    z_min = float(np.nanmin(z_valid))
    z_max = float(np.nanmax(z_valid))
    
    # If the data range is invalid, return an empty array
    if not np.isfinite(z_min) or not np.isfinite(z_max) or z_max <= z_min:
        return np.array([])
    
    # Transform data into norm (visual) space
    try:
        # Get vmin and vmax from norm
        vmin = norm.vmin if hasattr(norm, 'vmin') and np.isfinite(norm.vmin) else z_min
        vmax = norm.vmax if hasattr(norm, 'vmax') and np.isfinite(norm.vmax) else z_max
        
        # Clip valid data to norm range, then convert to visual space
        z_clipped = np.clip(z_valid, vmin, vmax)
        t_values = norm(z_clipped)
        # Ensure this is a plain array, not a MaskedArray
        if isinstance(t_values, np.ma.MaskedArray):
            t_values = np.array(t_values.data[~t_values.mask] if t_values.mask is not np.ma.nomask else t_values.data)
        else:
            t_values = np.array(t_values)
    except Exception:
        # If norm transformation fails, fall back to a linear mapping
        t_values = (z_valid - z_min) / (z_max - z_min) if (z_max > z_min) else np.zeros_like(z_valid)
        t_values = np.array(t_values)
    
    # Use a quantile range to avoid extreme‑value compression
    t_lo, t_hi = np.quantile(t_values, [q, 1 - q])
    
    # If the quantile span is too small, fall back to the full range
    if t_hi <= t_lo:
        t_lo = np.min(t_values)
        t_hi = np.max(t_values)
    
    # Compute the span in visual space
    span = t_hi - t_lo
    if span <= 0:
        return np.array([])
    
    # Decide the number of contour levels (adaptive on span, but clamped to [n_min, n_max])
    n = int(np.clip(np.round(n_min + (n_max - n_min) * min(span / 0.5, 1.0)), n_min, n_max))
    
    # Leave a margin in visual space and sample evenly spaced targets (excluding endpoints)
    t_levels = np.linspace(t_lo + margin * span, t_hi - margin * span, n)

    # Invert the transform to obtain raw levels in data space
    try:
        if hasattr(norm, "inverse"):
            levels_raw = norm.inverse(t_levels)
        else:
            levels_raw = t_levels * (z_max - z_min) + z_min
    except Exception:
        levels_raw = t_levels * (z_max - z_min) + z_min

    # Debug printing for raw levels and corresponding visual‑space quantiles
    # (kept commented out; enable for detailed diagnostics if needed)
    # if verbose:
    #     print(f"\n[auto_contour_levels] Raw levels:")
    #     print(f"  - visual‑space quantiles (t_levels): {np.round(t_levels, 4).tolist()}")
    #     print(f"  - corresponding raw levels: {np.round(levels_raw, 4).tolist()}")
    #     print(f"  - data range: [{z_min:.4g}, {z_max:.4g}]")
    #     print(f"  - visual span: [{t_lo:.4f}, {t_hi:.4f}], span={span:.4f}")
    #     print(f"  - number of raw contour candidates: {n}")

    # Main path: raw → `snap_levels_to_nice` (which chooses n visually nice levels)
    levels = snap_levels_to_nice(
        levels=levels_raw,
        z_min=z_min,
        z_max=z_max,
        n_min=n_min,
        prefer_integers=True,
        norm=norm,
        verbose=verbose,
    )
    return levels


def pick_nice_levels_by_norm_targets(
    Z_sub: np.ndarray,
    norm: mpl.colors.Normalize,
    n_min: int = 5,
    n_max: int = 6,
    q: float = 0.08,
    margin: float = 0.05,
    include_zero: bool = True,
    verbose: bool = False  # If True, print detailed debug information (mainly for "relative‑to‑minimum" mode)
) -> np.ndarray:
    """
    Choose visually nice contour levels based on evenly spaced targets
    in visual (norm) space, then snapping to multi‑scale "nice" values.
    
    Parameters
    ----------
    Z_sub:
        2D array of data values for the subplot.
    norm:
        Matplotlib normalization object (should support `inverse`,
        e.g. `SymLogNorm`, `LogNorm`).
    n_min:
        Minimum number of contour levels (default 4).
    n_max:
        Maximum number of contour levels (default 6).
    q:
        Central quantile range used to avoid extreme‑value compression
        (default 0.08 → 8–92% quantiles).
    margin:
        Margin fraction in visual space (default 0.05 → 5%).
    include_zero:
        If data span 0, include 0.0 as one candidate level (default True).
    
    Returns
    -------
    np.ndarray
        Sorted array of contour levels (excluding zmin/zmax, all "nice" values).
    """
    # Flatten and drop NaN/inf
    z_flat = Z_sub.flatten()
    z_valid = z_flat[np.isfinite(z_flat)]
    
    # If there are too few valid points, return an empty array
    if len(z_valid) < 10:
        return np.array([])
    
    z_min = float(np.nanmin(z_valid))
    z_max = float(np.nanmax(z_valid))
    
    # If the data range is invalid, return an empty array
    if not np.isfinite(z_min) or not np.isfinite(z_max) or z_max <= z_min:
        return np.array([])
    
    # Transform data into norm (visual) space
    try:
        # Get vmin and vmax from norm
        vmin = norm.vmin if hasattr(norm, 'vmin') and np.isfinite(norm.vmin) else z_min
        vmax = norm.vmax if hasattr(norm, 'vmax') and np.isfinite(norm.vmax) else z_max
        
        # Clip valid data to norm range, then convert to visual space
        z_clipped = np.clip(z_valid, vmin, vmax)
        t_values = norm(z_clipped)
        # Ensure this is a plain array, not a MaskedArray
        if isinstance(t_values, np.ma.MaskedArray):
            t_values = np.array(t_values.data[~t_values.mask] if t_values.mask is not np.ma.nomask else t_values.data)
        else:
            t_values = np.array(t_values)
    except Exception:
        # If norm transformation fails, fall back to a linear mapping
        t_values = (z_valid - z_min) / (z_max - z_min) if (z_max > z_min) else np.zeros_like(z_valid)
        t_values = np.array(t_values)
    
    # Use a quantile range to avoid extreme‑value compression
    t_lo, t_hi = np.quantile(t_values, [q, 1 - q])
    
    # If the quantile span is too small, fall back to the full range
    if t_hi <= t_lo:
        t_lo = np.min(t_values)
        t_hi = np.max(t_values)
    
    # Compute the span in visual space
    span = t_hi - t_lo
    if span <= 0:
        return np.array([])
    
    # Decide the number of contour levels (adaptive on span, but clamped to [n_min, n_max])
    n = int(np.clip(np.round(n_min + (n_max - n_min) * min(span / 0.5, 1.0)), n_min, n_max))

    
    # Leave a margin in visual space and sample evenly spaced targets (excluding endpoints)
    t_targets = np.linspace(t_lo + margin * span, t_hi - margin * span, n)
    
    # Generate multi‑scale "nice" candidate values
    # Use {1, 2, 5} × 10^k steps to cover multiple orders of magnitude
    data_range = z_max - z_min
    if data_range <= 0:
        return np.array([])
    
    # Determine the order‑of‑magnitude range for z_min and z_max
    # so that the full data range is covered
    if abs(z_min) > 1e-10:
        k_min = int(np.floor(np.log10(abs(z_min))))
    else:
        k_min = int(np.floor(np.log10(max(abs(z_max), 1e-10)))) - 3  # If z_min is tiny, extend 3 orders downward
    
    if abs(z_max) > 1e-10:
        k_max = int(np.ceil(np.log10(abs(z_max))))
    else:
        k_max = int(np.ceil(np.log10(max(abs(z_min), 1e-10)))) + 3  # If z_max is tiny, extend 3 orders upward
    
    # Ensure the exponent range is wide enough to cover z_min → z_max
    k_min = min(k_min, int(np.floor(np.log10(max(abs(z_min), 1e-10)))))
    k_max = max(k_max, int(np.ceil(np.log10(max(abs(z_max), 1e-10)))))
    
    # Constrain k_min to avoid overly small steps which would create
    # too many candidates (and risk memory issues)
    # The minimum step should keep the number of candidates below MAX_CANDIDATES_PER_STEP
    MAX_CANDIDATES_PER_STEP = 1000  # Hard cap on candidates per step to avoid memory blow‑up
    if data_range > 0:
        # Compute the minimum allowed step size: data_range / MAX_CANDIDATES_PER_STEP
        min_step_allowed = data_range / MAX_CANDIDATES_PER_STEP
        k_min_allowed = int(np.floor(np.log10(min_step_allowed)))
        # Enforce k_min ≥ k_min_allowed (i.e. step ≥ min_step_allowed)
        k_min = max(k_min, k_min_allowed)
        # Also clamp k_min to −6 (i.e. step ≥ 1e−6) as a hard lower bound
        k_min = max(k_min, -6)
    
    # Generate candidates using multiple steps (1, 2, 5) across exponents
    candidates = []
    
    # Loop from the smallest to largest exponent, using 1, 2, 5 as step multipliers
    for k in range(k_min, k_max + 1):
        base = 10 ** k
        for multiplier in [1, 2, 5]:
            step = multiplier * base
            # Generate candidates for this step to cover [z_min, z_max]
            grid_start = np.floor(z_min / step) * step
            grid_end = np.ceil(z_max / step) * step
            
            # Estimate how many candidates would be generated
            if step > 0:
                num_points = int((grid_end - grid_start) / step) + 1
            else:
                num_points = 0
            
            # If too many candidates would be generated, skip this step
            if num_points > MAX_CANDIDATES_PER_STEP:
                continue
            # Generate grid points
            if num_points > 0:
                grid_points = np.arange(grid_start, grid_end + step, step)
                candidates.extend(grid_points.tolist())
    
    # If include_zero and data span 0, ensure 0 is present
    if include_zero and z_min < 0 < z_max:
        if 0.0 not in candidates:
            candidates.append(0.0)
    
    # If the data range is small (max − min ≤ 5),
    # add candidate contour levels spaced by 0.5 to get finer detail
    if data_range <= 5.0:
        half_step_candidates = []
        # Generate 0.5‑spaced candidates over [z_min, z_max]
        start_i = int(np.floor(z_min * 2))
        end_i = int(np.ceil(z_max * 2))
        for i in range(start_i, end_i + 1):
            candidate = i * 0.5
            if z_min <= candidate <= z_max:
                half_step_candidates.append(candidate)
        candidates.extend(half_step_candidates)
    
    # Deduplicate and sort all candidates
    candidates = np.unique(np.array(candidates))
    candidates = np.sort(candidates)
    
    # Filter candidates: keep only those strictly inside the data range (with margin)
    eps = 1e-9 * data_range if data_range > 0 else 1e-9
    candidates = candidates[(candidates >= z_min + eps) & (candidates <= z_max - eps)]
    
    if len(candidates) == 0:
        return np.array([])
    
    # Map candidates into visual (norm) space
    try:
        if hasattr(norm, 'inverse'):
            # For candidates we need to apply the forward (data → visual) mapping
            t_candidates = norm(candidates)
            # Ensure this is a plain array
            if isinstance(t_candidates, np.ma.MaskedArray):
                t_candidates = np.array(t_candidates.data[~t_candidates.mask] if t_candidates.mask is not np.ma.nomask else t_candidates.data)
            else:
                t_candidates = np.array(t_candidates)
        else:
            # If no norm is available, fall back to linear mapping
            t_candidates = (candidates - z_min) / (z_max - z_min) if (z_max > z_min) else np.zeros_like(candidates)
    except Exception:
        # If mapping fails, fall back to linear mapping
        t_candidates = (candidates - z_min) / (z_max - z_min) if (z_max > z_min) else np.zeros_like(candidates)
    
    # Use percentile‑based targets for contour levels
    # (avoid values too close to the 0/1 visual boundaries)
    percentile_targets = [10, 30, 50, 75, 97]  # Percentile targets
    
    # Compute data percentiles (in data space)
    z_flat = Z_sub.flatten()
    z_valid = z_flat[np.isfinite(z_flat)]
    percentile_values = np.percentile(z_valid, percentile_targets)
    
    # Map percentile values into visual (norm) space
    try:
        if hasattr(norm, 'inverse'):
            t_percentiles = norm(percentile_values)
            if isinstance(t_percentiles, np.ma.MaskedArray):
                t_percentiles = np.array(t_percentiles.data[~t_percentiles.mask] if t_percentiles.mask is not np.ma.nomask else t_percentiles.data)
            else:
                t_percentiles = np.array(t_percentiles)
        else:
            t_percentiles = (percentile_values - z_min) / (z_max - z_min) if (z_max > z_min) else np.zeros_like(percentile_values)
    except Exception:
        t_percentiles = (percentile_values - z_min) / (z_max - z_min) if (z_max > z_min) else np.zeros_like(percentile_values)
    
    # Select the candidate level closest to each percentile target
    selected_levels = []
    last_selected_t = -np.inf
    
    for i, (percentile_val, t_percentile) in enumerate(zip(percentile_values, t_percentiles)):
        # Enforce monotonicity by only allowing candidates above last_selected_t
        valid_mask = t_candidates > last_selected_t
        if not np.any(valid_mask):
            continue
        
        valid_t_candidates = t_candidates[valid_mask]
        valid_candidates = candidates[valid_mask]
        
        # Find the candidate closest to the percentile position
        nearest_idx = np.argmin(np.abs(valid_t_candidates - t_percentile))
        selected_candidate = valid_candidates[nearest_idx]
        selected_t = valid_t_candidates[nearest_idx]
        
        selected_levels.append(selected_candidate)
        last_selected_t = selected_t
    
    if len(selected_levels) < 2:
        return np.array([])
    
    levels = np.array(selected_levels)
    # If we still have too few levels, try using a smaller step to densify candidates
    if len(levels) < n_min:
        # Compute the current minimum step between candidates
        if len(candidates) > 1:
            min_step = np.min(np.diff(np.sort(candidates)))
        else:
            min_step = data_range / 10.0
        
        # Add a smaller step (halve the step size)
        new_step = min_step / 2.0
        new_step = nice_step(new_step)  # Snap to a "nice" step
        
        # Constrain new_step so it does not become too small (memory safety)
        min_step_allowed = data_range / MAX_CANDIDATES_PER_STEP if data_range > 0 else 1e-6
        new_step = max(new_step, min_step_allowed, 1e-6)  # Hard lower bound: step ≥ 1e−6
        
        # Generate new candidates on this finer grid
        grid_start = np.floor(z_min / new_step) * new_step
        grid_end = np.ceil(z_max / new_step) * new_step
        
        # Estimate how many new candidates would be generated
        if new_step > 0:
            num_points_new = int((grid_end - grid_start) / new_step) + 1
        else:
            num_points_new = 0
        
        # If too many candidates would be generated, skip this refinement
        if num_points_new > MAX_CANDIDATES_PER_STEP:
            if verbose:
                print(f"    Skip adding smaller step={new_step:.6g}: would need {num_points_new} candidates, exceeding limit {MAX_CANDIDATES_PER_STEP}")
            # Do not add new candidates; keep existing ones
            all_candidates = candidates.copy()
        else:
            new_candidates = np.arange(grid_start, grid_end + new_step, new_step)
            new_candidates = new_candidates[(new_candidates >= z_min + eps) & (new_candidates <= z_max - eps)]
            # Merge new candidates into the existing pool
            all_candidates = np.unique(np.concatenate([candidates, new_candidates]))
        
        # If data range is between 1 and 10, add 0.5‑spaced candidates
        if z_min >= 1.0 and z_max <= 10.0:
            half_step_candidates = []
            for i in range(int(np.ceil(z_min * 2)), int(np.floor(z_max * 2)) + 1):
                candidate = i * 0.5
                if candidate >= z_min and candidate <= z_max:
                    half_step_candidates.append(candidate)
            all_candidates = np.unique(np.concatenate([all_candidates, half_step_candidates]))
        
        all_candidates = np.sort(all_candidates)
        
        # Re‑map refined candidates into visual space
        try:
            if hasattr(norm, 'inverse'):
                t_all_candidates = norm(all_candidates)
                if isinstance(t_all_candidates, np.ma.MaskedArray):
                    t_all_candidates = np.array(t_all_candidates.data[~t_all_candidates.mask] if t_all_candidates.mask is not np.ma.nomask else t_all_candidates.data)
                else:
                    t_all_candidates = np.array(t_all_candidates)
            else:
                t_all_candidates = (all_candidates - z_min) / (z_max - z_min) if (z_max > z_min) else np.zeros_like(all_candidates)
        except Exception:
            t_all_candidates = (all_candidates - z_min) / (z_max - z_min) if (z_max > z_min) else np.zeros_like(all_candidates)
        
        # Re‑select using the same percentile‑based method
        percentile_values_new = np.percentile(z_valid, percentile_targets)
        
        # Map percentile values into visual (norm) space
        try:
            if hasattr(norm, 'inverse'):
                t_percentiles_new = norm(percentile_values_new)
                if isinstance(t_percentiles_new, np.ma.MaskedArray):
                    t_percentiles_new = np.array(t_percentiles_new.data[~t_percentiles_new.mask] if t_percentiles_new.mask is not np.ma.nomask else t_percentiles_new.data)
                else:
                    t_percentiles_new = np.array(t_percentiles_new)
            else:
                t_percentiles_new = (percentile_values_new - z_min) / (z_max - z_min) if (z_max > z_min) else np.zeros_like(percentile_values_new)
        except Exception:
            t_percentiles_new = (percentile_values_new - z_min) / (z_max - z_min) if (z_max > z_min) else np.zeros_like(percentile_values_new)
        
        selected_levels = []
        last_selected_t = -np.inf
        
        for percentile_val, t_percentile in zip(percentile_values_new, t_percentiles_new):
            valid_mask = t_all_candidates > last_selected_t
            if not np.any(valid_mask):
                continue
            
            valid_t_candidates = t_all_candidates[valid_mask]
            valid_candidates = all_candidates[valid_mask]
            
            nearest_idx = np.argmin(np.abs(valid_t_candidates - t_percentile))
            selected_candidate = valid_candidates[nearest_idx]
            selected_t = valid_t_candidates[nearest_idx]
            
            selected_levels.append(selected_candidate)
            last_selected_t = selected_t
        
        if len(selected_levels) > 0:
            levels = np.array(selected_levels)
    
    # Final safety: ensure all contour levels lie strictly within data range
    eps_final = max(eps, 1e-9 * data_range) if data_range > 0 else 1e-9
    levels = np.clip(levels, z_min + eps_final, z_max - eps_final)
    levels = np.unique(np.sort(levels))
    return levels


def snap_to_nice_number(x: float, data_range: Optional[float] = None, z_min: Optional[float] = None, z_max: Optional[float] = None) -> float:
    """
    Snap a numeric value to a "visually nice" value:
    - Prefer integers when possible.
    - If a decimal is needed, keep one significant digit.
    - Consider many candidates so the result stays close to the original.
    - Automatically choose coarse vs. fine granularity based on data range.
    
    Parameters
    ----------
    x:
        Input value.
    data_range:
        Optional data range (z_max - z_min) used to choose granularity.
    z_min:
        Optional minimum data value.
    z_max:
        Optional maximum data value.
    
    Returns
    -------
    float
        Snapped "nice" value.
    """
    if x == 0.0:
        return 0.0
    
    abs_x = abs(x)
    sign = 1.0 if x >= 0 else -1.0
    
    # Compute order of magnitude of x
    if abs_x > 0:
        k = int(np.floor(np.log10(abs_x)))
        magnitude = 10 ** k
    else:
        return 0.0
    
    # Choose coarse vs. fine granularity based on data range
    use_fine_grain = True
    if data_range is not None and z_min is not None and z_max is not None:
        if z_min > 0:
            range_ratio = z_max / z_min
            if range_ratio > 10:
                use_fine_grain = False
        else:
            if data_range is not None and data_range > 10:
                use_fine_grain = False

    if abs_x > 1.0:
        # For values > 1:
        # - In 1–10 range we always allow fine granularity (keep 4, 6, 9, etc.).
        # - For larger magnitudes we choose coarse/fine based on `use_fine_grain`.
        integer_candidates = []

        if magnitude >= 100:
            # For ≥ 10^2: use templates which depend on coarse vs. fine mode
            if use_fine_grain:
                base_template_10_100 = [10, 12, 15, 18, 20, 25, 30, 35, 40, 45,
                                        50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
            else:
                base_template_10_100 = [10, 20, 50, 80, 100]

            for template_val in base_template_10_100:
                candidate = template_val * (magnitude // 10)
                if candidate >= 1:
                    integer_candidates.append(int(candidate))
            if abs_x >= magnitude * 10:
                for template_val in base_template_10_100:
                    candidate_next = template_val * (magnitude // 10) * 10
                    if candidate_next >= 1:
                        integer_candidates.append(int(candidate_next))

        elif magnitude >= 10:
            # 10–100 range
            if use_fine_grain:
                base_template_10_100 = [10, 12, 15, 18, 20, 25, 30, 35, 40, 45,
                                        50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
            else:
                base_template_10_100 = [10, 20, 50, 80, 100]
            integer_candidates.extend(base_template_10_100)
            if abs_x >= 8 * magnitude:
                for template_val in base_template_10_100:
                    integer_candidates.append(template_val * 10)

        elif magnitude >= 1:
            # 1–10 range: always use integer 1–10, unaffected by coarse/fine mode
            base_template_1_10 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            integer_candidates.extend(base_template_1_10)

        else:
            # 0.1–1 range: scale the 1–10 template by the magnitude
            base_template_1_10 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            for template_val in base_template_1_10:
                candidate = template_val * magnitude
                if candidate >= 1:
                    integer_candidates.append(int(candidate))

        integer_candidates = sorted(set(integer_candidates))
        # Prefer the closest value that does not exceed z_max (if provided)
        if z_max is not None and len(integer_candidates) > 0:
            le_candidates = [c for c in integer_candidates if c <= z_max]
            if le_candidates:
                nearest_integer = min(le_candidates, key=lambda c: abs(abs_x - c))
            else:
                nearest_integer = min(integer_candidates, key=lambda c: abs(abs_x - c))
        else:
            nearest_integer = min(integer_candidates, key=lambda c: abs(abs_x - c))
        result = sign * float(nearest_integer)
    else:
        # For values ≤ 1:
        # fine mode uses all one‑digit candidates; coarse mode uses 0.2/0.5/1.0 (2/5/10 series)
        normalized = abs_x / magnitude

        if use_fine_grain:
            candidates = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        else:
            candidates = [2.0, 5.0, 10.0]

        nearest_candidate = min(candidates, key=lambda c: abs(normalized - c))
        if nearest_candidate == 10.0:
            result = sign * 1.0 * (magnitude * 10)
        else:
            result = sign * nearest_candidate * magnitude
    
    return result


def nice_step(x: float) -> float:
    """
    Return a visually nice step size close to x,
    of the form {1, 2, 5, 10} × 10^k.
    
    Parameters
    ----------
    x:
        Input value.
    
    Returns
    -------
    float
        Nearest "nice" step (1, 2, 5, 10 or a power‑of‑10 multiple).
    """
    if x <= 0:
        return 1.0
    
    # Compute order of magnitude
    magnitude = 10 ** np.floor(np.log10(x))
    
    # Normalize to the 1–10 range
    normalized = x / magnitude
    
    # Choose the closest of {1, 2, 5, 10}
    if normalized <= 1.5:
        nice_normalized = 1.0
    elif normalized <= 3.5:
        nice_normalized = 2.0
    elif normalized <= 7.5:
        nice_normalized = 5.0
    else:
        nice_normalized = 10.0
    
    return nice_normalized * magnitude


def snap_levels_to_nice(
    levels: np.ndarray,
    z_min: float,
    z_max: float,
    n_min: int = 4,
    prefer_integers: bool = True,
    norm: Optional[mpl.colors.Normalize] = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Select n visually nice contour levels directly from a pre‑computed
    candidate set, keeping them roughly uniform in visual (norm) space.
    """
    if len(levels) == 0:
        return levels

    # Pre‑processing
    levels = np.unique(np.sort(np.array(levels, dtype=float)))
    data_range = z_max - z_min
    if not np.isfinite(data_range) or data_range <= 0:
        return np.clip(levels, z_min, z_max)

    n = len(levels)

    # ---------- Generate candidates: local grid (integers / whole numbers) + coarse 1‑2‑5 ----------
    def generate_125(zlo: float, zhi: float) -> np.ndarray:
        """Generate 1‑2‑5 × 10^k style ticks within [zlo, zhi]."""
        if zhi <= 0:
            return np.array([])
        eps = 1e-12
        lo = max(zlo, eps)
        k_min = int(np.floor(np.log10(lo)))
        k_max = int(np.ceil(np.log10(zhi)))
        vals = []
        for k in range(k_min - 1, k_max + 2):
            base = 10.0 ** k
            for m in [1.0, 2.0, 5.0]:
                v = m * base
                if zlo - eps <= v <= zhi + eps:
                    vals.append(v)
        return np.array(sorted(set(vals)), dtype=float)

    # if verbose:
    #     print("\n[contour] snap_levels_to_nice()")
    #     print(f"  - data range: [{z_min:.4g}, {z_max:.4g}], n={n}")
    #     print(f"  - raw levels: {np.round(levels, 4).tolist()}")

    # 1) Choose a step for the local grid based on data range (prefer coarser)
    step_list = [20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2]
    step_chosen = step_list[-1]
    for st in step_list:
        if np.floor(data_range / st) > n:
            step_chosen = st
            break

    # Use step_chosen to generate a local grid
    start = np.ceil(z_min / step_chosen) * step_chosen
    # Ensure the grid at least covers up to z_max
    k_max_local = int(np.floor((z_max - start) / step_chosen)) + 2
    grid_vals = start + step_chosen * np.arange(0, max(k_max_local, 1))
    eps_g = 1e-12 * max(1.0, abs(z_max))
    local_grid = grid_vals[(grid_vals > z_min + eps_g) & (grid_vals < z_max - eps_g)]

    # 2) Overlay coarse 1‑2‑5 × 10^k ticks
    coarse_125 = generate_125(z_min, z_max)

    candidates = np.unique(
        np.concatenate(
            [
                local_grid.astype(float),
                coarse_125.astype(float),
            ]
        )
    )

    # if verbose:
    #     print(f"  - step_chosen for local grid: {step_chosen}")
    #     print(f"  - local_grid: {np.round(local_grid,4).tolist()}")
    #     print(f"  - coarse 1-2-5: {np.round(coarse_125,4).tolist()}")
    #     print(f"  - candidates (local grid + 1-2-5): {np.round(candidates,4).tolist()}")

    if len(candidates) == 0:
        # In theory this should not happen because local_grid should always cover
        return np.clip(levels, z_min, z_max)

    # ---------- Map candidates into visual (norm) space ----------
    if norm is not None:
        try:
            t_candidates = norm(candidates)
            if isinstance(t_candidates, np.ma.MaskedArray):
                t_candidates = np.array(t_candidates.data[~t_candidates.mask] if t_candidates.mask is not np.ma.nomask else t_candidates.data)
            else:
                t_candidates = np.array(t_candidates)
        except Exception:
            norm = None  # fallback
    if norm is None:
        t_candidates = (candidates - z_min) / data_range

    # t_targets: keep visual spacing uniform (use t‑range of original levels)
    if norm is not None:
        t_raw = norm(levels)
        if isinstance(t_raw, np.ma.MaskedArray):
            t_raw = np.array(t_raw.data[~t_raw.mask] if t_raw.mask is not np.ma.nomask else t_raw.data)
        else:
            t_raw = np.array(t_raw)
    else:
        t_raw = (levels - z_min) / data_range

    t_lo = float(np.min(t_raw))
    t_hi = float(np.max(t_raw))
    if not np.isfinite(t_lo) or not np.isfinite(t_hi) or t_hi <= t_lo:
        return np.clip(levels, z_min, z_max)

    t_targets = np.linspace(t_lo, t_hi, n)

    # t_gap_min: prevent levels from becoming too crowded
    t_span = t_hi - t_lo
    # print(f"  - t_span: {t_span}")
    t_gap_min = t_span / (n * 3.0)

    # if verbose:
    #     print(f"  - t_gap_min: {t_gap_min}")
    #     print(f"  - t_targets: {np.round(t_targets, 4).tolist()}")

    # ---------- Backtracking selection ----------
    def try_select(cand_vals: np.ndarray) -> Optional[np.ndarray]:
        if cand_vals.size == 0:
            return None
        try:
            t_cand = norm(cand_vals) if norm is not None else (cand_vals - z_min) / data_range
            if isinstance(t_cand, np.ma.MaskedArray):
                t_cand = np.array(t_cand.data[~t_cand.mask] if t_cand.mask is not np.ma.nomask else t_cand.data)
            else:
                t_cand = np.array(t_cand)
        except Exception:
            t_cand = (cand_vals - z_min) / data_range

        best_solution: Optional[np.ndarray] = None

        def backtrack(idx, prev_t, prev_val, chosen):
            nonlocal best_solution
            if idx == n:
                best_solution = np.array(chosen, dtype=float)
                return True
            tgt = t_targets[idx]
            order = np.argsort(np.abs(t_cand - tgt))
            for k in order:
                y = cand_vals[k]
                ty = t_cand[k]
                if idx > 0:
                    if y <= prev_val:
                        continue
                    if ty - prev_t < t_gap_min:
                        continue
                chosen.append(y)
                if backtrack(idx + 1, ty, y, chosen):
                    return True
                chosen.pop()
            return False

        ok = backtrack(0, t_lo - t_gap_min, z_min - data_range, [])
        return np.unique(np.sort(best_solution)) if ok and best_solution is not None else None

    # 1) First, try using 1‑2‑5 + raw single‑sig candidates
    solution = try_select(candidates)

    # 2) If still insufficient, widen: add mantissas [1,1.5,2,3,5,7] × 10^k
    if solution is None or len(solution) < n_min:
        if verbose:
            print("  - widen candidates with mantissas [1,1.5,2,3,5,7]")
        dense_vals = []
        if z_max > 0:
            k_min = int(np.floor(np.log10(max(z_min, 1e-12))))
            k_max = int(np.ceil(np.log10(z_max)))
            for k in range(k_min - 1, k_max + 2):
                base = 10.0 ** k
                for m in [1.0, 1.5, 2.0, 3.0, 5.0, 7.0]:
                    v = m * base
                    if z_min <= v <= z_max:
                        dense_vals.append(v)
        candidates_dense = np.unique(np.concatenate([candidates, np.array(dense_vals, dtype=float)]))
        solution = try_select(candidates_dense)

    # 3) If still not enough, add all raw levels as candidates
    if (solution is None or len(solution) < n_min) and len(levels) > 0:
        if verbose:
            print("  - final fallback: add all raw levels to candidates")
        candidates_all = np.unique(np.concatenate([candidates, levels]))
        solution = try_select(candidates_all)

    if solution is not None and len(solution) > 0:
        # Upper Anchor: push the last contour level close to a nice value near z_max
        levels_chosen = np.array(solution, dtype=float)
        if levels_chosen.size >= 2 and candidates.size > 0:
            # Add a buffer below z_max (e.g. 2% of data range) to avoid anchoring too close
            eps_cap = max(1e-9 * max(1.0, abs(z_max)), 0.08 * data_range)
            upper_cap = z_max - eps_cap
            anchor_base = levels_chosen[-2]
            upper_pool = candidates[(candidates > anchor_base) & (candidates < upper_cap)]
            if upper_pool.size > 0:
                # Enforce a minimum visual spacing of at least t_gap_min
                anchor_level = levels_chosen[-1]
                if norm is not None:
                    try:
                        t_prev = float(norm(anchor_base))
                    except Exception:
                        t_prev = (anchor_base - z_min) / data_range
                else:
                    t_prev = (anchor_base - z_min) / data_range

                best_anchor = anchor_level
                for cand in np.sort(upper_pool)[::-1]:  # Try candidates from large to small
                    if norm is not None:
                        try:
                            t_cand = float(norm(cand))
                        except Exception:
                            t_cand = (cand - z_min) / data_range
                    else:
                        t_cand = (cand - z_min) / data_range
                    if t_cand - t_prev >= 2*t_gap_min:
                        best_anchor = cand
                        break
                levels_chosen[-1] = best_anchor

        # if verbose:
        #     print(f"  - final chosen levels (with upper anchor): {np.round(levels_chosen, 4).tolist()}")
        return np.unique(np.sort(levels_chosen))

    # Fallback: return the original levels (clipped)
    fallback = np.clip(levels, z_min, z_max)
    if verbose:
        print("  - backtrack failed, fallback to clipped raw levels")
        print(f"  - fallback levels: {np.round(fallback, 4).tolist()}")
    return fallback


def filter_contour_levels_to_target_count(
    levels: np.ndarray,
    z_min: float,
    z_max: float,
    master_levels: Optional[np.ndarray] = None,
    target_min_count: int = 4,
    target_max_count: int = 5,
    verbose: bool = False
) -> np.ndarray:
    """
    Filter contour levels down to a target count (typically 4–5 lines).
    
    Strategy: always keep the minimum and maximum levels, then choose
    the remaining levels roughly uniformly from the interior.
    
    Parameters
    ----------
    levels:
        Current contour level array (sorted).
    z_min:
        Data minimum value.
    z_max:
        Data maximum value.
    master_levels:
        Optional "master" contour array used to supplement levels.
    target_min_count:
        Minimum desired number of contour levels (default 4).
    target_max_count:
        Maximum desired number of contour levels (default 5).
    verbose:
        If True, print detailed information.
    
    Returns
    -------
    np.ndarray
        Filtered array of contour levels.
    """
    if len(levels) == 0:
        return levels
    
    levels = np.sort(np.array(levels, dtype=float))
    
    # If we exceed the target maximum, down‑select to the target range
    if len(levels) > target_max_count:
        levels_before_limit = levels.copy()
        
        # Keep minimum and maximum levels
        min_level = levels[0]
        max_level = levels[-1]
        
        # Extract interior levels (excluding min/max)
        if len(levels) > 2:
            middle_levels = levels[1:-1]
            
            # Number of interior levels needed: total target − 2 (for min/max)
            target_middle_count = target_max_count - 2  # 5 - 2 = 3
            
            if len(middle_levels) >= target_middle_count:
                # Select interior levels approximately uniformly
                indices = np.linspace(0, len(middle_levels) - 1, target_middle_count, dtype=int)
                selected_middle = middle_levels[indices]
                levels = np.sort(np.concatenate([[min_level], selected_middle, [max_level]]))
            else:
                # If not enough interior levels, keep all of them (total < target)
                levels = np.sort(np.concatenate([[min_level], middle_levels, [max_level]]))
        
        if verbose:
            print(f"    - Step5 [filter contour]: reduced from {len(levels_before_limit)} to {len(levels)} levels (keep min/max, choose interior uniformly)")
    
    # If we have fewer than the target minimum, try to supplement from master_levels
    elif len(levels) < target_min_count:
        levels_before_limit = levels.copy()
        if len(levels) > 0 and master_levels is not None and len(master_levels) > 0:
            min_level = levels[0]
            max_level = levels[-1]
            
            # Find candidate master levels inside data range but not in current levels
            candidate_levels = master_levels[
                (master_levels >= z_min) & 
                (master_levels <= z_max) &
                ~np.isin(master_levels, levels)
            ]
            
            if len(candidate_levels) > 0:
                # Number of levels needed to reach target_min_count
                need_count = target_min_count - len(levels)
                if need_count > 0:
                    # Prefer levels around the middle of candidate range
                    if len(candidate_levels) >= need_count:
                        # Evenly spaced selection
                        indices = np.linspace(0, len(candidate_levels) - 1, need_count, dtype=int)
                        selected = candidate_levels[indices]
                        levels = np.sort(np.concatenate([levels, selected]))
                    else:
                        # If not enough candidates, add all of them
                        levels = np.sort(np.concatenate([levels, candidate_levels]))
                    
                    if verbose:
                        print(f"    - Step5 [supplement contour]: increased from {len(levels_before_limit)} to {len(levels)} levels")
    
    # Final check: ensure contour count lies within the desired range
    if len(levels) > target_max_count:
        levels_before_limit = levels.copy()
        min_level = levels[0]
        max_level = levels[-1]
        if len(levels) > 2:
            middle_levels = levels[1:-1]
            target_middle_count = target_max_count - 2
            if len(middle_levels) >= target_middle_count:
                indices = np.linspace(0, len(middle_levels) - 1, target_middle_count, dtype=int)
                selected_middle = middle_levels[indices]
                levels = np.sort(np.concatenate([[min_level], selected_middle, [max_level]]))
            else:
                levels = np.sort(np.concatenate([[min_level], middle_levels, [max_level]]))
        
        if verbose:
            print(f"    - Final check [limit contour count]: forced from {len(levels_before_limit)} to {len(levels)} levels")
    
    return levels


def save_hatch_legend(save_path: str):
    """Generate and save a hatch legend figure."""
    setup_plot_style()
    hatch_map = {
        'HTS Tape Cost': '/',
        'Power Supply Cost': '|',
        'Coolant Cost': '\\'
    }
    handles = [mpatches.Patch(facecolor='white', edgecolor='black', hatch=hatch, label=label) for label, hatch in hatch_map.items()]
    fig = plt.figure(figsize=(10, 1.5))
    ax = fig.add_subplot(111)
    ax.axis('off')
    legend = ax.legend(
        handles=handles,
        loc='center',
        ncol=len(handles),
        frameon=False,
        fontsize=cfg.HEATMAP_FONT_SIZE_COLORBAR,
        title='Direct Magnet System Cost Component',
        handleheight=1.5
    )
    legend.get_frame().set_facecolor('none')
    legend.get_frame().set_edgecolor('none')
    legend.get_title().set_fontsize(cfg.HEATMAP_FONT_SIZE_COLORBAR)
    plt.tight_layout()
    plt.savefig(save_path, dpi=cfg.PLOT_DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Hatch legend saved to: {save_path}")
    
    # =============================================================================
# Plotting functions (new): ΔLCOE heatmap (style consistent with parasitic heatmaps)
# =============================================================================

def plot_delta_lcoe_heatmap_single(
    df: pd.DataFrame,
    output_dir: Path,
    cmap: str,
    norm: mpl.colors.Normalize,
    scenario: str,
    temperature_K: float,
    value_col: str = "delta_lcoe_$/MWh",
    baseline_point: Optional[Tuple[float, float]] = None,  # (Npw_baseline, R_joint_baseline)
    coolant: Optional[str] = None,  # Coolant label for filename disambiguation
    contour_levels: Optional[np.ndarray] = None,  # Optional contour‑level array (auto‑generated if None)
    auto_generate_levels: bool = True,  # Whether to auto‑generate contour levels (default True)
    filename_suffix: str = "",  # Optional filename suffix to distinguish color schemes
    use_stroke: bool = False,  # Whether to use contour halo/stroke styling
):
    """
    Generate and save a ΔLCOE heatmap for a single scenario and temperature,
    without colorbar, axis labels, or tick values.
    
    The style is kept consistent with `plot_parasitic_heatmap_single`.
    
    Parameters
    ----------
    df:
        DataFrame containing data for a single temperature.
    output_dir:
        Directory where the figure will be saved.
    cmap:
        Colormap name or object.
    norm:
        Global normalization object (typically `SymLogNorm`).
    scenario:
        Scenario name.
    temperature_K:
        Temperature value in kelvin.
    value_col:
        Column name for ΔLCOE values (default "delta_lcoe_$/MWh").
    baseline_point:
        Optional baseline point \((Npw_\mathrm{baseline}, R_\mathrm{joint,baseline})\) marked with a star.
    coolant:
        Optional coolant label (e.g. "He" or "H2") used in filenames.
    contour_levels:
        Optional contour‑level array; if None and `auto_generate_levels=True`,
        levels are generated automatically.
    auto_generate_levels:
        Whether to auto‑generate contour levels when not provided (default True).
    """
    font_size = cfg.HEATMAP_FONT_SIZE
    setup_plot_style()

    # Build a label for filenames; include coolant tag if provided
    if coolant is not None:
        label = f"{temperature_K}K_{coolant}"
    else:
        label = f"{temperature_K}K"

    fig, ax = plt.subplots(1, 1, figsize=cfg.HEATMAP_FIGSIZE_SINGLE)

    # Use the same gridding as for the parasitic heatmap: index=Npw, columns=R_joint
    pivot = df.pivot(index="Npw", columns="R_joint", values=value_col)
    X, Y = np.meshgrid(pivot.columns, pivot.index)
    Z = pivot.values

    # Record NaN locations (infeasible points); leave unfilled and mark with hatch
    nan_mask = np.isnan(Z)
    
    # Match the original style: convert R_joint to nΩ and take log10
    X_log = np.log10(X * 1e9)
    
    # Use center coordinates so both `pcolormesh` and `contour` share centers
    # For plotting, use a masked array so NaNs are not drawn; hatch fills will
    # then highlight infeasible points explicitly
    Z_plot = np.ma.masked_array(Z, mask=nan_mask)
    
    # Resolve colormap object (supporting custom/truncated names in config)
    cmap_resolved = cfg.resolve_cmap(cmap) if hasattr(cfg, "resolve_cmap") else cmap
    if isinstance(cmap_resolved, str):
        try:
            cmap_obj = plt.get_cmap(cmap_resolved)
        except ValueError:
                # If matplotlib cannot resolve it, try from seaborn
            try:
                import seaborn as sns
                cmap_obj = sns.color_palette(cmap_resolved, as_cmap=True)
            except (ImportError, ValueError):
                # If all attempts fail, fall back to "viridis"
                cmap_obj = plt.get_cmap("viridis")
    else:
        cmap_obj = cmap_resolved
    
    # Draw the heatmap with center coordinates (`shading="auto"`)
    # so that `pcolormesh` and `contour` are fully consistent
    ax.pcolormesh(X_log, Y, Z_plot, cmap=cmap_obj, norm=norm, shading="auto", alpha=cfg.HEATMAP_ALPHA)
    
    # Check whether we are plotting "relative‑to‑minimum" contours
    is_delta_min = "delta_lcoe_min" in value_col or "relative to scenario minimum" in str(value_col).lower()
    
    # Auto‑generate contour levels if requested
    if auto_generate_levels and contour_levels is None:
        contour_levels = auto_contour_levels(
            Z_sub=Z,
            norm=norm,
            n_min=4,
            n_max=6,
            q=0.08,
            margin=0.03,
            include_zero=True,
            verbose=is_delta_min  # Only print detailed logs for "relative‑to‑minimum" mode
        )
    
    # If we still have no contour levels, require user‑provided levels
    if contour_levels is None or len(contour_levels) == 0:
        if not auto_generate_levels:
            raise ValueError(
                f"plot_delta_lcoe_heatmap_single: contour_levels must be provided when auto_generate_levels=False. "
                f"Got None or empty array for scenario={scenario}, temperature_K={temperature_K}, coolant={coolant}"
            )
        # If auto‑generation failed, skip contour drawing
        contour_levels_to_use = np.array([])
    else:
        contour_levels_to_use = np.array(contour_levels, dtype=float)
    
    # For NaN regions (infeasible points), use hatch fills
    if nan_mask.any():
        # Use the pcolormesh grid to draw hatch patches
        # pcolormesh expects corner points, so expand the grid
        X_log_edges = np.zeros((len(Y) + 1, len(X_log[0]) + 1))
        Y_edges = np.zeros((len(Y) + 1, len(X_log[0]) + 1))
        
        # Compute corner coordinates
        for i in range(len(Y) + 1):
            for j in range(len(X_log[0]) + 1):
                if i == 0:
                    if len(Y) > 1:
                        Y_edges[i, j] = Y[0, 0] - (Y[1, 0] - Y[0, 0]) / 2
                    else:
                        Y_edges[i, j] = Y[0, 0] - 1
                elif i == len(Y):
                    Y_edges[i, j] = Y[-1, 0] + (Y[-1, 0] - Y[-2, 0]) / 2 if len(Y) > 1 else Y[0, 0] + 1
                else:
                    Y_edges[i, j] = (Y[i-1, 0] + Y[i, 0]) / 2
                
                if j == 0:
                    if len(X_log[0]) > 1:
                        X_log_edges[i, j] = X_log[0, 0] - (X_log[0, 1] - X_log[0, 0]) / 2
                    else:
                        X_log_edges[i, j] = X_log[0, 0] - 0.1
                elif j == len(X_log[0]):
                    X_log_edges[i, j] = X_log[0, -1] + (X_log[0, -1] - X_log[0, -2]) / 2 if len(X_log[0]) > 1 else X_log[0, 0] + 0.1
                else:
                    X_log_edges[i, j] = (X_log[0, j-1] + X_log[0, j]) / 2
        
        # Draw a hatched rectangle for each NaN cell
        for i in range(len(Y)):
            for j in range(len(X_log[0])):
                if nan_mask[i, j]:
                    # Draw the hatched rectangle
                    rect = mpatches.Rectangle(
                        (X_log_edges[i, j], Y_edges[i, j]),
                        X_log_edges[i, j+1] - X_log_edges[i, j],
                        Y_edges[i+1, j] - Y_edges[i, j],
                        facecolor='none',
                        edgecolor='black',
                        linewidth=0.5,
                        hatch='///',
                        alpha=0.5
                    )
                    ax.add_patch(rect)

    # --- Contours: contour_levels must be provided at this stage ---
    if contour_levels is None or len(contour_levels) == 0:
        raise ValueError(
            f"plot_delta_lcoe_heatmap_single: contour_levels must be provided. "
            f"Got None or empty array for scenario={scenario}, temperature_K={temperature_K}, coolant={coolant}"
        )
    
    contour_levels_to_use = contour_levels
    
    # Initialize filtered_levels for later colorbar labeling
    filtered_levels = np.array([])
    
    # Helper: detect contour labels that collide with axes or other labels
    def _detect_overlapping_labels(texts, axis, levels_map: dict, margin_axes: float = 0.05, margin_data_y: float = 5.0):
        """
        Detect which contour‑label texts collide with plot borders or
        with other labels.
        
        Parameters
        ----------
        texts:
            List of `clabel` text objects.
        axis:
            Matplotlib `Axes` object.
        levels_map:
            Mapping from text objects to contour level values.
        margin_axes:
            Margin in axes‑normalized coordinates \([0,1]\) for border checks
            (default 5%).
        margin_data_y:
            Margin in data coordinates at the bottom in y (used for x‑axis collision
            checks, default 5.0).
        
        Returns
        -------
        overlapping_levels:
            Levels whose labels collide with plot borders.
        overlapping_pairs:
            Levels whose labels collide with other labels.
        """
        if not texts:
            return [], []
        
        # Get current axis data limits
        xlim = axis.get_xlim()
        ylim = axis.get_ylim()
        
        overlapping_levels = []  # Levels whose labels collide with borders
        label_positions = []  # Store all label positions and corresponding values
        
        # Renderer used to compute bounding boxes
        try:
            renderer = axis.figure.canvas.get_renderer()
        except:
            renderer = None
        
        for txt in texts:
            if not txt.get_visible():
                continue
                
            xdata, ydata = txt.get_position()
            
            # Get the contour level value corresponding to this label
            level_value = levels_map.get(txt, None)
            if level_value is None:
                # If not present in the map, fall back to parsing the text
                try:
                    level_value = float(txt.get_text())
                except (ValueError, TypeError):
                    continue
            
            # Get label bounding box for collision tests
            if renderer is not None:
                try:
                    bbox = txt.get_window_extent(renderer=renderer)
                except:
                    bbox = None
            else:
                bbox = None
            
            # Method 1: use axes coordinates to test against all borders
            x_disp, y_disp = axis.transData.transform((xdata, ydata))
            x_axes, y_axes = axis.transAxes.inverted().transform((x_disp, y_disp))
            
            # Method 2: use data coordinates (especially for bottom x‑axis)
            near_bottom = ydata < (ylim[0] + margin_data_y)
            near_top = ydata > (ylim[1] - margin_data_y)
            near_left = xdata < (xlim[0] + (xlim[1] - xlim[0]) * margin_axes)
            near_right = xdata > (xlim[1] - (xlim[1] - xlim[0]) * margin_axes)
            
            # Axes‑coordinate border test (more stringent)
            near_edge_axes = (
                x_axes < margin_axes
                or x_axes > 1.0 - margin_axes
                or y_axes < margin_axes
                or y_axes > 1.0 - margin_axes
            )
            
            # If label is near any border, mark its level for removal
            if near_bottom or near_top or near_left or near_right or near_edge_axes:
                overlapping_levels.append(level_value)
            
            # Store this label position for later label‑label collision tests
            if bbox is not None:
                label_positions.append({
                    'text': txt,
                    'bbox': bbox,
                    'level': level_value,
                    'x': xdata,
                    'y': ydata,
                    'x_axes': x_axes,
                    'y_axes': y_axes
                })
        
        # Detect collisions between pairs of labels
        overlapping_pairs = []
        for i, label1 in enumerate(label_positions):
            if label1['level'] in overlapping_levels:
                continue  # Already marked for removal due to border collision
            for j, label2 in enumerate(label_positions[i+1:], start=i+1):
                if label2['level'] in overlapping_levels:
                    continue  # Already marked for removal due to border collision
                
                # Test the two bounding boxes for overlap
                bbox1 = label1['bbox']
                bbox2 = label2['bbox']
                
                # Compute overlapping area between bounding boxes
                overlap_x = max(0, min(bbox1.x1, bbox2.x1) - max(bbox1.x0, bbox2.x0))
                overlap_y = max(0, min(bbox1.y1, bbox2.y1) - max(bbox1.y0, bbox2.y0))
                overlap_area = overlap_x * overlap_y
                
                # Consider them colliding if overlap exceeds 30% of the smaller area
                min_area = min(bbox1.width * bbox1.height, bbox2.width * bbox2.height)
                if min_area > 0 and overlap_area / min_area > 0.3:
                    # Record which contour level should be removed (keep the smaller)
                    val1 = label1['level']
                    val2 = label2['level']
                    # Remove the larger value (which tends to sit higher/right and collide more)
                    if abs(val1) > abs(val2):
                        overlapping_pairs.append(val1)
                    else:
                        overlapping_pairs.append(val2)
        
        return overlapping_levels, overlapping_pairs
    
    # Helper: hide contour labels that lie too close to axes/borders
    def _hide_labels_near_axes(texts, axis, margin_axes: float = 0.05, margin_data_y: float = 5.0):
        """
        Hide contour labels that are too close to axes/borders so that
        numbers do not intersect with axes or frame lines.
        
        Parameters
        ----------
        texts:
            List of `clabel` text objects.
        axis:
            Matplotlib `Axes` object.
        margin_axes:
            Margin in axes‑normalized coordinates \([0,1]\); enlarge for stricter
            border avoidance (default 5%).
        margin_data_y:
            Margin in data coordinates at the bottom in y for x‑axis proximity
            checks (default 5.0).
        
        Returns
        -------
        bool
            True if all labels were hidden, False otherwise.
        """
        if not texts:
            return False
        
        # Get current axis limits
        xlim = axis.get_xlim()
        ylim = axis.get_ylim()
        
        hidden_count = 0
        for txt in texts:
            xdata, ydata = txt.get_position()
            
            # Method 1: axes‑coordinate based border test
            x_disp, y_disp = axis.transData.transform((xdata, ydata))
            x_axes, y_axes = axis.transAxes.inverted().transform((x_disp, y_disp))
            
            # Method 2: data‑coordinate based test near x‑axis (y lower bound)
            near_bottom = ydata < (ylim[0] + margin_data_y)
            near_top = ydata > (ylim[1] - margin_data_y)
            near_left = xdata < (xlim[0] + (xlim[1] - xlim[0]) * margin_axes)
            near_right = xdata > (xlim[1] - (xlim[1] - xlim[0]) * margin_axes)
            
            # Axes‑coordinate test (strict border proximity)
            near_edge_axes = (
                x_axes < margin_axes
                or x_axes > 1.0 - margin_axes
                or y_axes < margin_axes
                or y_axes > 1.0 - margin_axes
            )
            
            # Hide labels that are close to any plot border
            if near_bottom or near_top or near_left or near_right or near_edge_axes:
                txt.set_visible(False)
                hidden_count += 1
        
        # Return True if all labels ended up hidden
        return hidden_count == len(texts)

    # Draw contours using auto‑generated or user‑provided levels (already processed)
    if len(contour_levels_to_use) > 0:
        # Directly use provided contour levels (from `auto_contour_levels` or otherwise)
        filtered_levels = contour_levels_to_use
        
        # Define a smart formatter to avoid unnecessary scientific notation
        def format_contour_label(x):
            """Format contour labels, minimizing use of scientific notation."""
            # For very large (>=1000) or very small (<0.01, >0) values, use sci‑notation
            if abs(x) >= 1000:
                # For large values, prefer integers or one decimal place
                if abs(x) % 1 < 1e-6:  # Nearly integer
                    return f"{int(x)}"
                else:
                    return f"{x:.1f}"
            elif 0 < abs(x) < 0.01:
                # For very small non‑zero values, use scientific notation
                return f"{x:.2e}"
            else:
                # For moderate values, use standard decimal formatting (≤ 2 decimals)
                # Show integers when close enough
                if abs(x) % 1 < 1e-6:
                    return f"{int(x)}"
                elif abs(x) < 1:
                    return f"{x:.2f}"
                elif abs(x) < 10:
                    return f"{x:.1f}"
                else:
                    return f"{x:.0f}"
        
        # Draw filtered contour set: first draw all, then detect and remove overlaps
        if len(filtered_levels) > 0:
            # First draw all contours (including 0‑level, no need to draw twice)
            # Use a thicker line for the 0‑level when present
            all_contour_objects = []  # Store all contour objects
            all_text_objects = []  # Store all text objects
            all_levels_map = {}  # Map labels → contour level values
            
            if use_stroke:
                # Stroke mode: add white halo around black contour lines
                if 0.0 in filtered_levels:
                    # Draw 0-level (thick) and non-zero contours separately
                    non_zero_levels = filtered_levels[filtered_levels != 0.0]
                    if len(non_zero_levels) > 0:
                        for level in non_zero_levels:
                            # Draw main black contour line
                            CS_level = ax.contour(X_log, Y, Z, levels=[level], colors='black', linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH)
                            all_contour_objects.append(CS_level)
                            # Add white stroke/halo to contour collection
                            for collection in CS_level.collections:
                                collection.set_path_effects([
                                    patheffects.withStroke(
                                        linewidth=cfg.CONTOUR_STROKE_LINEWIDTH,
                                        foreground=cfg.CONTOUR_STROKE_FOREGROUND,
                                        alpha=cfg.CONTOUR_STROKE_ALPHA
                                    ),
                                    patheffects.Normal()
                                ])
                            texts_level = ax.clabel(
                                CS_level,
                                inline=True,
                                fontsize=font_size,
                                fmt=format_contour_label,
                                colors='black',
                            )
                            if texts_level:
                                # Add white stroke to each text label
                                for txt in texts_level:
                                    txt.set_path_effects([
                                        patheffects.withStroke(
                                            linewidth=cfg.LABEL_STROKE_LINEWIDTH,
                                            foreground=cfg.LABEL_STROKE_FOREGROUND,
                                            alpha=cfg.LABEL_STROKE_ALPHA
                                        ),
                                        patheffects.Normal()
                                    ])
                                # Check and hide labels near axes/borders
                                all_hidden = _hide_labels_near_axes(texts_level, ax)
                                # If all labels for this level were hidden, remove the contour
                                if all_hidden:
                                    for collection in CS_level.collections:
                                        collection.remove()
                                else:
                                    all_text_objects.extend(texts_level)
                                    for txt in texts_level:
                                        all_levels_map[txt] = level
                    # Draw 0-level contour (thick)
                    CS_zero = ax.contour(X_log, Y, Z, levels=[0.0], colors='black', linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH_ZERO)
                    all_contour_objects.append(CS_zero)
                    # Add white stroke to 0-level contour
                    for collection in CS_zero.collections:
                        collection.set_path_effects([
                            patheffects.withStroke(
                                linewidth=cfg.CONTOUR_STROKE_LINEWIDTH_ZERO,
                                foreground=cfg.CONTOUR_STROKE_FOREGROUND,
                                alpha=cfg.CONTOUR_STROKE_ALPHA
                            ),
                            patheffects.Normal()
                        ])
                    texts_zero = ax.clabel(
                        CS_zero,
                        inline=True,
                        fontsize=font_size,
                        fmt=format_contour_label,
                        colors='black',
                    )
                    if texts_zero:
                        for txt in texts_zero:
                            txt.set_path_effects([
                                patheffects.withStroke(
                                    linewidth=cfg.LABEL_STROKE_LINEWIDTH,
                                    foreground=cfg.LABEL_STROKE_FOREGROUND,
                                    alpha=cfg.LABEL_STROKE_ALPHA
                                ),
                                patheffects.Normal()
                            ])
                        # Check and hide labels near axes/borders
                        all_hidden = _hide_labels_near_axes(texts_zero, ax)
                        # If all labels for this level were hidden, remove the contour
                        if all_hidden:
                            for collection in CS_zero.collections:
                                collection.remove()
                        else:
                            all_text_objects.extend(texts_zero)
                            for txt in texts_zero:
                                all_levels_map[txt] = 0.0
                else:
                    # No 0-level; draw all contours normally
                    for level in filtered_levels:
                        # Draw main black contour line
                        CS_level = ax.contour(X_log, Y, Z, levels=[level], colors='black', linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH)
                        all_contour_objects.append(CS_level)
                        # Add white stroke to contour collection
                        for collection in CS_level.collections:
                            collection.set_path_effects([
                                patheffects.withStroke(
                                    linewidth=cfg.CONTOUR_STROKE_LINEWIDTH,
                                    foreground=cfg.CONTOUR_STROKE_FOREGROUND,
                                    alpha=cfg.CONTOUR_STROKE_ALPHA
                                ),
                                patheffects.Normal()
                            ])
                        texts_level = ax.clabel(
                            CS_level,
                            inline=True,
                            fontsize=font_size,
                            fmt=format_contour_label,
                            colors='black',
                        )
                        if texts_level:
                            for txt in texts_level:
                                txt.set_path_effects([
                                    patheffects.withStroke(
                                        linewidth=cfg.LABEL_STROKE_LINEWIDTH,
                                        foreground=cfg.LABEL_STROKE_FOREGROUND,
                                        alpha=cfg.LABEL_STROKE_ALPHA
                                    ),
                                    patheffects.Normal()
                                ])
                            # Check and hide labels near axes/borders
                            all_hidden = _hide_labels_near_axes(texts_level, ax)
                            # If all labels for this level were hidden, remove the contour
                            if all_hidden:
                                for collection in CS_level.collections:
                                    collection.remove()
                            else:
                                all_text_objects.extend(texts_level)
                                all_levels_map.update({txt: level for txt in texts_level})
            else:
                # Choose contour color by background luminance (black on light, white on dark)
                level_colors_list = get_contour_colors_for_levels(X_log, Y, Z, filtered_levels, cmap_obj, norm)
                level_colors_dict = {level: level_colors_list[i] for i, level in enumerate(filtered_levels)}
                
                if 0.0 in filtered_levels:
                    # Draw 0-level (thick) and non-zero contours separately
                    non_zero_levels = filtered_levels[filtered_levels != 0.0]
                    if len(non_zero_levels) > 0:
                        # Draw each contour with its chosen color
                        for level in non_zero_levels:
                            color = level_colors_dict.get(level, 'black')
                            CS_level = ax.contour(X_log, Y, Z, levels=[level], colors=color, linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH)
                            all_contour_objects.append(CS_level)
                            texts_level = ax.clabel(
                                CS_level,
                                inline=True,
                                fontsize=font_size,
                                fmt=format_contour_label,
                                colors=color,
                            )
                            if texts_level:
                                # Check and hide labels near axes/borders
                                all_hidden = _hide_labels_near_axes(texts_level, ax)
                                # If all labels for this level were hidden, remove the contour
                                if all_hidden:
                                    for collection in CS_level.collections:
                                        collection.remove()
                                else:
                                    all_text_objects.extend(texts_level)
                                    # Build label → contour level mapping
                                    for txt in texts_level:
                                        all_levels_map[txt] = level
                    # Draw 0-level contour (thick)
                    color_zero = level_colors_dict.get(0.0, 'black')
                    CS_zero = ax.contour(X_log, Y, Z, levels=[0.0], colors=color_zero, linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH_ZERO)
                    all_contour_objects.append(CS_zero)
                    texts_zero = ax.clabel(
                        CS_zero,
                        inline=True,
                        fontsize=font_size,
                        fmt=format_contour_label,
                        colors=color_zero,
                    )
                    if texts_zero:
                        all_text_objects.extend(texts_zero)
                        for txt in texts_zero:
                            all_levels_map[txt] = 0.0
                else:
                    # No 0-level; draw all contours normally
                    # Draw each contour with its chosen color
                    for level in filtered_levels:
                        color = level_colors_dict.get(level, 'black')
                        CS_level = ax.contour(X_log, Y, Z, levels=[level], colors=color, linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH)
                        all_contour_objects.append(CS_level)
                        texts_level = ax.clabel(
                            CS_level,
                            inline=True,
                            fontsize=font_size,
                            fmt=format_contour_label,
                            colors=color,
                        )
                        if texts_level:
                            # Check and hide labels near axes/borders
                            all_hidden = _hide_labels_near_axes(texts_level, ax)
                            # If all labels for this level were hidden, remove the contour
                            if all_hidden:
                                for collection in CS_level.collections:
                                    collection.remove()
                            else:
                                all_text_objects.extend(texts_level)
                                all_levels_map.update({txt: level for txt in texts_level})
            
            # Detect overlapping contour labels
            if all_text_objects:
                # Levels to remove (labels overlap with borders)
                overlapping_with_border, overlapping_with_labels = _detect_overlapping_labels(all_text_objects, ax, all_levels_map)
                
                # Merge levels to remove
                levels_to_remove = set(overlapping_with_border) | set(overlapping_with_labels)
                
                # If overlapping levels were found, redraw without them
                # but always keep min and max level contours
                if levels_to_remove:
                    min_level = filtered_levels[0] if len(filtered_levels) > 0 else None
                    max_level = filtered_levels[-1] if len(filtered_levels) > 0 else None
                    
                    # Exclude min and max from levels_to_remove
                    levels_to_remove = set(level for level in levels_to_remove if level != min_level and level != max_level)
                    
                    # Remove overlapping levels from filtered_levels
                    filtered_levels_new = filtered_levels[~np.isin(filtered_levels, list(levels_to_remove))]
                    
                    # Ensure min and max levels are kept
                    if min_level is not None and min_level not in filtered_levels_new:
                        filtered_levels_new = np.concatenate([[min_level], filtered_levels_new])
                    if max_level is not None and max_level not in filtered_levels_new:
                        filtered_levels_new = np.concatenate([filtered_levels_new, [max_level]])
                    filtered_levels_new = np.sort(filtered_levels_new)
                    
                    # Clear current contours and labels
                    # Contour collections must be removed from axes
                    for collection in all_contour_objects:
                        try:
                            collection.remove()
                        except:
                            pass
                    # Text objects must be removed too
                    for txt in all_text_objects:
                        try:
                            txt.remove()
                        except:
                            pass
                    
                    # Redraw only non-overlapping contours
                    if len(filtered_levels_new) > 0:
                        if use_stroke:
                            # Stroke: add white halo to black contours via path_effects
                            if 0.0 in filtered_levels_new:
                                non_zero_levels_new = filtered_levels_new[filtered_levels_new != 0.0]
                                if len(non_zero_levels_new) > 0:
                                    for level in non_zero_levels_new:
                                        CS_level = ax.contour(X_log, Y, Z, levels=[level], colors='black', linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH)
                                        # Add white stroke/halo to contour collection
                                        for collection in CS_level.collections:
                                            collection.set_path_effects([
                                                patheffects.withStroke(
                                                    linewidth=cfg.CONTOUR_STROKE_LINEWIDTH,
                                                    foreground=cfg.CONTOUR_STROKE_FOREGROUND,
                                                    alpha=cfg.CONTOUR_STROKE_ALPHA
                                                ),
                                                patheffects.Normal()
                                            ])
                                        texts_level = ax.clabel(CS_level, inline=True, fontsize=font_size, fmt=format_contour_label, colors='black')
                                        if texts_level:
                                            for txt in texts_level:
                                                txt.set_path_effects([
                                                    patheffects.withStroke(
                                                        linewidth=cfg.LABEL_STROKE_LINEWIDTH,
                                                        foreground=cfg.LABEL_STROKE_FOREGROUND,
                                                        alpha=cfg.LABEL_STROKE_ALPHA
                                                    ),
                                                    patheffects.Normal()
                                                ])
                                            # Check and hide labels near axes/borders
                                            all_hidden = _hide_labels_near_axes(texts_level, ax)
                                            # If all labels for this level were hidden, remove the contour
                                            if all_hidden:
                                                for collection in CS_level.collections:
                                                    collection.remove()
                                CS_zero = ax.contour(X_log, Y, Z, levels=[0.0], colors='black', linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH_ZERO)
                                # Add white stroke to 0-level contour
                                for collection in CS_zero.collections:
                                    collection.set_path_effects([
                                        patheffects.withStroke(
                                            linewidth=cfg.CONTOUR_STROKE_LINEWIDTH_ZERO,
                                            foreground=cfg.CONTOUR_STROKE_FOREGROUND,
                                            alpha=cfg.CONTOUR_STROKE_ALPHA
                                        ),
                                        patheffects.Normal()
                                    ])
                                texts_zero = ax.clabel(CS_zero, inline=True, fontsize=font_size, fmt=format_contour_label, colors='black')
                                if texts_zero:
                                    for txt in texts_zero:
                                        txt.set_path_effects([
                                            patheffects.withStroke(
                                                linewidth=cfg.LABEL_STROKE_LINEWIDTH,
                                                foreground=cfg.LABEL_STROKE_FOREGROUND,
                                                alpha=cfg.LABEL_STROKE_ALPHA
                                            ),
                                            patheffects.Normal()
                                        ])
                                    # Check and hide labels near axes/borders
                                    all_hidden = _hide_labels_near_axes(texts_zero, ax)
                                    # If all labels for this level were hidden, remove the contour
                                    if all_hidden:
                                        for collection in CS_zero.collections:
                                            collection.remove()
                            else:
                                for level in filtered_levels_new:
                                    CS_level = ax.contour(X_log, Y, Z, levels=[level], colors='black', linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH)
                                    # Add white stroke/halo to contour collection
                                    for collection in CS_level.collections:
                                        collection.set_path_effects([
                                            patheffects.withStroke(
                                                linewidth=cfg.CONTOUR_STROKE_LINEWIDTH,
                                                foreground=cfg.CONTOUR_STROKE_FOREGROUND,
                                                alpha=cfg.CONTOUR_STROKE_ALPHA
                                            ),
                                            patheffects.Normal()
                                        ])
                                    texts_level = ax.clabel(CS_level, inline=True, fontsize=font_size, fmt=format_contour_label, colors='black')
                                    if texts_level:
                                        for txt in texts_level:
                                            txt.set_path_effects([
                                                patheffects.withStroke(
                                                    linewidth=cfg.LABEL_STROKE_LINEWIDTH,
                                                    foreground=cfg.LABEL_STROKE_FOREGROUND,
                                                    alpha=cfg.LABEL_STROKE_ALPHA
                                                ),
                                                patheffects.Normal()
                                            ])
                                        # Check and hide labels near axes/borders
                                        all_hidden = _hide_labels_near_axes(texts_level, ax)
                                        # If all labels for this level were hidden, remove the contour
                                        if all_hidden:
                                            for collection in CS_level.collections:
                                                collection.remove()
                        else:
                            # Get color for each contour from background luminance
                            level_colors_list_new = get_contour_colors_for_levels(X_log, Y, Z, filtered_levels_new, cmap_obj, norm)
                            level_colors_dict_new = {level: level_colors_list_new[i] for i, level in enumerate(filtered_levels_new)}
                            
                            if 0.0 in filtered_levels_new:
                                non_zero_levels_new = filtered_levels_new[filtered_levels_new != 0.0]
                                if len(non_zero_levels_new) > 0:
                                    # Draw each contour with its chosen color
                                    for level in non_zero_levels_new:
                                        color = level_colors_dict_new.get(level, 'black')
                                        CS_level = ax.contour(X_log, Y, Z, levels=[level], colors=color, linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH)
                                        texts_level = ax.clabel(CS_level, inline=True, fontsize=font_size, fmt=format_contour_label, colors=color)
                                        if texts_level:
                                            # Check and hide labels near axes/borders
                                            all_hidden = _hide_labels_near_axes(texts_level, ax)
                                            # If all labels for this level were hidden, remove the contour
                                            if all_hidden:
                                                for collection in CS_level.collections:
                                                    collection.remove()
                                # Draw 0-level contour (thick)
                                color_zero = level_colors_dict_new.get(0.0, 'black')
                                CS_zero = ax.contour(X_log, Y, Z, levels=[0.0], colors=color_zero, linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH_ZERO)
                                texts_zero = ax.clabel(CS_zero, inline=True, fontsize=font_size, fmt=format_contour_label, colors=color_zero)
                                if texts_zero:
                                    # Check and hide labels near axes/borders
                                    all_hidden = _hide_labels_near_axes(texts_zero, ax)
                                    # If all labels for this level were hidden, remove the contour
                                    if all_hidden:
                                        for collection in CS_zero.collections:
                                            collection.remove()
                            else:
                                # No 0-level; draw all contours normally
                                # Draw each contour with its chosen color
                                for level in filtered_levels_new:
                                    color = level_colors_dict_new.get(level, 'black')
                                    CS_level = ax.contour(X_log, Y, Z, levels=[level], colors=color, linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH)
                                    texts_level = ax.clabel(CS_level, inline=True, fontsize=font_size, fmt=format_contour_label, colors=color)
                                    if texts_level:
                                        # Check and hide labels near axes/borders
                                        all_hidden = _hide_labels_near_axes(texts_level, ax)
                                        # If all labels for this level were hidden, remove the contour
                                        if all_hidden:
                                            for collection in CS_level.collections:
                                                collection.remove()
                else:
                    # No overlap; hide labels near borders (no need to remove contours here)
                    _hide_labels_near_axes(all_text_objects, ax)
                    
                    # Find contours whose labels are all hidden or have no labels
                    # Build reverse map: level value → list of label texts
                    level_to_texts = {}
                    for txt, level in all_levels_map.items():
                        if level not in level_to_texts:
                            level_to_texts[level] = []
                        level_to_texts[level].append(txt)
                    
                    # Identify levels with all labels hidden or no labels
                    # Always keep min and max level contours even if labels hidden
                    levels_with_no_visible_labels = []
                    min_level = filtered_levels[0] if len(filtered_levels) > 0 else None
                    max_level = filtered_levels[-1] if len(filtered_levels) > 0 else None
                    
                    for level in filtered_levels:
                        # Skip min/max; keep them even if their labels were hidden
                        if level == min_level or level == max_level:
                            continue
                        
                        if level in level_to_texts:
                            # Has labels; check if all of them are hidden
                            texts = level_to_texts[level]
                            has_visible = any(txt.get_visible() for txt in texts)
                            if not has_visible:
                                levels_with_no_visible_labels.append(level)
                        else:
                            # No labels (clabel produced none for this level); remove it
                            levels_with_no_visible_labels.append(level)
                    
                    # If any levels have all labels hidden or no labels, remove them and redraw
                    # but always keep min and max levels
                    if levels_with_no_visible_labels:
                        # Remove levels with no visible labels from filtered_levels
                        filtered_levels_new = filtered_levels[~np.isin(filtered_levels, levels_with_no_visible_labels)]
                        
                        # Ensure min and max levels are kept
                        if min_level is not None and min_level not in filtered_levels_new:
                            filtered_levels_new = np.concatenate([[min_level], filtered_levels_new])
                        if max_level is not None and max_level not in filtered_levels_new:
                            filtered_levels_new = np.concatenate([filtered_levels_new, [max_level]])
                        filtered_levels_new = np.sort(filtered_levels_new)
                        
                        # Clear current contours and labels
                        for collection in all_contour_objects:
                            try:
                                collection.remove()
                            except:
                                pass
                        for txt in all_text_objects:
                            try:
                                txt.remove()
                            except:
                                pass
                        
                        # Redraw only contours that have visible labels
                        if len(filtered_levels_new) > 0:
                            if use_stroke:
                                # Stroke: add white halo to black contours via path_effects
                                if 0.0 in filtered_levels_new:
                                    non_zero_levels_new = filtered_levels_new[filtered_levels_new != 0.0]
                                    if len(non_zero_levels_new) > 0:
                                        for level in non_zero_levels_new:
                                            CS_level = ax.contour(X_log, Y, Z, levels=[level], colors='black', linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH)
                                            # Add white stroke/halo to contour collection
                                            for collection in CS_level.collections:
                                                collection.set_path_effects([
                                                    patheffects.withStroke(
                                                        linewidth=cfg.CONTOUR_STROKE_LINEWIDTH,
                                                        foreground=cfg.CONTOUR_STROKE_FOREGROUND,
                                                        alpha=cfg.CONTOUR_STROKE_ALPHA
                                                    ),
                                                    patheffects.Normal()
                                                ])
                                            texts_level = ax.clabel(CS_level, inline=True, fontsize=font_size, fmt=format_contour_label, colors='black')
                                            if texts_level:
                                                for txt in texts_level:
                                                    txt.set_path_effects([
                                                        patheffects.withStroke(
                                                            linewidth=cfg.LABEL_STROKE_LINEWIDTH,
                                                            foreground=cfg.LABEL_STROKE_FOREGROUND,
                                                            alpha=cfg.LABEL_STROKE_ALPHA
                                                        ),
                                                        patheffects.Normal()
                                                    ])
                                                # Check and hide labels near axes/borders
                                                all_hidden = _hide_labels_near_axes(texts_level, ax)
                                                # If all labels for this level were hidden, remove the contour
                                                if all_hidden:
                                                    for collection in CS_level.collections:
                                                        collection.remove()
                                    CS_zero = ax.contour(X_log, Y, Z, levels=[0.0], colors='black', linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH_ZERO)
                                    # Add white stroke to 0-level contour
                                    for collection in CS_zero.collections:
                                        collection.set_path_effects([
                                            patheffects.withStroke(
                                                linewidth=cfg.CONTOUR_STROKE_LINEWIDTH_ZERO,
                                                foreground=cfg.CONTOUR_STROKE_FOREGROUND,
                                                alpha=cfg.CONTOUR_STROKE_ALPHA
                                            ),
                                            patheffects.Normal()
                                        ])
                                    texts_zero = ax.clabel(CS_zero, inline=True, fontsize=font_size, fmt=format_contour_label, colors='black')
                                    if texts_zero:
                                        for txt in texts_zero:
                                            txt.set_path_effects([
                                                patheffects.withStroke(
                                                    linewidth=cfg.LABEL_STROKE_LINEWIDTH,
                                                    foreground=cfg.LABEL_STROKE_FOREGROUND,
                                                    alpha=cfg.LABEL_STROKE_ALPHA
                                                ),
                                                patheffects.Normal()
                                            ])
                                        # Check and hide labels near axes/borders
                                        all_hidden = _hide_labels_near_axes(texts_zero, ax)
                                        # If all labels for this level were hidden, remove the contour
                                        if all_hidden:
                                            for collection in CS_zero.collections:
                                                collection.remove()
                                else:
                                    for level in filtered_levels_new:
                                        CS_level = ax.contour(X_log, Y, Z, levels=[level], colors='black', linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH)
                                        # Add white stroke/halo to contour collection
                                        for collection in CS_level.collections:
                                            collection.set_path_effects([
                                                patheffects.withStroke(
                                                    linewidth=cfg.CONTOUR_STROKE_LINEWIDTH,
                                                    foreground=cfg.CONTOUR_STROKE_FOREGROUND,
                                                    alpha=cfg.CONTOUR_STROKE_ALPHA
                                                ),
                                                patheffects.Normal()
                                            ])
                                        texts_level = ax.clabel(CS_level, inline=True, fontsize=font_size, fmt=format_contour_label, colors='black')
                                        if texts_level:
                                            for txt in texts_level:
                                                txt.set_path_effects([
                                                    patheffects.withStroke(
                                                        linewidth=cfg.LABEL_STROKE_LINEWIDTH,
                                                        foreground=cfg.LABEL_STROKE_FOREGROUND,
                                                        alpha=cfg.LABEL_STROKE_ALPHA
                                                    ),
                                                    patheffects.Normal()
                                                ])
                                            # Check and hide labels near axes/borders
                                            all_hidden = _hide_labels_near_axes(texts_level, ax)
                                            # If all labels for this level were hidden, remove the contour
                                            if all_hidden:
                                                for collection in CS_level.collections:
                                                    collection.remove()
                            else:
                                # Get color for each contour from background luminance
                                level_colors_list_new = get_contour_colors_for_levels(X_log, Y, Z, filtered_levels_new, cmap_obj, norm)
                                level_colors_dict_new = {level: level_colors_list_new[i] for i, level in enumerate(filtered_levels_new)}
                                
                                if 0.0 in filtered_levels_new:
                                    non_zero_levels_new = filtered_levels_new[filtered_levels_new != 0.0]
                                    if len(non_zero_levels_new) > 0:
                                        # Draw each contour with its chosen color
                                        for level in non_zero_levels_new:
                                            color = level_colors_dict_new.get(level, 'black')
                                            CS_level = ax.contour(X_log, Y, Z, levels=[level], colors=color, linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH)
                                            texts_level = ax.clabel(CS_level, inline=True, fontsize=font_size, fmt=format_contour_label, colors=color)
                                            if texts_level:
                                                # Check and hide labels near axes/borders
                                                all_hidden = _hide_labels_near_axes(texts_level, ax)
                                                # If all labels for this level were hidden, remove the contour
                                                if all_hidden:
                                                    for collection in CS_level.collections:
                                                        collection.remove()
                                    # Draw 0-level contour (thick)
                                    color_zero = level_colors_dict_new.get(0.0, 'black')
                                    CS_zero = ax.contour(X_log, Y, Z, levels=[0.0], colors=color_zero, linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH_ZERO)
                                    texts_zero = ax.clabel(CS_zero, inline=True, fontsize=font_size, fmt=format_contour_label, colors=color_zero)
                                    if texts_zero:
                                        # Check and hide labels near axes/borders
                                        all_hidden = _hide_labels_near_axes(texts_zero, ax)
                                        # If all labels for this level were hidden, remove the contour
                                        if all_hidden:
                                            for collection in CS_zero.collections:
                                                collection.remove()
                                else:
                                    # No 0-level; draw all contours normally
                                    # Draw each contour with its chosen color
                                    for level in filtered_levels_new:
                                        color = level_colors_dict_new.get(level, 'black')
                                        CS_level = ax.contour(X_log, Y, Z, levels=[level], colors=color, linewidths=cfg.HEATMAP_CONTOUR_LINEWIDTH)
                                        texts_level = ax.clabel(CS_level, inline=True, fontsize=font_size, fmt=format_contour_label, colors=color)
                                        if texts_level:
                                            # Check and hide labels near axes/borders
                                            all_hidden = _hide_labels_near_axes(texts_level, ax)
                                            # If all labels for this level were hidden, remove the contour
                                            if all_hidden:
                                                for collection in CS_level.collections:
                                                    collection.remove()

    # Use same display range as reference heatmaps
    xlim_min = np.min(df["R_joint"])*1e9
    xlim_max = np.max(df["R_joint"])*1e9
    ax.set_xlim(np.log10(xlim_min), np.log10(xlim_max))
    ylim_min = np.min(df["Npw"])
    ylim_max = np.max(df["Npw"])
    ax.set_ylim(ylim_min, ylim_max)

    # Remove all tick values and labels (match parasitic heatmap)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Set frame style (match AF heatmaps)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(3)

    # Optional: mark baseline point A (star, white fill + black edge for visibility on dark)
    if baseline_point is not None:
        bNpw, bR = baseline_point
        bx = np.log10(bR * 1e9)
        by = bNpw
        ax.scatter([bx], [by], marker="*", s=180, c="white", edgecolors="black", linewidths=1.5, zorder=10)

    # Tight layout: remove white margins so subplot fills figure (match AF heatmaps)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    ax.set_position([0, 0, 1, 1])

    # Detect which contour levels are actually drawn
    displayed_levels = []
    displayed_levels_with_labels = []
    
    # Collect all contour sets (QuadContourSet or LineCollection)
    for collection in ax.collections:
        try:
            # Check if it is a contour set (QuadContourSet has .levels)
            if hasattr(collection, 'levels'):
                levels = collection.levels
                if levels is not None:
                    if isinstance(levels, (list, np.ndarray)):
                        displayed_levels.extend(levels)
                    else:
                        displayed_levels.append(levels)
            # Or check for _levels attribute
            elif hasattr(collection, '_levels'):
                levels = collection._levels
                if levels is not None:
                    if isinstance(levels, (list, np.ndarray)):
                        displayed_levels.extend(levels)
                    else:
                        displayed_levels.append(levels)
        except Exception:
            pass
    
    # Deduplicate and sort
    displayed_levels = np.unique(displayed_levels)
    displayed_levels = np.sort(displayed_levels)
    
    # Find which levels have visible labels
    # Method 1: search ax.texts (objects created by clabel)
    for txt in ax.texts:
        if txt.get_visible():
            try:
                # Try to parse contour value from label text
                txt_text = txt.get_text().strip()
                # Strip possible formatting characters
                txt_text = txt_text.replace('$', '').replace('\\', '')
                txt_value = float(txt_text)
                # Find closest contour level value
                if len(displayed_levels) > 0:
                    closest_level = min(displayed_levels, key=lambda x: abs(x - txt_value))
                    # Allow larger tolerance (formatting may change the value)
                    tolerance = max(abs(closest_level) * 0.2, 0.01)
                    if abs(closest_level - txt_value) < tolerance:
                        if closest_level not in displayed_levels_with_labels:
                            displayed_levels_with_labels.append(closest_level)
            except (ValueError, TypeError):
                pass
    
    # Method 2: if displayed_levels is empty, infer from contour paths
    if len(displayed_levels) == 0:
        # Check if contour has paths
        for collection in ax.collections:
            try:
                if hasattr(collection, 'get_paths'):
                    paths = collection.get_paths()
                    if len(paths) > 0:
                        # If paths exist, at least some contours were drawn
                        # We cannot get levels from paths; use original levels
                        displayed_levels = contour_levels_to_use.copy()
                        break
            except Exception:
                pass
    
    displayed_levels_with_labels = np.sort(displayed_levels_with_labels)

    # Build filename from value_col (normal, global-ref, or relative-to-min)
    # Relative-to-min kept as SVG for vector stitching; others as PDF.
    suffix_str = f"_{filename_suffix}" if filename_suffix else ""
    if "global" in value_col.lower():
        plot_path = output_dir / f"delta_lcoe_global_heatmap_{scenario}_{label}{suffix_str}.{cfg.PLOT_FORMAT}"
    elif "min" in value_col.lower():
        plot_path = output_dir / f"delta_lcoe_min_heatmap_{scenario}_{label}{suffix_str}.{cfg.PLOT_FORMAT}"
    else:
        plot_path = output_dir / f"delta_lcoe_heatmap_{scenario}_{label}{suffix_str}.{cfg.PLOT_FORMAT}"
    # Same save options as AF heatmaps: bbox_inches='tight', pad_inches=0.0
    plt.savefig(plot_path, dpi=cfg.PLOT_DPI, bbox_inches='tight', pad_inches=0.0)
    plt.close(fig)
    print(f"Heatmap saved to: {plot_path}")


def save_delta_lcoe_heatmap_colorbar(
    output_dir: Path,
    cmap: str,
    norm: mpl.colors.Normalize,
    label: str = r"$\Delta$LCOE ($/\mathrm{MWh}$)",
    data_min: Optional[float] = None,  # Actual data min (for asymmetric ticks)
    data_max: Optional[float] = None,   # Actual data max (for asymmetric ticks)
    is_global_ref: bool = False,  # Whether this is the global-reference colorbar
    filename_suffix: str = ""  # Optional filename suffix for color scheme
):
    """
    Generate and save a shared ΔLCOE colorbar (all subplots share it).
    Layout/font match save_parasitic_heatmap_colorbar.
    Data may be asymmetric; ticks are generated from the actual data range.
    """
    setup_plot_style()

    # Support custom/truncated colormap names from config (e.g. YlGnBu_trunc_0p8)
    cmap_resolved = cfg.resolve_cmap(cmap) if hasattr(cfg, "resolve_cmap") else cmap
    sm = plt.cm.ScalarMappable(cmap=cmap_resolved, norm=norm)
    sm.set_array([])

    fig_cbar = plt.figure(figsize=(0.5, 4))
    ax_cbar = fig_cbar.add_axes([0.1, 0.15, 0.3, 0.7])
    cbar = fig_cbar.colorbar(sm, cax=ax_cbar)

    # Ticks: use a fixed tick list
    vmin = getattr(norm, "vmin", None)
    vmax = getattr(norm, "vmax", None)
    
    # If data_min/data_max are provided, use them for tick range
    if data_min is not None and data_max is not None and np.isfinite(data_min) and np.isfinite(data_max):
        # Use provided data_min/data_max for tick range
        tick_vmin = data_min
        tick_vmax = data_max
    else:
        tick_vmin = vmin
        tick_vmax = vmax
    
    # If label is CNY/kWh, convert ticks accordingly
    is_cny_unit = "CNY" in label or "cny" in label.lower()
    USD_TO_CNY_EXCHANGE_RATE = 7.2  # 1 USD = 7.2 CNY
    
    # If unit is CNY/kWh, use CNY tick list
    if is_cny_unit:
        # Fixed tick list for CNY/kWh
        fixed_ticks = getattr(cfg, 'LCOE_CONTOUR_FIXED_LEVELS_CNY', 
                              np.array([0, 0.001, 0.002, 0.003, 0.004, 0.004, 0.005, 0.01, 0.02, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6]))
    else:
        # Fixed tick list (USD/MWh)
        fixed_ticks = getattr(cfg, 'LCOE_CONTOUR_FIXED_LEVELS', 
                              np.array([-40, -20, -10, -5, -3, -2, -1, -0.5, -0.2, -0.1, -0.05, 
                                       0, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 40, 65, 80, 100, 120, 140, 160, 180, 200]))
    
    # Filter out ticks outside range
    if (tick_vmin is not None) and (tick_vmax is not None) and np.isfinite(tick_vmin) and np.isfinite(tick_vmax):
        ticks = fixed_ticks[(fixed_ticks >= tick_vmin) & (fixed_ticks <= tick_vmax)]
    else:
        ticks = fixed_ticks
        
        # Set tick format by magnitude; avoid scientific notation
        if len(ticks) > 0:
            cbar.set_ticks(ticks)
            # Use integer format for integers, decimal for decimals; no scientific notation
            def format_tick(x, pos):
                # Plain numeric format; handle all ranges without scientific notation
                abs_x = abs(x)
                if abs_x == 0.0:
                    return "0"
                elif abs_x < 0.0001:
                    # Very small values: 5 decimal places
                    return f"{x:.5f}"
                elif abs_x < 0.001:
                    # Values < 0.001: 4 decimal places
                    return f"{x:.4f}"
                elif abs_x < 0.01:
                    # Values < 0.01: 3 decimal places
                    return f"{x:.3f}"
                elif abs_x < 0.1:
                    # Values < 0.1: 2 decimal places
                    return f"{x:.2f}"
                elif abs_x < 1:
                    # Values < 1: 2 decimal places
                    return f"{x:.2f}"
                else:
                    # Values >= 1
                    if abs_x >= 1 and abs(x - int(x)) < 1e-10:
                        # Integer: show as integer
                        return str(int(x))
                    else:
                        # Non-integer: 2 decimal places
                        return f"{x:.2f}"
            
            # Use FuncFormatter so tick strings avoid scientific notation
            formatter = ticker.FuncFormatter(format_tick)
            cbar.ax.yaxis.set_major_formatter(formatter)
            
            # Explicitly disable scientific notation and offset
            # Hide offset text if present
            cbar.ax.yaxis.offsetText.set_visible(False)
            
            # Ensure tick labels do not use scientific notation
            # by setting each label's text directly (after formatter, before save)
            def update_tick_labels():
                """Update all tick labels to avoid scientific notation."""
                for tick in cbar.ax.yaxis.get_major_ticks():
                    tick_value = tick.get_loc()
                    # Set label text with our formatter
                    tick_label = format_tick(tick_value, None)
                    # Set label text
                    tick.label1.set_text(tick_label)
                    # Ensure label is visible
                    tick.label1.set_visible(True)
            
            # Update labels before save (draw_idle to refresh without blocking)
            try:
                fig_cbar.canvas.draw_idle()
                update_tick_labels()
            except:
                # If draw_idle fails, update labels directly
                update_tick_labels()
    
    cbar.ax.tick_params(labelsize=cfg.HEATMAP_FONT_SIZE_COLORBAR)
    cbar.set_label(label, fontsize=cfg.HEATMAP_FONT_SIZE_COLORBAR, labelpad=15)
    
    # Show ticks on one side only (left)
    cbar.ax.yaxis.set_tick_params(labelleft=True, labelright=False)

    # Force-update all tick labels before save (no scientific notation)
    if len(ticks) > 0:
        # Redefine format_tick in scope
        def format_tick_final(x, pos):
            """Format tick label without scientific notation."""
            abs_x = abs(x)
            if abs_x == 0.0:
                return "0"
            elif abs_x < 0.0001:
                return f"{x:.5f}"
            elif abs_x < 0.001:
                return f"{x:.4f}"
            elif abs_x < 0.01:
                return f"{x:.3f}"
            elif abs_x < 0.1:
                return f"{x:.2f}"
            elif abs_x < 1:
                return f"{x:.2f}"
            else:
                if abs_x >= 1 and abs(x - int(x)) < 1e-10:
                    return str(int(x))
                else:
                    return f"{x:.2f}"
        
        # Force one draw so formatter takes effect
        fig_cbar.canvas.draw()
        
        # Update all tick labels again, set text directly
        for tick in cbar.ax.yaxis.get_major_ticks():
            tick_value = tick.get_loc()
            # Generate label text with formatter
            tick_label = format_tick_final(tick_value, None)
            # Set left tick labels directly, overriding auto format
            tick.label1.set_text(tick_label)
            # Ensure left labels are visible
            tick.label1.set_visible(True)
            # Hide right-side labels
            tick.label2.set_visible(False)
            # Disable LaTeX rendering if possible
            try:
                tick.label1.set_usetex(False)
            except:
                pass
        
        # Draw again so changes take effect
        fig_cbar.canvas.draw()
    

    # Build filename suffix from filename_suffix
    suffix_str = f"_{filename_suffix}" if filename_suffix else ""
    
    if is_global_ref or ("Global Ref" in label) or ("global" in label.lower()):
        cbar_path = output_dir / f"delta_lcoe_global_heatmap_colorbar{suffix_str}.{cfg.PLOT_FORMAT}"
    elif "min" in label.lower() or "relative to scenario minimum" in label.lower():
        cbar_path = output_dir / f"delta_lcoe_min_heatmap_colorbar{suffix_str}.{cfg.PLOT_FORMAT}"
    else:
        cbar_path = output_dir / f"delta_lcoe_heatmap_colorbar{suffix_str}.{cfg.PLOT_FORMAT}"
    
    plt.savefig(cbar_path, dpi=cfg.PLOT_DPI, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig_cbar)
