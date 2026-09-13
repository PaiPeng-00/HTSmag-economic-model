"""
9.0_stitch_delta_lcoe_min_svgs.py

将单个场景 / 单个温度制冷剂组合的 delta_lcoe_min SVG 子图，
按网格拼接成一个大的 SVG 图：

- 行（从上到下）：S1, S2, S3
- 列（从左到右）：4.2K He, 10K He, 20K He, 20K H2

依赖：svgutils
    pip install svgutils
"""

import hashlib
import json
import os as _os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
from fusion_tem import device as cfg
from fusion_tem.economic.cost_boundary import COST_BOUNDARY_VERSION
from fusion_tem.economic.price_basis import (
    HTS_PRICE_2025_USD_PER_KAM_BY_SCENARIO,
)
from fusion_tem.publication_svg import finalize_economic_heatmap_grid

FIGURE_OUTPUT_ROOT = Path(
    _os.environ.get("SCAN_FIGURE_OUTPUT_ROOT", cfg.ECONOMIC_FIGURES_DIR)
)
SUBMISSION_CMAP = _os.environ.get("SUBMISSION_FIGURE_CMAP", "YlGnBu_trunc_0p8")
SUBMISSION_CANDIDATE_ROOT = Path(
    _os.environ.get(
        "SUBMISSION_FIGURE_OUTPUT_ROOT",
        FIGURE_OUTPUT_ROOT.parent / f"{FIGURE_OUTPUT_ROOT.name}_submission_candidate",
    )
)

# 投稿数据只保留全厂 LCOE；所有经济图固定从 fullplant 子目录读取。
METRIC_TAG = COST_BOUNDARY_VERSION
cfg.PARASITIC_RATIO_OUTPUT_DIR = str(Path(cfg.PARASITIC_RATIO_OUTPUT_DIR) / METRIC_TAG)
print(f"[9.0] metric = {METRIC_TAG}  ->  PARASITIC_RATIO_OUTPUT_DIR = {cfg.PARASITIC_RATIO_OUTPUT_DIR}")
INKSCAPE_EXE = _os.environ.get("INKSCAPE_EXE", "inkscape")
# PNG 导出分辨率（DPI）；默认 96，本处提高到 300
PNG_EXPORT_DPI = 300

# 从配置文件导入轴刻度配置
# 横轴（R_joint）：对数刻度，范围 1-100 nOhm（从 config.py 和 plot_library.py 的注释中获取）
# 根据 plot_library.py 中的注释：显示范围 1e-9 ~ 100e-9 Ohm，对应 1nOhm ~ 100nOhm，刻度(推荐, nΩ): 1, 10, 100
X_AXIS_TICKS_NOHM = [1, 10, 100]  # nOhm，对应 log10(1)=0, log10(10)=1, log10(100)=2
X_AXIS_MIN_LOG = 0  # log10(1)
X_AXIS_MAX_LOG = 2  # log10(100)
X_AXIS_MIN_NOHM = 1  # nOhm
X_AXIS_MAX_NOHM = 100  # nOhm

# 纵轴（Npw）：线性刻度，范围从 config.py 和 plot_library.py 获取
# 根据 plot_library.py：显示范围基于数据的 min/max，从 NPW_SCAN_VALUES 中选择关键刻度值
# 注意：plot_library.py 中实际使用 ylim_min = np.min(df["Npw"]) 和 ylim_max = np.max(df["Npw"])
# 根据 NPW_SCAN_VALUES = np.concatenate([np.arange(1, 21, 1), np.arange(10, 201, 10)])
# 实际数据范围是 1 ~ 200
NPW_VALUES = np.array(cfg.NPW_SCAN_VALUES)  # 从 config.py 获取
Y_AXIS_MIN = 1  # 实际显示范围的最小值（从 NPW_SCAN_VALUES 获取，实际数据从1开始）
Y_AXIS_MAX = 200  # 实际显示范围的最大值（从 NPW_SCAN_VALUES 获取）
# 从 NPW_SCAN_VALUES 中选择关键刻度值（10, 50, 100, 200）
# 这些值应该与 plot_library.py 中推荐的刻度值一致
# 线性轴上把起点 1 标出来 —— 否则读者容易默认纵轴从 0 起。
# 不能同时保留 10：面板高 360 单位、字号 34，而 1 与 10 仅相距 16.3 单位会重叠；
# 1 与 50 相距 88.6 单位，安全。(log 版 Fig5-1 无此限制，见下方 FIG5_YLOG。)
Y_AXIS_TICKS = [1, 50, 100, 200]

# FIG5_YLOG=1: Npw 纵轴改对数, 输出命名为 Fig5-1(与线性版 Fig5 并存, 不覆盖)。
# 必须与 plotting/library.py 里同名开关一致 —— 那边决定面板怎么画, 这边决定刻度画在哪;
# 只改一边会让刻度与图面错位(无报错, 静默错)。仅作用于 Fig5/Fig6(同一面板生成器),
# Fig1/Fig4/FigS10 的面板仍是线性, 其刻度映射不得走 log 分支。
FIG5_YLOG = _os.environ.get("FIG5_YLOG", "0") == "1"
if FIG5_YLOG:
    Y_AXIS_TICKS = [1, 5, 20, 100, 200]   # log 轴下低 Npw 段被展开, 可给更密的低值刻度
    print(f"[FIG5_YLOG] Npw 纵轴改对数, 刻度 {Y_AXIS_TICKS}, 输出名 Fig5-1")

# iso-充电模式(ISO_CHARGE_H 置位): 8.1 把 Fig4/5/6 的 Npw 网格改为公共窗 [ISO_NPW_MIN, ISO_NPW_MAX]
# (默认 3-20), 纵轴随之改; 与固定-rho(1-200)不同, 此处同步切换刻度, 否则刻度错位。
if _os.environ.get("ISO_CHARGE_H", "").strip():
    Y_AXIS_MIN = int(_os.environ.get("ISO_NPW_MIN", "3"))
    Y_AXIS_MAX = int(_os.environ.get("ISO_NPW_MAX", "20"))
    Y_AXIS_TICKS = [3, 5, 10, 20]
    print(f"[ISO] 拼图纵轴改为 iso 窗口 [{Y_AXIS_MIN},{Y_AXIS_MAX}], 刻度 {Y_AXIS_TICKS}")

# AF 图的配置（从 2.9_af_time999_3&3.py 中获取）
# 注意：这些值需要与 2.9_af_time999_3&3.py 中的 FIXED_NPW 和 FIXED_RHOT 保持一致
# 为了避免循环导入，这里直接使用与 2.9_af_time999_3&3.py 中相同的值
# FIXED_NPW = np.arange(1,21,1)，所以范围是 1-20
# DESIRED_NPW_TICKS = [1, 5, 10, 20]
# 这些刻度是**硬编码**的，不从面板 SVG 反推 —— 必须与 2.9 的 FIXED_NPW/FIXED_RHOT
# 保持一致，否则轴标签会与图面数据对不上。故用同一个 FIG1_FULL_RANGE 开关联动。
# **默认全幅**（与 2.9 一致）；FIG1_FULL_RANGE=0 退回旧窄幅。
FIG1_FULL_RANGE = _os.environ.get("FIG1_FULL_RANGE", "1") == "1"

if FIG1_FULL_RANGE:
    # 全幅：与全网格/Fig5/6 同口径（见 2.9 中同名开关与工作文档 §14.48）
    import math as _math
    _NPW_MAX = int(_os.environ.get("FIG1_NPW_MAX", "40"))
    _RHO_MAX = float(_os.environ.get("FIG1_RHO_MAX", "10000"))
    AF_Y_AXIS_MIN = 1
    AF_Y_AXIS_MAX = _NPW_MAX
    AF_Y_AXIS_TICKS = sorted(set(
        [t for t in (1, 5, 10, 20, 50, 100, 200) if t <= _NPW_MAX] + [_NPW_MAX]))
    AF_X_AXIS_TICKS_UOHM_CM2 = [t for t in (10, 100, 1000, 10000) if t <= _RHO_MAX]
    AF_X_AXIS_MIN_LOG = 1
    AF_X_AXIS_MAX_LOG = _math.log10(_RHO_MAX)
    AF_X_AXIS_MIN_UOHM_CM2 = 10
else:
    # 窄幅（默认，稿件现用）
    AF_Y_AXIS_MIN = 1
    AF_Y_AXIS_MAX = 20
    AF_Y_AXIS_TICKS = [1, 5, 10, 20]  # 从 2.9_af_time999_3&3.py 的 DESIRED_NPW_TICKS 获取

    # FIXED_RHOT 范围：10-1000 μΩ·cm²，DESIRED_RHO_TICKS = [10, 100, 1000]
    AF_X_AXIS_TICKS_UOHM_CM2 = [10, 100, 1000]  # μΩ·cm²，对应 log10(10)=1, log10(100)=2, log10(1000)=3
    AF_X_AXIS_MIN_LOG = 1  # log10(10)
    AF_X_AXIS_MAX_LOG = 3  # log10(1000)
    AF_X_AXIS_MIN_UOHM_CM2 = 10  # μΩ·cm²
AF_X_AXIS_MAX_UOHM_CM2 = 1000  # μΩ·cm²

# 右侧标签区域最小宽度（pt），避免长标签如 Intermittent、dwell 被裁剪
RIGHT_LABEL_WIDTH_MIN = 120

try:
    import svgutils.transform as sg
except ImportError as e:
    raise ImportError(
        "需要安装 svgutils 才能拼接 SVG 图像，请先运行：\n\n"
        "    pip install svgutils\n"
    ) from e


def export_svg_to_pdf(svg_path: Path) -> None:
    pdf_path = svg_path.with_suffix(".pdf")
    try:
        subprocess.run(
            [
                INKSCAPE_EXE,
                str(svg_path),
                "--export-type=pdf",
                f"--export-filename={pdf_path}",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print(f"[OK] 已额外保存 PDF: {pdf_path}")
    except FileNotFoundError:
        print(f"[info] 未找到 inkscape.exe（路径错误），跳过导出 PDF: {svg_path}")
    except subprocess.CalledProcessError as exc:
        print(f"[warning] inkscape 导出 PDF 失败: {exc.stderr.strip()}")


def export_svg_to_png(svg_path: Path) -> None:
    """将 SVG 转换为 PNG 格式"""
    png_path = svg_path.with_suffix(".png")
    try:
        subprocess.run(
            [
                INKSCAPE_EXE,
                str(svg_path),
                "--export-type=png",
                f"--export-dpi={PNG_EXPORT_DPI}",
                f"--export-filename={png_path}",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print(f"[OK] 已额外保存 PNG: {png_path}")
    except FileNotFoundError:
        print(f"[info] 未找到 inkscape.exe（路径错误），跳过导出 PNG: {svg_path}")
    except subprocess.CalledProcessError as exc:
        print(f"[warning] inkscape 导出 PNG 失败: {exc.stderr.strip()}")

def export_svg_to_pdf_svglib(svg_path: Path) -> None:
    pdf_path = svg_path.with_suffix(".pdf")
    try:
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPDF
    except Exception:
        print("[info] 未安装 svglib/reportlab，跳过 SVG→PDF")
        return

    try:
        drawing = svg2rlg(str(svg_path))
        renderPDF.drawToFile(drawing, str(pdf_path))
        print(f"[OK] 已额外保存 PDF: {pdf_path}")
    except Exception as exc:
        print(f"[warning] svglib 导出失败: {svg_path} -> {pdf_path}，原因: {exc}")

def _parse_size(size_str: str) -> float:
    """
    将 SVG 中的 width/height 字符串（可能带单位，如 '432pt', '800px'）解析为 float 数值。
    仅保留数字和小数点。
    """
    if size_str is None:
        return 0.0
    m = re.findall(r"[0-9.]+", str(size_str))
    return float(m[0]) if m else 0.0


def to_title_case(text: str) -> str:
    """
    将文本转换为Sentence Case（句子首字母大写，其余小写）。
    用于确保轴标签的学术严谨性。
    保留单位符号（如Ω、μ、cm²等）和括号内容不变。
    
    参数:
        text: 输入文本
        
    返回:
        Sentence Case格式的文本（只有句子首字母大写）
        
    示例:
        "turn-to-turn resistivity" -> "Turn-to-turn resistivity"
        "Coil-to-coil Joint Resistance (nΩ)" -> "Coil-to-coil joint resistance (nΩ)"
        "Turn-to-turn Resistivity (μΩ·cm²)" -> "Turn-to-turn resistivity (μΩ·cm²)"
        "Number of Parallel Tapes" -> "Number of parallel tapes"
    """
    # 如果文本已经包含单位符号（在括号中），需要特殊处理
    # 匹配模式：文本主体 + 可选的括号内容（包含单位）
    pattern = r'^(.+?)(\s*\([^)]+\))?$'
    match = re.match(pattern, text)
    
    if match:
        main_text = match.group(1).strip()
        unit_part = match.group(2) if match.group(2) else ''
    else:
        main_text = text
        unit_part = ''
    
    # 将整个主文本转换为小写，然后只将第一个字母大写
    if main_text:
        # 先转换为全小写
        main_text_lower = main_text.lower()
        # 只将第一个字母（如果是字母）大写
        if main_text_lower and main_text_lower[0].isalpha():
            main_text_result = main_text_lower[0].upper() + main_text_lower[1:]
        else:
            # 如果第一个字符不是字母，找到第一个字母并大写
            for i, char in enumerate(main_text_lower):
                if char.isalpha():
                    main_text_result = main_text_lower[:i] + char.upper() + main_text_lower[i+1:]
                    break
            else:
                main_text_result = main_text_lower
    else:
        main_text_result = main_text
    
    # 组合结果：主文本 + 单位部分（保持不变）
    return main_text_result + unit_part

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _svg_path_payload(path: Path) -> dict[str, object]:
    """Hash vector path geometry and styling to prove artwork preservation."""
    import xml.etree.ElementTree as ET

    root = ET.parse(path).getroot()
    payload = []
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1] != "path":
            continue
        payload.append(
            "|".join(
                node.get(key, "")
                for key in ("d", "style", "fill", "stroke")
            )
        )
    digest = hashlib.sha256("\n".join(payload).encode("utf-8")).hexdigest()
    return {"path_count": len(payload), "sha256": digest}


# 3x3 版式(Fig. 1 / Fig. 4 / fig. S10)的 160 mm 定稿参数。
# 2026-07-26 之前这三张只能靠 scratchpad 里的一次性脚本链定稿; 现已并入上游,
# 参数由冻结 160 mm 版实测反推, 见工作文档 §14.42。
_GRID3_FINALIZE = {
    # 刻度数必须与上面 AF_*_AXIS_TICKS 一致(定稿函数按 n_x_ticks/n_y_ticks 切片共享标签)。
    "Fig1": {"columns": 3, "bottom_tick_y": 1457.0, "y_tick_baseline": 84.0,
             "label_round": 3,
             "n_x_ticks": len(AF_X_AXIS_TICKS_UOHM_CM2),
             "n_y_ticks": len(AF_Y_AXIS_TICKS)},
    "Fig4": {"columns": 3, "y_tick_baseline": 66.0, "label_round": 3},
    "FigS10_cryo_power_grid": {"columns": 3, "y_title_x": 370.0,
                               "y_tick_baseline": 66.0, "label_round": 3},
    # Fig. 6 用 4 列版式(默认), 只覆盖行标签定位。其行标签 "100 US$ kA-1 m-1" 宽约 270 单位,
    # 从 _GRID_LAYOUTS[4]['row_label_x']=1952 起会伸到 2222, 超出 viewBox 宽 2200 -> 右侧被裁
    # (Fig. 5 的 "S1/S2/S3" 仅 34 单位, 无此问题)。改右对齐锁在 2190: 右边界必在画布内,
    # 左边界 1920 仍在末列面板右缘(1900)之外。加宽 viewBox 不可行 —— 物理宽固定 160 mm,
    # 加宽会把 7 pt 标准字号压到 6.8 pt。
    # n_y_ticks 必须跟着 Y_AXIS_TICKS 走: FIG5_YLOG 时刻度 4 个变 5 个,
    # 共享标签数 33->36, 定稿函数按此切片, 不同步会报 "Expected 33 ... found 36"。
    "Fig5": {"n_y_ticks": len(Y_AXIS_TICKS)},
    "Fig5-1": {"n_y_ticks": len(Y_AXIS_TICKS)},
    "Fig6": {"row_label_x": 2190.0, "row_label_anchor": "end",
             "n_y_ticks": len(Y_AXIS_TICKS)},
}


def _write_submission_candidate(
    source_svg: Path,
    figure_name: str,
    cmap_name: str,
) -> None:
    """Write the 160-mm candidate without touching the frozen package."""
    if cmap_name != SUBMISSION_CMAP:
        return

    import cairosvg
    import fitz

    svg_dir = SUBMISSION_CANDIDATE_ROOT / "svg"
    pdf_dir = SUBMISSION_CANDIDATE_ROOT / "pdf"
    preview_dir = SUBMISSION_CANDIDATE_ROOT / "previews"
    for directory in (svg_dir, pdf_dir, preview_dir):
        directory.mkdir(parents=True, exist_ok=True)

    candidate_svg = svg_dir / f"{figure_name}.svg"
    candidate_pdf = pdf_dir / f"{figure_name}.pdf"
    candidate_preview = preview_dir / f"{figure_name}.png"
    source_artwork = _svg_path_payload(source_svg)
    shutil.copyfile(source_svg, candidate_svg)
    layout = finalize_economic_heatmap_grid(
        candidate_svg, **_GRID3_FINALIZE.get(figure_name, {})
    )
    candidate_artwork = _svg_path_payload(candidate_svg)
    if candidate_artwork != source_artwork:
        raise ValueError(
            f"{figure_name}: vector path geometry/style changed during finalization"
        )
    cairosvg.svg2pdf(url=str(candidate_svg), write_to=str(candidate_pdf))

    document = fitz.open(candidate_pdf)
    if len(document) != 1:
        raise ValueError(f"{figure_name}: expected a single-page PDF")
    page = document[0]
    preview_scale = 2000.0 / page.rect.width
    page.get_pixmap(
        matrix=fitz.Matrix(preview_scale, preview_scale), alpha=False
    ).save(str(candidate_preview))
    pdf_width_mm = page.rect.width / 72.0 * 25.4
    pdf_height_mm = page.rect.height / 72.0 * 25.4
    document.close()

    if abs(pdf_width_mm - 160.0) > 0.02:
        raise ValueError(
            f"{figure_name}: PDF width {pdf_width_mm:.3f} mm is not 160 mm"
        )

    scan_input = _os.environ.get("SCAN_INPUT_CSV", "").strip()
    scan_path = Path(scan_input) if scan_input else None
    record = {
        "figure": figure_name,
        "status": "author-review candidate; frozen package not overwritten",
        "metric": METRIC_TAG,
        "colormap": cmap_name,
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": _sha256(Path(__file__)),
        "finalizer": str(
            Path(finalize_economic_heatmap_grid.__code__.co_filename).resolve()
        ),
        "finalizer_sha256": _sha256(
            Path(finalize_economic_heatmap_grid.__code__.co_filename)
        ),
        "source_svg": str(source_svg.resolve()),
        "source_svg_sha256": _sha256(source_svg),
        "scan_input": (
            str(scan_path.resolve())
            if scan_path and scan_path.is_file()
            else scan_input
        ),
        "scan_input_sha256": (
            _sha256(scan_path) if scan_path and scan_path.is_file() else None
        ),
        "candidate_svg_sha256": _sha256(candidate_svg),
        "candidate_pdf_sha256": _sha256(candidate_pdf),
        "candidate_preview_sha256": _sha256(candidate_preview),
        "pdf_size_mm": [pdf_width_mm, pdf_height_mm],
        "typography_pt": {"standard": 7.0, "contour": 6.0, "panel": 9.0},
        "artwork_path_payload": source_artwork,
        "artwork_preserved": True,
        "layout": layout,
    }
    manifest_path = SUBMISSION_CANDIDATE_ROOT / "figure_manifest.json"
    manifest = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[figure_name] = record
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[submission candidate] {figure_name}: "
        f"{pdf_width_mm:.2f} x {pdf_height_mm:.2f} mm -> {candidate_svg}"
    )


def stitch_delta_lcoe_min_svgs_single_cmap(cmap_suffix: str = ""):
    """
    拼接 delta_lcoe_min_heatmap 子图为一个大的 SVG（单个配色方案）。
    
    参数:
        cmap_suffix: 配色方案名称（如 "cividis" 或 "YlGnBu"），如果为空则使用默认文件名
    """
    # 新的路径结构：outputs/figures/economic/{cmap_name}/parasitic_ratio/USD
    if cmap_suffix:
        scan_base_dir = FIGURE_OUTPUT_ROOT / cmap_suffix / cfg.PARASITIC_RATIO_OUTPUT_DIR
    else:
        # 如果没有指定配色方案，使用旧路径（向后兼容）
        scan_base_dir = FIGURE_OUTPUT_ROOT / cfg.PARASITIC_RATIO_OUTPUT_DIR
    scan_usd_dir = scan_base_dir / cfg.PARASITIC_RATIO_USD_DIR

    # 行：技术情景
    scenarios = ["S1", "S2", "S3"]
    # 列：温度-制冷剂组合（与参数扫描一致）
    temp_coolant_pairs = [
        (4.2, "He"),
        (10.0, "He"),
        (20.0, "He"),
        (20.0, "H2"),
    ]

    n_rows = len(scenarios)
    n_cols = len(temp_coolant_pairs)

    # 存储 (root, row_idx, col_idx)，方便在知道最终面板尺寸后统一布局
    panel_entries = []
    panel_w = None   # 使用所有子图中的最大宽度
    panel_h = None   # 使用所有子图中的最大高度

    for row_idx, scenario in enumerate(scenarios):
        scenario_dir = scan_usd_dir / scenario
        for col_idx, (temp, cool) in enumerate(temp_coolant_pairs):
            label = f"{temp}K_{cool}"
            # 根据配色方案构建文件名（文件名格式：delta_lcoe_min_heatmap_{scenario}_{label}_{cmap_suffix}.svg）
            if cmap_suffix:
                fname = scenario_dir / f"delta_lcoe_min_heatmap_{scenario}_{label}_{cmap_suffix}.svg"
            else:
                fname = scenario_dir / f"delta_lcoe_min_heatmap_{scenario}_{label}.svg"
            if not fname.exists():
                # 如果缺图，就跳过该位置（保留空白）
                print(f"[warning] 未找到子图: {fname}")
                continue

            fig = sg.fromfile(str(fname))
            root = fig.getroot()

            # 记录所有子图中的最大宽高：从 SVGFigure 读取，而不是 group root
            size_w, size_h = fig.get_size()
            w = _parse_size(size_w)
            h = _parse_size(size_h)
            panel_w = w if panel_w is None else max(panel_w, w)
            panel_h = h if panel_h is None else max(panel_h, h)

            panel_entries.append((root, row_idx, col_idx))

    if not panel_entries or panel_w is None or panel_h is None:
        print(f"没有找到任何可拼接的 delta_lcoe_min SVG 子图（配色方案: {cmap_suffix or '默认'}），退出。")
        return None

    # 依据最大面板尺寸设置统一边距和间隔（适当放大，避免裁剪）
    margin_x = panel_w * 0.1
    margin_y = panel_h * 0.1

    # 为标签预留空间（单位：pt）
    label_font_size = 22  # 标签字体大小
    left_margin = 20       # 左边边界与纵轴标签之间的空隙
    
    # 固定间距定义（确保刻度、标签与子图边框之间的距离固定）
    x_axis_subplot_to_tick = 25  # 子图下边框到横轴刻度的固定距离
    x_axis_tick_to_label = 15    # 横轴刻度到横轴标签的固定距离
    y_axis_subplot_to_tick = 15  # 子图左边框到纵轴刻度的固定距离
    y_axis_tick_to_label = 15    # 纵轴刻度到纵轴标签的固定距离
    
    # 计算标签区域高度/宽度
    top_label_height = label_font_size + 8  # 上方温度-制冷剂标签空间
    bottom_label_height = x_axis_subplot_to_tick + label_font_size + x_axis_tick_to_label + label_font_size  # 下方横轴刻度和名称空间
    left_label_width = left_margin + y_axis_subplot_to_tick + label_font_size * 2 + y_axis_tick_to_label + label_font_size  # 左边纵轴刻度和名称空间
    right_label_width = max(label_font_size * 2 + 16, RIGHT_LABEL_WIDTH_MIN)  # 右边情景标签空间

    # 计算子图区域的起始位置（考虑左边标签）
    subplot_start_x = left_label_width
    subplot_start_y = top_label_height

    # 预估整张大图所需尺寸（包含标签区域）
    subplot_area_w = margin_x * (n_cols+1 ) + panel_w * (n_cols)
    subplot_area_h = margin_y * (n_rows+1) + panel_h * (n_rows )
    total_w = subplot_area_w + left_label_width + right_label_width
    total_h = subplot_area_h + top_label_height + bottom_label_height

    # 构造大画布：必须带单位（子图使用 "pt"），否则 svgutils 可能无法正确设置尺寸
    fig_out = sg.SVGFigure(f"{total_w}pt", f"{total_h}pt")
    print(f"total_w: {total_w}pt, total_h: {total_h}pt")

    # 按统一网格布局每个子图（考虑标签偏移）
    placed_roots = []
    for root, row_idx, col_idx in panel_entries:
        x = subplot_start_x + margin_x + col_idx * (panel_w + margin_x)
        y = subplot_start_y + margin_y + row_idx * (panel_h + margin_y)
        root.moveto(x, y)
        placed_roots.append(root)

    fig_out.append(placed_roots)

    # 创建 stitched 文件夹（如果不存在）
    stitched_dir = scan_usd_dir / "stitched"
    stitched_dir.mkdir(parents=True, exist_ok=True)

    # 根据配色方案后缀构建输出文件名
    if cmap_suffix:
        out_path = stitched_dir / f"delta_lcoe_min_heatmap_grid_S1-S3_{cmap_suffix}.svg"
    else:
        out_path = stitched_dir / "delta_lcoe_min_heatmap_grid_S1-S3.svg"
    fig_out.save(str(out_path))

    # 保存后，手动修改 SVG 文件，添加标签和设置尺寸
    import xml.etree.ElementTree as ET
    tree = ET.parse(str(out_path))
    root_svg = tree.getroot()
    
    # 获取 SVG 命名空间（从根元素获取）
    svg_ns = None
    for prefix, uri in root_svg.attrib.items():
        if prefix.startswith('xmlns') and 'svg' in uri.lower():
            svg_ns = uri
            break
    if svg_ns is None:
        # 如果没有找到，使用默认命名空间
        svg_ns = 'http://www.w3.org/2000/svg'
    
    # 注册命名空间
    ET.register_namespace('', svg_ns)
    ns_map = {'svg': svg_ns}
    
    # 设置根 SVG 元素的尺寸
    root_svg.set('width', f"{total_w}pt")
    root_svg.set('height', f"{total_h}pt")
    root_svg.set('viewBox', f"0 0 {total_w} {total_h}")
    
    # 创建标签 group（使用正确的命名空间）
    labels_group = ET.SubElement(root_svg, f'{{{svg_ns}}}g', {'id': 'labels'})
    
    # 添加文本样式（统一字体和大小）
    def create_text(x, y, text, font_size=label_font_size, anchor='middle', baseline='middle', bold=False):
        """创建文本元素（使用正确的命名空间）"""
        text_elem = ET.SubElement(labels_group, f'{{{svg_ns}}}text', {
            'x': str(x),
            'y': str(y),
            'font-family': 'Arial, sans-serif',
            'font-size': str(font_size),
            'text-anchor': anchor,
            'dominant-baseline': baseline,
            'fill': 'black'
        })
        if bold:
            text_elem.set('font-weight', 'bold')
        text_elem.text = text
        return text_elem
    
    def create_temp_coolant_label(x, y, temp, cool, font_size=label_font_size):
        """创建温度-制冷剂标签，支持下标（H2的2作为下标）"""
        # 格式化温度：10.0 和 20.0 显示为整数
        if temp == 10.0 or temp == 20.0:
            temp_str = str(int(temp))
        else:
            temp_str = str(temp)
        
        text_elem = ET.SubElement(labels_group, f'{{{svg_ns}}}text', {
            'x': str(x),
            'y': str(y),
            'font-family': 'Arial, sans-serif',
            'font-size': str(font_size),
            'text-anchor': 'middle',
            'dominant-baseline': 'bottom',
            'fill': 'black'
        })
        
        # 添加温度部分（数字和单位之间加空格）
        tspan1 = ET.SubElement(text_elem, f'{{{svg_ns}}}tspan')
        tspan1.text = f"{temp_str} K "
        
        # 添加制冷剂部分（如果是H2，2作为下标）
        if cool == "H2":
            tspan2 = ET.SubElement(text_elem, f'{{{svg_ns}}}tspan')
            tspan2.text = "H"
            tspan3 = ET.SubElement(text_elem, f'{{{svg_ns}}}tspan', {
                'baseline-shift': 'sub',
                'font-size': str(font_size * 0.7)
            })
            tspan3.text = "2"
        else:
            tspan2 = ET.SubElement(text_elem, f'{{{svg_ns}}}tspan')
            tspan2.text = cool
        
        return text_elem
    
    # 1. 上方：每个子图对应的温度-制冷剂标签
    for col_idx, (temp, cool) in enumerate(temp_coolant_pairs):
        x_center = subplot_start_x + margin_x + col_idx * (panel_w + margin_x) + panel_w / 2
        y_pos = top_label_height - 8
        create_temp_coolant_label(x_center, y_pos, temp, cool, font_size=label_font_size)
    
    # 2. 下方：横轴刻度（从配置文件获取）和横轴名称
    # 横轴刻度位置（基于子图的横轴范围，对数刻度）
    x_ticks = X_AXIS_TICKS_NOHM
    x_tick_log = [np.log10(tick) for tick in x_ticks]  # 计算对数值
    
    # 计算最下一排子图的下边框位置
    bottom_row_subplot_bottom = subplot_start_y + margin_y + (n_rows - 1) * (panel_h + margin_y) + panel_h
    
    for col_idx in range(n_cols):
        subplot_x_start = subplot_start_x + margin_x + col_idx * (panel_w + margin_x)
        subplot_x_width = panel_w
        
        for tick_val, tick_log in zip(x_ticks, x_tick_log):
            # 将 log 坐标转换为子图内的像素位置
            tick_x_in_subplot = (tick_log - X_AXIS_MIN_LOG) / (X_AXIS_MAX_LOG - X_AXIS_MIN_LOG) * subplot_x_width
            tick_x_global = subplot_x_start + tick_x_in_subplot
            
            # 刻度标签位置（使用固定间距：子图下边框 + 固定距离）
            y_tick = bottom_row_subplot_bottom + x_axis_subplot_to_tick
            create_text(tick_x_global, y_tick, str(tick_val), font_size=label_font_size, anchor='middle', baseline='top')
    
    # 横轴名称（居中，使用固定间距：刻度位置 + 固定距离）
    x_axis_label_x = subplot_start_x + subplot_area_w / 2
    x_axis_label_y = bottom_row_subplot_bottom + x_axis_subplot_to_tick + label_font_size + x_axis_tick_to_label
    create_text(x_axis_label_x, x_axis_label_y, to_title_case("Coil-to-coil Joint Resistance (nΩ)"), font_size=label_font_size, anchor='middle', baseline='top', bold=False)
    
    # 3. 左边：纵轴刻度（从配置文件获取）和纵轴名称
    y_ticks = Y_AXIS_TICKS
    # 使用 NPW 扫描值推导“边界坐标”，刻度对齐到每个色块中心
    y_min = Y_AXIS_MIN
    y_max = Y_AXIS_MAX
    y_range = y_max - y_min
    
    # 计算最左一列子图的左边框位置
    left_col_subplot_left = subplot_start_x + margin_x
    
    for row_idx in range(n_rows):
        subplot_y_start = subplot_start_y + margin_y + row_idx * (panel_h + margin_y)
        subplot_y_height = panel_h
        
        for tick_val in y_ticks:
            # 将数据坐标转换为子图内的像素位置（基于“边界坐标”，使刻度与色块中心对齐）
            # 注意：SVG y 轴向下，所以需要从底部开始计算
            # 线性映射：tick_val在[y_min, y_max]范围内，映射到子图的[底部, 顶部]
            if FIG5_YLOG:
                # 对数映射, 与 library.py 的 ax.set_yscale("log") 对应
                t_norm = ((_math.log10(tick_val) - _math.log10(y_min))
                          / (_math.log10(y_max) - _math.log10(y_min))
                          if y_max > y_min > 0 else 0.0)
            else:
                t_norm = (tick_val - y_min) / y_range if y_range > 0 else 0.0
            # 确保t_norm在[0, 1]范围内
            t_norm = max(0.0, min(1.0, t_norm))
            # 计算从子图底部向上的距离（在子图坐标系中）
            tick_y_from_bottom = t_norm * subplot_y_height
            # 转换为全局Y坐标（SVG坐标系：y向下，所以从底部减去距离）
            tick_y_global = subplot_y_start + subplot_y_height - tick_y_from_bottom
            # 确保tick_y_global在子图范围内（顶部边界）
            tick_y_global = max(subplot_y_start, min(subplot_y_start + subplot_y_height, tick_y_global))
            
            # Keep every tick label centred on its unchanged data coordinate.
            baseline_align = 'middle'
            
            # 刻度标签位置（使用固定间距：子图左边框 - 固定距离）
            x_tick = left_col_subplot_left - y_axis_subplot_to_tick
            create_text(x_tick, tick_y_global, str(tick_val), font_size=label_font_size, anchor='end', baseline=baseline_align)
    
    # 纵轴名称（居中，旋转90度，使用固定间距：刻度位置 - 固定距离）
    y_axis_label_x = left_col_subplot_left - y_axis_subplot_to_tick - label_font_size * 2 - y_axis_tick_to_label
    y_axis_label_y = subplot_start_y + subplot_area_h / 2
    y_label_elem = ET.SubElement(labels_group, f'{{{svg_ns}}}text', {
        'x': str(y_axis_label_x),
        'y': str(y_axis_label_y),
        'font-family': 'Arial, sans-serif',
        'font-size': str(label_font_size),
        'text-anchor': 'middle',
        'dominant-baseline': 'middle',
        'fill': 'black',
        'transform': f'rotate(-90 {y_axis_label_x} {y_axis_label_y})'
    })
    y_label_elem.text = to_title_case("Number of Parallel Tapes")
    
    # 4. 右边：每个子图对应的情景标签
    for row_idx, scenario in enumerate(scenarios):
        label_text = scenario
        x_pos = subplot_start_x + subplot_area_w + 16  # 固定间距
        y_center = subplot_start_y + margin_y + row_idx * (panel_h + margin_y) + panel_h / 2
        create_text(x_pos, y_center, label_text, font_size=label_font_size, anchor='start', baseline='middle', bold=False)
    
    # 确保 labels group 在根元素下，并且使用正确的命名空间前缀
    # 将 labels group 移到根元素的最后（确保它在最上层显示）
    root_svg.remove(labels_group)
    root_svg.append(labels_group)
    
    # 保存文件，确保命名空间正确
    tree.write(str(out_path), encoding='utf-8', xml_declaration=True)

    # 最后再导出 PDF 和 PNG（此时 width/height/viewBox 已更新）
    # export_svg_to_pdf(out_path)
    export_svg_to_png(out_path)
    _write_submission_candidate(out_path, "Fig5-1" if FIG5_YLOG else "Fig5", cmap_suffix)

    print(f"拼接完成，大图已保存为: {out_path}")
    print(f"标签已添加：上方温度-制冷剂标签，下方横轴刻度(1,10,100)和名称，左边纵轴刻度(10,50,100,200)和名称，右边情景标签(S1,S2,S3)")
    return out_path


def stitch_delta_lcoe_min_svgs():
    """
    拼接 delta_lcoe_min_heatmap 子图为一个大的 SVG（支持所有配色方案）。
    """
    for cmap_name in cfg.color_schemes:
        print(f"\n开始拼接 delta_lcoe_min 热力图（{cmap_name} 配色方案）...")
        stitch_delta_lcoe_min_svgs_single_cmap(cmap_name)
    
    print("\n所有配色方案的拼接完成！")


def stitch_hts_tape_sensitivity_svgs():
    """
    拼接 HTS 带材影响对比的 delta_lcoe_min_heatmap 子图为一个大的 SVG。
    场景：S4, S5, S6（对应 constant-2025-US$ converted HTS prices）
    支持所有配色方案。
    """
    for cmap_name in cfg.color_schemes:
        print(f"\n开始拼接 HTS 带材敏感性热力图（{cmap_name} 配色方案）...")
        # 新的路径结构：outputs/figures/economic/{cmap_name}/parasitic_ratio/USD
        scan_base_dir = FIGURE_OUTPUT_ROOT / cmap_name / cfg.PARASITIC_RATIO_OUTPUT_DIR
        scan_usd_dir = scan_base_dir / cfg.PARASITIC_RATIO_USD_DIR

        # 行：技术情景（HTS带材价格敏感性分析）
        scenarios = ["S4", "S5", "S6"]
        # 右边标签：对应的HTS带材价格
        hts_price_labels = [
            f"{HTS_PRICE_2025_USD_PER_KAM_BY_SCENARIO[scenario]:.3g} US$ kA⁻¹ m⁻¹"
            for scenario in scenarios
        ]
        # 列：温度-制冷剂组合（与参数扫描一致）
        temp_coolant_pairs = [
            (4.2, "He"),
            (10.0, "He"),
            (20.0, "He"),
            (20.0, "H2"),
        ]

        n_rows = len(scenarios)
        n_cols = len(temp_coolant_pairs)

        # 存储 (root, row_idx, col_idx)，方便在知道最终面板尺寸后统一布局
        panel_entries = []
        panel_w = None   # 使用所有子图中的最大宽度
        panel_h = None   # 使用所有子图中的最大高度

        for row_idx, scenario in enumerate(scenarios):
            scenario_dir = scan_usd_dir / scenario
            for col_idx, (temp, cool) in enumerate(temp_coolant_pairs):
                label = f"{temp}K_{cool}"
                # 文件名格式：delta_lcoe_min_heatmap_{scenario}_{label}_{cmap_name}.svg
                fname = scenario_dir / f"delta_lcoe_min_heatmap_{scenario}_{label}_{cmap_name}.svg"
                if not fname.exists():
                    # 如果缺图，就跳过该位置（保留空白）
                    print(f"[warning] 未找到子图: {fname}")
                    continue

                fig = sg.fromfile(str(fname))
                root = fig.getroot()

                # 记录所有子图中的最大宽高：从 SVGFigure 读取，而不是 group root
                size_w, size_h = fig.get_size()
                w = _parse_size(size_w)
                h = _parse_size(size_h)
                panel_w = w if panel_w is None else max(panel_w, w)
                panel_h = h if panel_h is None else max(panel_h, h)

                panel_entries.append((root, row_idx, col_idx))

        if not panel_entries or panel_w is None or panel_h is None:
            print(f"没有找到任何可拼接的 delta_lcoe_min SVG 子图（配色方案: {cmap_name}），跳过。")
            continue

        # 依据最大面板尺寸设置统一边距和间隔（适当放大，避免裁剪）
        margin_x = panel_w * 0.1
        margin_y = panel_h * 0.1

        # 为标签预留空间（单位：pt）
        label_font_size = 22  # 标签字体大小
        left_margin = 20       # 左边边界与纵轴标签之间的空隙
        
        # 固定间距定义（确保刻度、标签与子图边框之间的距离固定）
        x_axis_subplot_to_tick = 25  # 子图下边框到横轴刻度的固定距离
        x_axis_tick_to_label = 15    # 横轴刻度到横轴标签的固定距离
        y_axis_subplot_to_tick = 15  # 子图左边框到纵轴刻度的固定距离
        y_axis_tick_to_label = 15    # 纵轴刻度到纵轴标签的固定距离
        
        # 计算标签区域高度/宽度
        top_label_height = label_font_size + 8  # 上方温度-制冷剂标签空间
        bottom_label_height = x_axis_subplot_to_tick + label_font_size + x_axis_tick_to_label + label_font_size  # 下方横轴刻度和名称空间
        # 右边标签需要更多空间（因为文本较长）
        right_label_width = max(max(len(label) * label_font_size * 0.5 for label in hts_price_labels) + 20, RIGHT_LABEL_WIDTH_MIN)
        left_label_width = left_margin + y_axis_subplot_to_tick + label_font_size * 2 + y_axis_tick_to_label + label_font_size  # 左边纵轴刻度和名称空间

        # 计算子图区域的起始位置（考虑左边标签）
        subplot_start_x = left_label_width
        subplot_start_y = top_label_height

        # 预估整张大图所需尺寸（包含标签区域）
        subplot_area_w = margin_x * (n_cols+1 ) + panel_w * (n_cols)
        subplot_area_h = margin_y * (n_rows+1) + panel_h * (n_rows )
        total_w = subplot_area_w + left_label_width + right_label_width
        total_h = subplot_area_h + top_label_height + bottom_label_height

        # 构造大画布：必须带单位（子图使用 "pt"），否则 svgutils 可能无法正确设置尺寸
        fig_out = sg.SVGFigure(f"{total_w}pt", f"{total_h}pt")
        print(f"total_w: {total_w}pt, total_h: {total_h}pt")

        # 按统一网格布局每个子图（考虑标签偏移）
        placed_roots = []
        for root, row_idx, col_idx in panel_entries:
            x = subplot_start_x + margin_x + col_idx * (panel_w + margin_x)
            y = subplot_start_y + margin_y + row_idx * (panel_h + margin_y)
            root.moveto(x, y)
            placed_roots.append(root)

        fig_out.append(placed_roots)

        # 创建 stitched 文件夹（如果不存在）
        stitched_dir = scan_usd_dir / "stitched"
        stitched_dir.mkdir(parents=True, exist_ok=True)

        out_path = stitched_dir / f"delta_lcoe_min_heatmap_grid_HTS_tape_sensitivity_{cmap_name}.svg"
        fig_out.save(str(out_path))

        # 保存后，手动修改 SVG 文件，添加标签和设置尺寸
        import xml.etree.ElementTree as ET
        tree = ET.parse(str(out_path))
        root_svg = tree.getroot()
        
        # 获取 SVG 命名空间（从根元素获取）
        svg_ns = None
        for prefix, uri in root_svg.attrib.items():
            if prefix.startswith('xmlns') and 'svg' in uri.lower():
                svg_ns = uri
                break
        if svg_ns is None:
            # 如果没有找到，使用默认命名空间
            svg_ns = 'http://www.w3.org/2000/svg'
        
        # 注册命名空间
        ET.register_namespace('', svg_ns)
        ns_map = {'svg': svg_ns}
        
        # 设置根 SVG 元素的尺寸
        root_svg.set('width', f"{total_w}pt")
        root_svg.set('height', f"{total_h}pt")
        root_svg.set('viewBox', f"0 0 {total_w} {total_h}")
        
        # 创建标签 group（使用正确的命名空间）
        labels_group = ET.SubElement(root_svg, f'{{{svg_ns}}}g', {'id': 'labels'})
        
        # 添加文本样式（统一字体和大小）
        def create_text(x, y, text, font_size=label_font_size, anchor='middle', baseline='middle', bold=False):
            """创建文本元素（使用正确的命名空间）"""
            text_elem = ET.SubElement(labels_group, f'{{{svg_ns}}}text', {
                'x': str(x),
                'y': str(y),
                'font-family': 'Arial, sans-serif',
                'font-size': str(font_size),
                'text-anchor': anchor,
                'dominant-baseline': baseline,
                'fill': 'black'
            })
            if bold:
                text_elem.set('font-weight', 'bold')
            text_elem.text = text
            return text_elem
        
        def create_temp_coolant_label(x, y, temp, cool, font_size=label_font_size):
            """创建温度-制冷剂标签，支持下标（H2的2作为下标）"""
            # 格式化温度：10.0 和 20.0 显示为整数
            if temp == 10.0 or temp == 20.0:
                temp_str = str(int(temp))
            else:
                temp_str = str(temp)
            
            text_elem = ET.SubElement(labels_group, f'{{{svg_ns}}}text', {
                'x': str(x),
                'y': str(y),
                'font-family': 'Arial, sans-serif',
                'font-size': str(font_size),
                'text-anchor': 'middle',
                'dominant-baseline': 'bottom',
                'fill': 'black'
            })
            
            # 添加温度部分（数字和单位之间加空格）
            tspan1 = ET.SubElement(text_elem, f'{{{svg_ns}}}tspan')
            tspan1.text = f"{temp_str} K "
            
            # 添加制冷剂部分（如果是H2，2作为下标）
            if cool == "H2":
                tspan2 = ET.SubElement(text_elem, f'{{{svg_ns}}}tspan')
                tspan2.text = "H"
                tspan3 = ET.SubElement(text_elem, f'{{{svg_ns}}}tspan', {
                    'baseline-shift': 'sub',
                    'font-size': str(font_size * 0.7)
                })
                tspan3.text = "2"
            else:
                tspan2 = ET.SubElement(text_elem, f'{{{svg_ns}}}tspan')
                tspan2.text = cool
            
            return text_elem
        
        # 1. 上方：每个子图对应的温度-制冷剂标签
        for col_idx, (temp, cool) in enumerate(temp_coolant_pairs):
            x_center = subplot_start_x + margin_x + col_idx * (panel_w + margin_x) + panel_w / 2
            y_pos = top_label_height - 8
            create_temp_coolant_label(x_center, y_pos, temp, cool, font_size=label_font_size)
        
        # 2. 下方：横轴刻度（从配置文件获取）和横轴名称
        x_ticks = X_AXIS_TICKS_NOHM
        x_tick_log = [np.log10(tick) for tick in x_ticks]  # 计算对数值
        
        # 计算最下一排子图的下边框位置
        bottom_row_subplot_bottom = subplot_start_y + margin_y + (n_rows - 1) * (panel_h + margin_y) + panel_h
        
        for col_idx in range(n_cols):
            subplot_x_start = subplot_start_x + margin_x + col_idx * (panel_w + margin_x)
            subplot_x_width = panel_w
            
            for tick_val, tick_log in zip(x_ticks, x_tick_log):
                # 将 log 坐标转换为子图内的像素位置
                tick_x_in_subplot = (tick_log - X_AXIS_MIN_LOG) / (X_AXIS_MAX_LOG - X_AXIS_MIN_LOG) * subplot_x_width
                tick_x_global = subplot_x_start + tick_x_in_subplot
                
                # 刻度标签位置（使用固定间距：子图下边框 + 固定距离）
                y_tick = bottom_row_subplot_bottom + x_axis_subplot_to_tick
                create_text(tick_x_global, y_tick, str(tick_val), font_size=label_font_size, anchor='middle', baseline='top')
        
        # 横轴名称（居中，使用固定间距：刻度位置 + 固定距离）
        x_axis_label_x = subplot_start_x + subplot_area_w / 2
        x_axis_label_y = bottom_row_subplot_bottom + x_axis_subplot_to_tick + label_font_size + x_axis_tick_to_label
        create_text(x_axis_label_x, x_axis_label_y, to_title_case("Coil-to-coil Joint Resistance (nΩ)"), font_size=label_font_size, anchor='middle', baseline='top', bold=False)
        
        # 3. 左边：纵轴刻度（从配置文件获取）和纵轴名称
        y_ticks = Y_AXIS_TICKS
        # 使用 NPW 扫描值推导“边界坐标”，刻度对齐到每个色块中心
        y_min = Y_AXIS_MIN
        y_max = Y_AXIS_MAX
        y_range = y_max - y_min
        
        # 计算最左一列子图的左边框位置
        left_col_subplot_left = subplot_start_x + margin_x
        
        for row_idx in range(n_rows):
            subplot_y_start = subplot_start_y + margin_y + row_idx * (panel_h + margin_y)
            subplot_y_height = panel_h
            
            for tick_val in y_ticks:
                # 将数据坐标转换为子图内的像素位置（基于“边界坐标”，使刻度与色块中心对齐）
                # 注意：SVG y 轴向下，所以需要从底部开始计算
                # 将数据坐标转换为子图内的像素位置（使刻度与格子中心对齐）
                # 在matplotlib的pcolormesh中，数据点就是格子的中心位置
                if FIG5_YLOG:
                    # 对数映射, 与 library.py 的 ax.set_yscale("log") 对应
                    t_norm = ((_math.log10(tick_val) - _math.log10(y_min))
                              / (_math.log10(y_max) - _math.log10(y_min))
                              if y_max > y_min > 0 else 0.0)
                else:
                    t_norm = (tick_val - y_min) / y_range if y_range > 0 else 0.0
                # 确保t_norm在[0, 1]范围内
                t_norm = max(0.0, min(1.0, t_norm))
                # 计算从子图底部向上的距离（在子图坐标系中）
                tick_y_from_bottom = t_norm * subplot_y_height
                # 转换为全局Y坐标（SVG坐标系：y向下，所以从底部减去距离）
                tick_y_global = subplot_y_start + subplot_y_height - tick_y_from_bottom
                # 确保tick_y_global在子图范围内
                tick_y_global = max(subplot_y_start, min(subplot_y_start + subplot_y_height, tick_y_global))
                
                # Keep every tick label centred on its unchanged data coordinate.
                baseline_align = 'middle'
                
                # 刻度标签位置（使用固定间距：子图左边框 - 固定距离）
                x_tick = left_col_subplot_left - y_axis_subplot_to_tick
                create_text(x_tick, tick_y_global, str(tick_val), font_size=label_font_size, anchor='end', baseline=baseline_align)
        
        # 纵轴名称（居中，旋转90度，使用固定间距：刻度位置 - 固定距离）
        y_axis_label_x = left_col_subplot_left - y_axis_subplot_to_tick - label_font_size * 2 - y_axis_tick_to_label
        y_axis_label_y = subplot_start_y + subplot_area_h / 2
        y_label_elem = ET.SubElement(labels_group, f'{{{svg_ns}}}text', {
            'x': str(y_axis_label_x),
            'y': str(y_axis_label_y),
            'font-family': 'Arial, sans-serif',
            'font-size': str(label_font_size),
            'text-anchor': 'middle',
            'dominant-baseline': 'middle',
            'fill': 'black',
            'transform': f'rotate(-90 {y_axis_label_x} {y_axis_label_y})'
        })
        y_label_elem.text = to_title_case("Number of Parallel Tapes")
        
        # 4. 右边：每个子图对应的 HTS 带材价格标签。
        # Arial does not render Unicode superscript minus reliably in the
        # Windows SVG/PDF path, so use explicit superscript tspans.
        for row_idx, scenario in enumerate(scenarios):
            x_pos = subplot_start_x + subplot_area_w + 16  # 固定间距
            y_center = subplot_start_y + margin_y + row_idx * (panel_h + margin_y) + panel_h / 2
            label = ET.SubElement(labels_group, f'{{{svg_ns}}}text', {
                'x': str(x_pos),
                'y': str(y_center),
                'font-family': 'Arial, sans-serif',
                'font-size': str(label_font_size),
                'font-weight': 'normal',
                'text-anchor': 'start',
                'dominant-baseline': 'middle',
                'fill': 'black',
            })
            price = HTS_PRICE_2025_USD_PER_KAM_BY_SCENARIO[scenario]
            label.text = f"{price:.3g} US$ kA"
            inverse_ampere = ET.SubElement(label, f'{{{svg_ns}}}tspan', {
                'baseline-shift': 'super',
                'font-size': str(label_font_size * 0.75),
            })
            inverse_ampere.text = '−1'
            inverse_ampere.tail = ' m'
            inverse_metre = ET.SubElement(label, f'{{{svg_ns}}}tspan', {
                'baseline-shift': 'super',
                'font-size': str(label_font_size * 0.75),
            })
            inverse_metre.text = '−1'
        
        # 确保 labels group 在根元素下，并且使用正确的命名空间前缀
        # 将 labels group 移到根元素的最后（确保它在最上层显示）
        root_svg.remove(labels_group)
        root_svg.append(labels_group)
        
        # 保存文件，确保命名空间正确
        tree.write(str(out_path), encoding='utf-8', xml_declaration=True)
        
        # 最后再导出 PDF 和 PNG（此时 width/height/viewBox 已更新）
        # export_svg_to_pdf(out_path)
        export_svg_to_png(out_path)
        _write_submission_candidate(out_path, "Fig6", cmap_name)

        print(f"拼接完成，大图已保存为: {out_path}")
        print("标签已添加：上方温度-制冷剂标签，下方横轴刻度(1,10,100)和名称，"
              "左边纵轴刻度(10,50,100,200)和名称，右边 HTS 带材价格标签"
              "(100,50,10 constant-2025-US$ kA^-1 m^-1)")


def stitch_parasitic_ratio_svgs():
    """
    拼接寄生功耗热力图（parasitic_ratio_heatmap）子图为一个大的 SVG。
    布局：3行3列
    - 行（从上到下）：S1, S2, S3
    - 列（从左到右）：4.2K, 10K, 20K（都是He）
    支持所有配色方案。
    """
    for cmap_name in cfg.color_schemes:
        print(f"\n开始拼接寄生功耗热力图（{cmap_name} 配色方案）...")
        # 新的路径结构：outputs/figures/economic/{cmap_name}/parasitic_ratio/USD
        scan_base_dir = FIGURE_OUTPUT_ROOT / cmap_name / cfg.PARASITIC_RATIO_OUTPUT_DIR
        scan_usd_dir = scan_base_dir / cfg.PARASITIC_RATIO_USD_DIR

        # 行：技术情景
        scenarios = ["S1", "S2", "S3"]
        # 列：温度（都是He，所以不需要制冷剂标签）
        temperatures = [4.2, 10.0, 20.0]

        n_rows = len(scenarios)
        n_cols = len(temperatures)

        # 存储 (root, row_idx, col_idx)，方便在知道最终面板尺寸后统一布局
        panel_entries = []
        panel_w = None   # 使用所有子图中的最大宽度
        panel_h = None   # 使用所有子图中的最大高度

        for row_idx, scenario in enumerate(scenarios):
            scenario_dir = scan_usd_dir / scenario
            for col_idx, temp in enumerate(temperatures):
                # 文件名格式：parasitic_ratio_heatmap_{scenario}_{temp}K_{cmap_name}.svg
                fname = scenario_dir / f"parasitic_ratio_heatmap_{scenario}_{temp}K_{cmap_name}.svg"
                if not fname.exists():
                    # 如果缺图，就跳过该位置（保留空白）
                    print(f"[warning] 未找到子图: {fname}")
                    continue

                fig = sg.fromfile(str(fname))
                root = fig.getroot()

                # 记录所有子图中的最大宽高：从 SVGFigure 读取，而不是 group root
                size_w, size_h = fig.get_size()
                w = _parse_size(size_w)
                h = _parse_size(size_h)
                panel_w = w if panel_w is None else max(panel_w, w)
                panel_h = h if panel_h is None else max(panel_h, h)

                panel_entries.append((root, row_idx, col_idx))

        if not panel_entries or panel_w is None or panel_h is None:
            print(f"没有找到任何可拼接的 parasitic_ratio SVG 子图（配色方案: {cmap_name}），跳过。")
            continue

            # 依据最大面板尺寸设置统一边距和间隔（适当放大，避免裁剪）
        margin_x = panel_w * 0.1
        margin_y = panel_h * 0.1

        # 为标签预留空间（单位：pt）
        label_font_size = 22  # 标签字体大小
        left_margin = 20       # 左边边界与纵轴标签之间的空隙
        
        # 固定间距定义（确保刻度、标签与子图边框之间的距离固定）
        x_axis_subplot_to_tick = 25  # 子图下边框到横轴刻度的固定距离
        x_axis_tick_to_label = 15    # 横轴刻度到横轴标签的固定距离
        y_axis_subplot_to_tick = 15  # 子图左边框到纵轴刻度的固定距离
        y_axis_tick_to_label = 15    # 纵轴刻度到纵轴标签的固定距离
        
        # 计算标签区域高度/宽度
        top_label_height = label_font_size + 8  # 上方温度标签空间
        bottom_label_height = x_axis_subplot_to_tick + label_font_size + x_axis_tick_to_label + label_font_size  # 下方横轴刻度和名称空间
        left_label_width = left_margin + y_axis_subplot_to_tick + label_font_size * 2 + y_axis_tick_to_label + label_font_size  # 左边纵轴刻度和名称空间
        right_label_width = max(label_font_size * 2 + 16, RIGHT_LABEL_WIDTH_MIN)  # 右边情景标签空间

        # 计算子图区域的起始位置（考虑左边标签）
        subplot_start_x = left_label_width
        subplot_start_y = top_label_height

        # 预估整张大图所需尺寸（包含标签区域）
        subplot_area_w = margin_x * (n_cols+1 ) + panel_w * (n_cols)
        subplot_area_h = margin_y * (n_rows+1) + panel_h * (n_rows )
        total_w = subplot_area_w + left_label_width + right_label_width
        total_h = subplot_area_h + top_label_height + bottom_label_height

        # 构造大画布：必须带单位（子图使用 "pt"），否则 svgutils 可能无法正确设置尺寸
        fig_out = sg.SVGFigure(f"{total_w}pt", f"{total_h}pt")
        print(f"total_w: {total_w}pt, total_h: {total_h}pt")

        # 按统一网格布局每个子图（考虑标签偏移）
        placed_roots = []
        for root, row_idx, col_idx in panel_entries:
            x = subplot_start_x + margin_x + col_idx * (panel_w + margin_x)
            y = subplot_start_y + margin_y + row_idx * (panel_h + margin_y)
            root.moveto(x, y)
            placed_roots.append(root)

        fig_out.append(placed_roots)

        # 创建 stitched 文件夹（如果不存在）
        stitched_dir = scan_usd_dir / "stitched"
        stitched_dir.mkdir(parents=True, exist_ok=True)

        out_path = stitched_dir / "parasitic_ratio_heatmap_grid_S1-S3.svg"
        fig_out.save(str(out_path))

        # 保存后，手动修改 SVG 文件，添加标签和设置尺寸
        import xml.etree.ElementTree as ET
        tree = ET.parse(str(out_path))
        root_svg = tree.getroot()
        
        # 获取 SVG 命名空间（从根元素获取）
        svg_ns = None
        for prefix, uri in root_svg.attrib.items():
            if prefix.startswith('xmlns') and 'svg' in uri.lower():
                svg_ns = uri
                break
        if svg_ns is None:
            # 如果没有找到，使用默认命名空间
            svg_ns = 'http://www.w3.org/2000/svg'
        
        # 注册命名空间
        ET.register_namespace('', svg_ns)
        ns_map = {'svg': svg_ns}    
        
        # 设置根 SVG 元素的尺寸
        root_svg.set('width', f"{total_w}pt")
        root_svg.set('height', f"{total_h}pt")
        root_svg.set('viewBox', f"0 0 {total_w} {total_h}")
        
        # 创建标签 group（使用正确的命名空间）
        labels_group = ET.SubElement(root_svg, f'{{{svg_ns}}}g', {'id': 'labels'})
        
        # 添加文本样式（统一字体和大小）
        def create_text(x, y, text, font_size=label_font_size, anchor='middle', baseline='middle', bold=False):
            """创建文本元素（使用正确的命名空间）"""
            text_elem = ET.SubElement(labels_group, f'{{{svg_ns}}}text', {
                'x': str(x),
                'y': str(y),
                'font-family': 'Arial, sans-serif',
                'font-size': str(font_size),
                'text-anchor': anchor,
                'dominant-baseline': baseline,
                'fill': 'black'
            })
            if bold:
                text_elem.set('font-weight', 'bold')
            text_elem.text = text
            return text_elem
        
        def create_temp_label(x, y, temp, font_size=label_font_size):
            """创建温度标签（10.0和20.0显示为整数）"""
            # 格式化温度：10.0 和 20.0 显示为整数
            if temp == 10.0 or temp == 20.0:
                temp_str = str(int(temp))
            else:
                temp_str = str(temp)
            
            label_text = f"{temp_str} K"
            create_text(x, y, label_text, font_size=font_size, anchor='middle', baseline='middle')
        
        # 1. 上方：每个子图对应的温度标签
        for col_idx, temp in enumerate(temperatures):
            x_center = subplot_start_x + margin_x + col_idx * (panel_w + margin_x) + panel_w / 2
            y_pos = top_label_height - 8
            create_temp_label(x_center, y_pos, temp, font_size=label_font_size)
        
        # 2. 下方：横轴刻度（从配置文件获取）和横轴名称
        x_ticks = X_AXIS_TICKS_NOHM
        x_tick_log = [np.log10(tick) for tick in x_ticks]  # 计算对数值
        
        # 计算最下一排子图的下边框位置
        bottom_row_subplot_bottom = subplot_start_y + margin_y + (n_rows - 1) * (panel_h + margin_y) + panel_h
        
        for col_idx in range(n_cols):
            subplot_x_start = subplot_start_x + margin_x + col_idx * (panel_w + margin_x)
            subplot_x_width = panel_w
            
            for tick_val, tick_log in zip(x_ticks, x_tick_log):
                # 将 log 坐标转换为子图内的像素位置
                tick_x_in_subplot = (tick_log - X_AXIS_MIN_LOG) / (X_AXIS_MAX_LOG - X_AXIS_MIN_LOG) * subplot_x_width
                tick_x_global = subplot_x_start + tick_x_in_subplot
                
                # 刻度标签位置（使用固定间距：子图下边框 + 固定距离）
                y_tick = bottom_row_subplot_bottom + x_axis_subplot_to_tick
                create_text(tick_x_global, y_tick, str(tick_val), font_size=label_font_size, anchor='middle', baseline='top')
        
        # 横轴名称（居中，使用固定间距：刻度位置 + 固定距离）
        x_axis_label_x = subplot_start_x + subplot_area_w / 2
        x_axis_label_y = bottom_row_subplot_bottom + x_axis_subplot_to_tick + label_font_size + x_axis_tick_to_label
        create_text(x_axis_label_x, x_axis_label_y, to_title_case("Coil-to-coil Joint Resistance (nΩ)"), font_size=label_font_size, anchor='middle', baseline='top', bold=False)
        
        # 3. 左边：纵轴刻度（从配置文件获取）和纵轴名称
        y_ticks = Y_AXIS_TICKS
        # 使用 NPW 扫描值推导“边界坐标”，刻度对齐到每个色块中心
        y_min = Y_AXIS_MIN
        y_max = Y_AXIS_MAX
        y_range = y_max - y_min
        
        # 计算最左一列子图的左边框位置
        left_col_subplot_left = subplot_start_x + margin_x
        
        for row_idx in range(n_rows):
            subplot_y_start = subplot_start_y + margin_y + row_idx * (panel_h + margin_y)
            subplot_y_height = panel_h
            
            for tick_val in y_ticks:
                # 将数据坐标转换为子图内的像素位置（基于“边界坐标”，使刻度与色块中心对齐）
                # 注意：SVG y 轴向下，所以需要从底部开始计算
                # 将数据坐标转换为子图内的像素位置（使刻度与格子中心对齐）
                # 在matplotlib的pcolormesh中，数据点就是格子的中心位置
                t_norm = (tick_val - y_min) / y_range if y_range > 0 else 0.0
                # 确保t_norm在[0, 1]范围内
                t_norm = max(0.0, min(1.0, t_norm))
                # 计算从子图底部向上的距离（在子图坐标系中）
                tick_y_from_bottom = t_norm * subplot_y_height
                # 转换为全局Y坐标（SVG坐标系：y向下，所以从底部减去距离）
                tick_y_global = subplot_y_start + subplot_y_height - tick_y_from_bottom
                # 确保tick_y_global在子图范围内
                tick_y_global = max(subplot_y_start, min(subplot_y_start + subplot_y_height, tick_y_global))
                
                # 刻度标签位置（使用固定间距：子图左边框 - 固定距离）
                x_tick = left_col_subplot_left - y_axis_subplot_to_tick
                create_text(x_tick, tick_y_global, str(tick_val), font_size=label_font_size, anchor='end', baseline='middle')
        
        # 纵轴名称（居中，旋转90度，使用固定间距：刻度位置 - 固定距离）
        y_axis_label_x = left_col_subplot_left - y_axis_subplot_to_tick - label_font_size * 2 - y_axis_tick_to_label
        y_axis_label_y = subplot_start_y + subplot_area_h / 2
        y_label_elem = ET.SubElement(labels_group, f'{{{svg_ns}}}text', {
            'x': str(y_axis_label_x),
            'y': str(y_axis_label_y),
            'font-family': 'Arial, sans-serif',
            'font-size': str(label_font_size),
            'text-anchor': 'middle',
            'dominant-baseline': 'middle',
            'fill': 'black',
            'transform': f'rotate(-90 {y_axis_label_x} {y_axis_label_y})'
        })
        y_label_elem.text = to_title_case("Number of Parallel Tapes")
        
        # 4. 右边：每个子图对应的情景标签
        for row_idx, scenario in enumerate(scenarios):
            x_pos = subplot_start_x + subplot_area_w + 16  # 固定间距
            y_center = subplot_start_y + margin_y + row_idx * (panel_h + margin_y) + panel_h / 2
            create_text(x_pos, y_center, scenario, font_size=label_font_size, anchor='start', baseline='middle', bold=False)
        
        # 确保 labels group 在根元素下，并且使用正确的命名空间前缀
        # 将 labels group 移到根元素的最后（确保它在最上层显示）
        root_svg.remove(labels_group)
        root_svg.append(labels_group)
        
        # 保存文件，确保命名空间正确
        tree.write(str(out_path), encoding='utf-8', xml_declaration=True)
        
        # 最后再导出 PDF 和 PNG（此时 width/height/viewBox 已更新）
        # export_svg_to_pdf(out_path)
        export_svg_to_png(out_path)

        print(f"拼接完成，大图已保存为: {out_path}")
        # 160 mm 投稿候选(3x3 版式)，见工作文档 §14.42
        _write_submission_candidate(out_path, "Fig4", cmap_name)
        print(f"标签已添加：上方温度标签(4.2K,10K,20K)，下方横轴刻度(1,10,100)和名称，左边纵轴刻度(10,50,100,200)和名称，右边情景标签(S1,S2,S3)")


def stitch_af_heatmap_svgs():
    """
    拼接 AF 热力图（AF_heatmap）子图为一个大的 SVG。
    布局：3行3列
    - 行（从上到下）：S1, S2, S3
    - 列（从左到右）：4.2K, 10K, 20K
    支持所有配色方案。
    """
    for cmap_name in cfg.color_schemes:
        print(f"\n开始拼接 AF 热力图（{cmap_name} 配色方案）...")
        # 新的路径结构：outputs/figures/economic/{cmap_name}/AF_heatmaps_scenarios/figures
        af_base_dir = FIGURE_OUTPUT_ROOT / cmap_name / "AF_heatmaps_scenarios" / "figures"

        # 行：技术情景
        scenarios = ["S1", "S2", "S3"]
        # 列：温度
        temperatures = [4.2, 10.0, 20.0]

        n_rows = len(scenarios)
        n_cols = len(temperatures)

        # 存储 (root, row_idx, col_idx)，方便在知道最终面板尺寸后统一布局
        panel_entries = []
        panel_w = None   # 使用所有子图中的最大宽度
        panel_h = None   # 使用所有子图中的最大高度

        # --- 关键：同一配色方案内保持一致（不混用原版与_truncated） ---
        # 如果该配色方案下任意场景目录存在 *_truncated.svg，则本次拼接优先全部使用_truncated版本。
        prefer_truncated = False
        for _sc in scenarios:
            _dir = af_base_dir / _sc
            if _dir.exists() and any(_dir.glob("AF_heatmap_*_truncated.svg")):
                prefer_truncated = True
                break

        for row_idx, scenario in enumerate(scenarios):
            scenario_dir = af_base_dir / scenario
            if not scenario_dir.exists():
                print(f"[warning] 场景目录不存在: {scenario_dir}")
                continue
            for col_idx, temp in enumerate(temperatures):
                # 确保温度格式一致：使用浮点数格式化，保留一位小数
                temp_str = f"{temp:.1f}" if temp % 1 != 0 else f"{int(temp)}.0"

                # 根据 prefer_truncated 决定优先匹配的文件名
                base_name = f"AF_heatmap_{scenario}_Top_{temp_str}K"
                alt_temp_str = f"{int(temp) if temp % 1 == 0 else temp}"
                alt_base_name = f"AF_heatmap_{scenario}_Top_{alt_temp_str}K"

                candidates: list[Path] = []
                if prefer_truncated:
                    candidates.extend([
                        scenario_dir / f"{base_name}_truncated.svg",
                        scenario_dir / f"{alt_base_name}_truncated.svg",
                    ])
                else:
                    candidates.extend([
                        scenario_dir / f"{base_name}.svg",
                        scenario_dir / f"{alt_base_name}.svg",
                    ])

                fname = None
                for cand in candidates:
                    if cand.exists():
                        fname = cand
                        break

                if fname is None:
                    existing_files = list(scenario_dir.glob(f"AF_heatmap_{scenario}_Top_*K*.svg"))
                    if existing_files:
                        print(f"[warning] 未找到子图: {candidates[0]}")
                        print(f"[info] 目录中存在以下文件: {[f.name for f in existing_files]}")
                    else:
                        print(f"[warning] 未找到子图: {candidates[0]}")
                        print(f"[info] 场景目录 {scenario_dir} 中没有任何AF_heatmap文件")
                    continue

                if prefer_truncated:
                    print(f"[info] 使用截取版文件: {fname.name}")

                fig = sg.fromfile(str(fname))
                root = fig.getroot()

                # 记录所有子图中的最大宽高：从 SVGFigure 读取，而不是 group root
                size_w, size_h = fig.get_size()
                w = _parse_size(size_w)
                h = _parse_size(size_h)
                panel_w = w if panel_w is None else max(panel_w, w)
                panel_h = h if panel_h is None else max(panel_h, h)

                panel_entries.append((root, row_idx, col_idx))

        if not panel_entries or panel_w is None or panel_h is None:
            print(f"没有找到任何可拼接的 AF_heatmap SVG 子图（配色方案: {cmap_name}），跳过。")
            continue

        # 依据最大面板尺寸设置统一边距和间隔（适当放大，避免裁剪）
        margin_x = panel_w * 0.15
        margin_y = panel_h * 0.15

        # 为标签预留空间（单位：pt）
        label_font_size = 22  # 标签字体大小
        left_margin = 20       # 左边边界与纵轴标签之间的空隙
        
        # 固定间距定义（确保刻度、标签与子图边框之间的距离固定）
        x_axis_subplot_to_tick = 25  # 子图下边框到横轴刻度的固定距离
        x_axis_tick_to_label = 15    # 横轴刻度到横轴标签的固定距离
        y_axis_subplot_to_tick = 15  # 子图左边框到纵轴刻度的固定距离
        y_axis_tick_to_label = 15    # 纵轴刻度到纵轴标签的固定距离
        
        # 计算标签区域高度/宽度
        top_label_height = label_font_size + 8  # 上方温度标签空间
        bottom_label_height = x_axis_subplot_to_tick + label_font_size + x_axis_tick_to_label + label_font_size  # 下方横轴刻度和名称空间
        left_label_width = left_margin + y_axis_subplot_to_tick + label_font_size * 2 + y_axis_tick_to_label + label_font_size  # 左边纵轴刻度和名称空间
        right_label_width = max(label_font_size * 2 + 16, RIGHT_LABEL_WIDTH_MIN)  # 右边情景标签空间

        # 计算子图区域的起始位置（考虑左边标签）
        subplot_start_x = left_label_width
        subplot_start_y = top_label_height

        # 预估整张大图所需尺寸（包含标签区域）
        subplot_area_w = margin_x * (n_cols+1 ) + panel_w * (n_cols)
        subplot_area_h = margin_y * (n_rows+1) + panel_h * (n_rows )
        total_w = subplot_area_w + left_label_width + right_label_width
        total_h = subplot_area_h + top_label_height + bottom_label_height

        # 构造大画布：必须带单位（子图使用 "pt"），否则 svgutils 可能无法正确设置尺寸
        fig_out = sg.SVGFigure(f"{total_w}pt", f"{total_h}pt")
        print(f"total_w: {total_w}pt, total_h: {total_h}pt")

        # 按统一网格布局每个子图（考虑标签偏移）
        placed_roots = []
        for root, row_idx, col_idx in panel_entries:
            x = subplot_start_x + margin_x + col_idx * (panel_w + margin_x)
            y = subplot_start_y + margin_y + row_idx * (panel_h + margin_y)
            root.moveto(x, y)
            placed_roots.append(root)

        fig_out.append(placed_roots)

        # 输出命名：与本次选图版本保持一致（不混用）
        output_suffix = "_truncated" if prefer_truncated else ""
        out_path = af_base_dir / f"AF_heatmap_grid_S1-S3{output_suffix}.svg"
        fig_out.save(str(out_path))

        # 保存后，手动修改 SVG 文件，添加标签和设置尺寸
        import xml.etree.ElementTree as ET
        tree = ET.parse(str(out_path))
        root_svg = tree.getroot()
        
        # 获取 SVG 命名空间（从根元素获取）
        svg_ns = None
        for prefix, uri in root_svg.attrib.items():
            if prefix.startswith('xmlns') and 'svg' in uri.lower():
                svg_ns = uri
                break
        if svg_ns is None:
            # 如果没有找到，使用默认命名空间
            svg_ns = 'http://www.w3.org/2000/svg'
        
        # 注册命名空间
        ET.register_namespace('', svg_ns)
        ns_map = {'svg': svg_ns}
        
        # 设置根 SVG 元素的尺寸
        root_svg.set('width', f"{total_w}pt")
        root_svg.set('height', f"{total_h}pt")
        root_svg.set('viewBox', f"0 0 {total_w} {total_h}")
        
        # 创建标签 group（使用正确的命名空间）
        labels_group = ET.SubElement(root_svg, f'{{{svg_ns}}}g', {'id': 'labels'})
        
        # 添加文本样式（统一字体和大小）
        def create_text(x, y, text, font_size=label_font_size, anchor='middle', baseline='middle', bold=False):
            """创建文本元素（使用正确的命名空间）"""
            text_elem = ET.SubElement(labels_group, f'{{{svg_ns}}}text', {
                'x': str(x),
                'y': str(y),
                'font-family': 'Arial, sans-serif',
                'font-size': str(font_size),
                'text-anchor': anchor,
                'dominant-baseline': baseline,
                'fill': 'black'
            })
            if bold:
                text_elem.set('font-weight', 'bold')
            text_elem.text = text
            return text_elem
        
        def create_temp_label(x, y, temp, font_size=label_font_size):
            """创建温度标签（10.0和20.0显示为整数）"""
            # 格式化温度：10.0 和 20.0 显示为整数
            if temp == 10.0 or temp == 20.0:
                temp_str = str(int(temp))
            else:
                temp_str = str(temp)
            
            label_text = f"{temp_str} K"
            create_text(x, y, label_text, font_size=font_size, anchor='middle', baseline='middle')
        
        # 1. 上方：每个子图对应的温度标签
        for col_idx, temp in enumerate(temperatures):
            x_center = subplot_start_x + margin_x + col_idx * (panel_w + margin_x) + panel_w / 2
            y_pos = top_label_height - 8
            create_temp_label(x_center, y_pos, temp, font_size=label_font_size)
        
        # 2. 下方：横轴刻度（从配置文件获取）和横轴名称
        # AF热力图的横轴是对数刻度
        x_ticks = AF_X_AXIS_TICKS_UOHM_CM2
        x_tick_log = [np.log10(tick) for tick in x_ticks]  # 计算对数值
        
        # 计算最下一排子图的下边框位置
        bottom_row_subplot_bottom = subplot_start_y + margin_y + (n_rows - 1) * (panel_h + margin_y) + panel_h
        
        for col_idx in range(n_cols):
            subplot_x_start = subplot_start_x + margin_x + col_idx * (panel_w + margin_x)
            subplot_x_width = panel_w
            
            for tick_val, tick_log in zip(x_ticks, x_tick_log):
                # 将 log 坐标转换为子图内的像素位置
                tick_x_in_subplot = (tick_log - AF_X_AXIS_MIN_LOG) / (AF_X_AXIS_MAX_LOG - AF_X_AXIS_MIN_LOG) * subplot_x_width
                tick_x_global = subplot_x_start + tick_x_in_subplot
                
                # 刻度标签位置（使用固定间距：子图下边框 + 固定距离）
                y_tick = bottom_row_subplot_bottom + x_axis_subplot_to_tick
                create_text(tick_x_global, y_tick, str(tick_val), font_size=label_font_size, anchor='middle', baseline='top')
        
        # 横轴名称（居中，使用固定间距：刻度位置 + 固定距离）
        x_axis_label_x = subplot_start_x + subplot_area_w / 2
        x_axis_label_y = bottom_row_subplot_bottom + x_axis_subplot_to_tick + label_font_size + x_axis_tick_to_label
        create_text(x_axis_label_x, x_axis_label_y, to_title_case("Turn-to-turn Resistivity (μΩ·cm²)"), font_size=label_font_size, anchor='middle', baseline='top', bold=False)
        
        # 3. 左边：纵轴刻度（从配置文件获取）和纵轴名称
        y_ticks = AF_Y_AXIS_TICKS
        y_min = AF_Y_AXIS_MIN
        y_max = AF_Y_AXIS_MAX
        y_range = y_max - y_min
        
        # 计算最左一列子图的左边框位置
        left_col_subplot_left = subplot_start_x + margin_x
        
        for row_idx in range(n_rows):
            subplot_y_start = subplot_start_y + margin_y + row_idx * (panel_h + margin_y)
            subplot_y_height = panel_h
            
            for tick_val in y_ticks:
                # 将数据坐标转换为子图内的像素位置（使刻度与格子中心对齐）
                # 在matplotlib的pcolormesh中，数据点就是格子的中心位置
                # 注意：SVG y 轴向下，所以需要从底部开始计算
                t_norm = (tick_val - y_min) / y_range if y_range > 0 else 0.0
                # 确保t_norm在[0, 1]范围内
                t_norm = max(0.0, min(1.0, t_norm))
                # 计算从子图底部向上的距离（在子图坐标系中）
                tick_y_from_bottom = t_norm * subplot_y_height
                # 转换为全局Y坐标（SVG坐标系：y向下，所以从底部减去距离）
                tick_y_global = subplot_y_start + subplot_y_height - tick_y_from_bottom
                # 确保tick_y_global在子图范围内
                tick_y_global = max(subplot_y_start, min(subplot_y_start + subplot_y_height, tick_y_global))
                
                # 刻度标签位置（使用固定间距：子图左边框 - 固定距离）
                x_tick = left_col_subplot_left - y_axis_subplot_to_tick
                create_text(x_tick, tick_y_global, str(tick_val), font_size=label_font_size, anchor='end', baseline='middle')
        
        # 纵轴名称（居中，旋转90度，使用固定间距：刻度位置 - 固定距离）
        y_axis_label_x = left_col_subplot_left - y_axis_subplot_to_tick - label_font_size * 2 - y_axis_tick_to_label
        y_axis_label_y = subplot_start_y + subplot_area_h / 2
        y_label_elem = ET.SubElement(labels_group, f'{{{svg_ns}}}text', {
            'x': str(y_axis_label_x),
            'y': str(y_axis_label_y),
            'font-family': 'Arial, sans-serif',
            'font-size': str(label_font_size),
            'text-anchor': 'middle',
            'dominant-baseline': 'middle',
            'fill': 'black',
            'transform': f'rotate(-90 {y_axis_label_x} {y_axis_label_y})'
        })
        y_label_elem.text = to_title_case("Number of Parallel Tapes")
        
        # 4. 右边：每个子图对应的情景标签
        for row_idx, scenario in enumerate(scenarios):
            x_pos = subplot_start_x + subplot_area_w + 16  # 固定间距
            y_center = subplot_start_y + margin_y + row_idx * (panel_h + margin_y) + panel_h / 2
            create_text(x_pos, y_center, scenario, font_size=label_font_size, anchor='start', baseline='middle', bold=False)
        
        # 确保 labels group 在根元素下，并且使用正确的命名空间前缀
        # 将 labels group 移到根元素的最后（确保它在最上层显示）
        root_svg.remove(labels_group)
        root_svg.append(labels_group)
        
        # 保存文件，确保命名空间正确
        tree.write(str(out_path), encoding='utf-8', xml_declaration=True)
        
        # 最后再导出 PDF 和 PNG（此时 width/height/viewBox 已更新）
        # export_svg_to_pdf(out_path)
        export_svg_to_png(out_path)

        print(f"拼接完成，大图已保存为: {out_path}")
        print(f"标签已添加：上方温度标签(4.2K,10K,20K)，下方横轴刻度(10,100,1000)和名称，左边纵轴刻度(1,5,10,20)和名称，右边情景标签(S1,S2,S3)")



def stitch_af_ref_heatmap_svgs():
    """
    拼接 AF_ref 热力图（AF_ref_heatmap）子图为一个大的 SVG。
    布局：3行3列
    - 行（从上到下）：S1, S2, S3
    - 列（从左到右）：4.2K, 10K, 20K
    支持所有配色方案。
    """
    for cmap_name in cfg.color_schemes:
        print(f"\n开始拼接 AF_ref 热力图（{cmap_name} 配色方案）...")
        # 新的路径结构：outputs/figures/economic/{cmap_name}/AF_ref_heatmaps_scenarios/figures
        af_base_dir = FIGURE_OUTPUT_ROOT / cmap_name / "AF_ref_heatmaps_scenarios" / "figures"

        # 行：技术情景
        scenarios = ["S1", "S2", "S3"]
        # 列：温度
        temperatures = [4.2, 10.0, 20.0]

        n_rows = len(scenarios)
        n_cols = len(temperatures)

        # 存储 (root, row_idx, col_idx)，方便在知道最终面板尺寸后统一布局
        panel_entries = []
        panel_w = None   # 使用所有子图中的最大宽度
        panel_h = None   # 使用所有子图中的最大高度

        for row_idx, scenario in enumerate(scenarios):
            scenario_dir = af_base_dir / scenario
            for col_idx, temp in enumerate(temperatures):
                fname = scenario_dir / f"AF_ref_heatmap_{scenario}_Top_{temp}K.svg"
                if not fname.exists():
                    # 如果缺图，就跳过该位置（保留空白）
                    print(f"[warning] 未找到子图: {fname}")
                    continue

                fig = sg.fromfile(str(fname))
                root = fig.getroot()

                # 记录所有子图中的最大宽高：从 SVGFigure 读取，而不是 group root
                size_w, size_h = fig.get_size()
                w = _parse_size(size_w)
                h = _parse_size(size_h)
                panel_w = w if panel_w is None else max(panel_w, w)
                panel_h = h if panel_h is None else max(panel_h, h)

                panel_entries.append((root, row_idx, col_idx))

        if not panel_entries or panel_w is None or panel_h is None:
            print(f"没有找到任何可拼接的 AF_ref_heatmap SVG 子图（配色方案: {cmap_name}），跳过。")
            continue

        # 依据最大面板尺寸设置统一边距和间隔（适当放大，避免裁剪）
        margin_x = panel_w * 0.15
        margin_y = panel_h * 0.15

        # 为标签预留空间（单位：pt）
        label_font_size = 22  # 标签字体大小
        left_margin = 20       # 左边边界与纵轴标签之间的空隙
        
        # 固定间距定义（确保刻度、标签与子图边框之间的距离固定）
        x_axis_subplot_to_tick = 25  # 子图下边框到横轴刻度的固定距离
        x_axis_tick_to_label = 15    # 横轴刻度到横轴标签的固定距离
        y_axis_subplot_to_tick = 15  # 子图左边框到纵轴刻度的固定距离
        y_axis_tick_to_label = 15    # 纵轴刻度到纵轴标签的固定距离
        
        # 计算标签区域高度/宽度
        top_label_height = label_font_size + 8  # 上方温度标签空间
        bottom_label_height = x_axis_subplot_to_tick + label_font_size + x_axis_tick_to_label + label_font_size  # 下方横轴刻度和名称空间
        left_label_width = left_margin + y_axis_subplot_to_tick + label_font_size * 2 + y_axis_tick_to_label + label_font_size  # 左边纵轴刻度和名称空间
        right_label_width = max(label_font_size * 2 + 16, RIGHT_LABEL_WIDTH_MIN)  # 右边情景标签空间

        # 计算子图区域的起始位置（考虑左边标签）
        subplot_start_x = left_label_width
        subplot_start_y = top_label_height

        # 预估整张大图所需尺寸（包含标签区域）
        subplot_area_w = margin_x * (n_cols+1 ) + panel_w * (n_cols)
        subplot_area_h = margin_y * (n_rows+1) + panel_h * (n_rows )
        total_w = subplot_area_w + left_label_width + right_label_width
        total_h = subplot_area_h + top_label_height + bottom_label_height

        # 构造大画布：必须带单位（子图使用 "pt"），否则 svgutils 可能无法正确设置尺寸
        fig_out = sg.SVGFigure(f"{total_w}pt", f"{total_h}pt")
        print(f"total_w: {total_w}pt, total_h: {total_h}pt")

        # 按统一网格布局每个子图（考虑标签偏移）
        placed_roots = []
        for root, row_idx, col_idx in panel_entries:
            x = subplot_start_x + margin_x + col_idx * (panel_w + margin_x)
            y = subplot_start_y + margin_y + row_idx * (panel_h + margin_y)
            root.moveto(x, y)
            placed_roots.append(root)

        fig_out.append(placed_roots)

        out_path = af_base_dir / "AF_ref_heatmap_grid_S1-S3.svg"
        fig_out.save(str(out_path))

        # 保存后，手动修改 SVG 文件，添加标签和设置尺寸
        import xml.etree.ElementTree as ET
        tree = ET.parse(str(out_path))
        root_svg = tree.getroot()
        
        # 获取 SVG 命名空间（从根元素获取）
        svg_ns = None
        for prefix, uri in root_svg.attrib.items():
            if prefix.startswith('xmlns') and 'svg' in uri.lower():
                svg_ns = uri
                break
        if svg_ns is None:
            # 如果没有找到，使用默认命名空间
            svg_ns = 'http://www.w3.org/2000/svg'
        
        # 注册命名空间
        ET.register_namespace('', svg_ns)
        ns_map = {'svg': svg_ns}
        
        # 设置根 SVG 元素的尺寸
        root_svg.set('width', f"{total_w}pt")
        root_svg.set('height', f"{total_h}pt")
        root_svg.set('viewBox', f"0 0 {total_w} {total_h}")
        
        # 创建标签 group（使用正确的命名空间）
        labels_group = ET.SubElement(root_svg, f'{{{svg_ns}}}g', {'id': 'labels'})
        
        # 添加文本样式（统一字体和大小）
        def create_text(x, y, text, font_size=label_font_size, anchor='middle', baseline='middle', bold=False):
            """创建文本元素（使用正确的命名空间）"""
            text_elem = ET.SubElement(labels_group, f'{{{svg_ns}}}text', {
                'x': str(x),
                'y': str(y),
                'font-family': 'Arial, sans-serif',
                'font-size': str(font_size),
                'text-anchor': anchor,
                'dominant-baseline': baseline,
                'fill': 'black'
            })
            if bold:
                text_elem.set('font-weight', 'bold')
            text_elem.text = text
            return text_elem
        
        def create_temp_label(x, y, temp, font_size=label_font_size):
            """创建温度标签（10.0和20.0显示为整数）"""
            # 格式化温度：10.0 和 20.0 显示为整数
            if temp == 10.0 or temp == 20.0:
                temp_str = str(int(temp))
            else:
                temp_str = str(temp)
            
            label_text = f"{temp_str} K"
            create_text(x, y, label_text, font_size=font_size, anchor='middle', baseline='middle')
        
        # 1. 上方：每个子图对应的温度标签
        for col_idx, temp in enumerate(temperatures):
            x_center = subplot_start_x + margin_x + col_idx * (panel_w + margin_x) + panel_w / 2
            y_pos = top_label_height - 8
            create_temp_label(x_center, y_pos, temp, font_size=label_font_size)
        
        # 2. 下方：横轴刻度（从配置文件获取）和横轴名称
        # AF_ref热力图的横轴是对数刻度
        x_ticks = AF_X_AXIS_TICKS_UOHM_CM2
        x_tick_log = [np.log10(tick) for tick in x_ticks]  # 计算对数值
        
        # 计算最下一排子图的下边框位置
        bottom_row_subplot_bottom = subplot_start_y + margin_y + (n_rows - 1) * (panel_h + margin_y) + panel_h
        
        for col_idx in range(n_cols):
            subplot_x_start = subplot_start_x + margin_x + col_idx * (panel_w + margin_x)
            subplot_x_width = panel_w
            
            for tick_val, tick_log in zip(x_ticks, x_tick_log):
                # 将 log 坐标转换为子图内的像素位置
                tick_x_in_subplot = (tick_log - AF_X_AXIS_MIN_LOG) / (AF_X_AXIS_MAX_LOG - AF_X_AXIS_MIN_LOG) * subplot_x_width
                tick_x_global = subplot_x_start + tick_x_in_subplot
                
                # 刻度标签位置（使用固定间距：子图下边框 + 固定距离）
                y_tick = bottom_row_subplot_bottom + x_axis_subplot_to_tick
                create_text(tick_x_global, y_tick, str(tick_val), font_size=label_font_size, anchor='middle', baseline='top')
        
        # 横轴名称（居中，使用固定间距：刻度位置 + 固定距离）
        x_axis_label_x = subplot_start_x + subplot_area_w / 2
        x_axis_label_y = bottom_row_subplot_bottom + x_axis_subplot_to_tick + label_font_size + x_axis_tick_to_label
        create_text(x_axis_label_x, x_axis_label_y, to_title_case("Turn-to-turn Resistivity (μΩ·cm²)"), font_size=label_font_size, anchor='middle', baseline='top', bold=False)
        
        # 3. 左边：纵轴刻度（从配置文件获取）和纵轴名称
        y_ticks = AF_Y_AXIS_TICKS
        y_min = AF_Y_AXIS_MIN
        y_max = AF_Y_AXIS_MAX
        y_range = y_max - y_min
        
        # 计算最左一列子图的左边框位置
        left_col_subplot_left = subplot_start_x + margin_x
        
        for row_idx in range(n_rows):
            subplot_y_start = subplot_start_y + margin_y + row_idx * (panel_h + margin_y)
            subplot_y_height = panel_h
            
            for tick_val in y_ticks:
                # 将数据坐标转换为子图内的像素位置（使刻度与格子中心对齐）
                # 在matplotlib的pcolormesh中，数据点就是格子的中心位置
                # 注意：SVG y 轴向下，所以需要从底部开始计算
                t_norm = (tick_val - y_min) / y_range if y_range > 0 else 0.0
                # 确保t_norm在[0, 1]范围内
                t_norm = max(0.0, min(1.0, t_norm))
                # 计算从子图底部向上的距离（在子图坐标系中）
                tick_y_from_bottom = t_norm * subplot_y_height
                # 转换为全局Y坐标（SVG坐标系：y向下，所以从底部减去距离）
                tick_y_global = subplot_y_start + subplot_y_height - tick_y_from_bottom
                # 确保tick_y_global在子图范围内
                tick_y_global = max(subplot_y_start, min(subplot_y_start + subplot_y_height, tick_y_global))
                
                # 刻度标签位置（使用固定间距：子图左边框 - 固定距离）
                x_tick = left_col_subplot_left - y_axis_subplot_to_tick
                create_text(x_tick, tick_y_global, str(tick_val), font_size=label_font_size, anchor='end', baseline='middle')
        
        # 纵轴名称（居中，旋转90度，使用固定间距：刻度位置 - 固定距离）
        y_axis_label_x = left_col_subplot_left - y_axis_subplot_to_tick - label_font_size * 2 - y_axis_tick_to_label
        y_axis_label_y = subplot_start_y + subplot_area_h / 2
        y_label_elem = ET.SubElement(labels_group, f'{{{svg_ns}}}text', {
            'x': str(y_axis_label_x),
            'y': str(y_axis_label_y),
            'font-family': 'Arial, sans-serif',
            'font-size': str(label_font_size),
            'text-anchor': 'middle',
            'dominant-baseline': 'middle',
            'fill': 'black',
            'transform': f'rotate(-90 {y_axis_label_x} {y_axis_label_y})'
        })
        y_label_elem.text = to_title_case("Number of Parallel Tapes")
        
        # 4. 右边：每个子图对应的情景标签
        for row_idx, scenario in enumerate(scenarios):
            x_pos = subplot_start_x + subplot_area_w + 16  # 固定间距
            y_center = subplot_start_y + margin_y + row_idx * (panel_h + margin_y) + panel_h / 2
            create_text(x_pos, y_center, scenario, font_size=label_font_size, anchor='start', baseline='middle', bold=False)
        
        # 确保 labels group 在根元素下，并且使用正确的命名空间前缀
        # 将 labels group 移到根元素的最后（确保它在最上层显示）
        root_svg.remove(labels_group)
        root_svg.append(labels_group)
        
        # 保存文件，确保命名空间正确
        tree.write(str(out_path), encoding='utf-8', xml_declaration=True)
        
        # 最后再导出 PDF 和 PNG（此时 width/height/viewBox 已更新）
        #export_svg_to_pdf(out_path)
        export_svg_to_png(out_path)

        print(f"拼接完成，大图已保存为: {out_path}")
        # 160 mm 投稿候选(3x3 版式)，见工作文档 §14.42
        _write_submission_candidate(out_path, "Fig1", cmap_name)
        print(f"标签已添加：上方温度标签(4.2K,10K,20K)，下方横轴刻度(10,100,1000)和名称，左边纵轴刻度(1,5,10,20)和名称，右边情景标签(S1,S2,S3)")


def stitch_cryo_power_MWe_svgs():
    """
    拼接制冷功率 MWe 热力图子图为 3×3 九宫格。
    按配色方案分别拼接：子图来自 outputs/figures/economic/{cmap}/parasitic_ratio/cryo_power_MWe。
    布局：3 行 × 3 列
    - 行（从上到下）：Pulse（脉冲）, dwell（间歇）, Static（静态）
    - 列（从左到右）：4.2 K, 10 K, 20 K
    横轴：接头电阻 (nΩ)，纵轴：并绕根数，单位 MWe。
    """
    modes = [("pulse", "Pulse"), ("dwell", "Dwell"), ("static", "Static")]
    temperatures = [4.2, 10.0, 20.0]
    n_rows = len(modes)
    n_cols = len(temperatures)

    for cmap_name in cfg.color_schemes:
        cryo_dir = FIGURE_OUTPUT_ROOT / cmap_name / cfg.PARASITIC_RATIO_OUTPUT_DIR / "cryo_power_MWe"
        panel_entries = []
        panel_w = None
        panel_h = None

        for row_idx, (mode_key, _mode_label) in enumerate(modes):
            for col_idx, temp in enumerate(temperatures):
                if temp == 10.0 or temp == 20.0:
                    label = f"{int(temp)}K"
                else:
                    label = f"{temp}K"
                fname = cryo_dir / f"cryo_power_MWe_heatmap_{mode_key}_{label}.svg"
                if not fname.exists():
                    continue
                fig = sg.fromfile(str(fname))
                root = fig.getroot()
                size_w, size_h = fig.get_size()
                w = _parse_size(size_w)
                h = _parse_size(size_h)
                panel_w = w if panel_w is None else max(panel_w, w)
                panel_h = h if panel_h is None else max(panel_h, h)
                panel_entries.append((root, row_idx, col_idx))

        if not panel_entries or panel_w is None or panel_h is None:
            print(f"[warning] 配色 {cmap_name} 下未找到足够的制冷功率 MWe 子图，跳过拼接。")
            continue

        margin_x = panel_w * 0.1
        margin_y = panel_h * 0.1
        label_font_size = 22
        left_margin = 20
        x_axis_subplot_to_tick = 25
        x_axis_tick_to_label = 15
        y_axis_subplot_to_tick = 15
        y_axis_tick_to_label = 15
        top_label_height = label_font_size + 8
        bottom_label_height = x_axis_subplot_to_tick + label_font_size + x_axis_tick_to_label + label_font_size
        left_label_width = left_margin + y_axis_subplot_to_tick + label_font_size * 2 + y_axis_tick_to_label + label_font_size
        right_label_width = max(label_font_size * 2 + 16, RIGHT_LABEL_WIDTH_MIN)
        subplot_start_x = left_label_width
        subplot_start_y = top_label_height
        subplot_area_w = margin_x * (n_cols + 1) + panel_w * n_cols
        subplot_area_h = margin_y * (n_rows + 1) + panel_h * n_rows
        total_w = subplot_area_w + left_label_width + right_label_width
        total_h = subplot_area_h + top_label_height + bottom_label_height

        fig_out = sg.SVGFigure(f"{total_w}pt", f"{total_h}pt")
        placed_roots = []
        for root, row_idx, col_idx in panel_entries:
            x = subplot_start_x + margin_x + col_idx * (panel_w + margin_x)
            y = subplot_start_y + margin_y + row_idx * (panel_h + margin_y)
            root.moveto(x, y)
            placed_roots.append(root)
        fig_out.append(placed_roots)

        stitched_dir = cryo_dir / "stitched"
        stitched_dir.mkdir(parents=True, exist_ok=True)
        out_path = stitched_dir / "cryo_power_MWe_heatmap_grid_3x3.svg"
        fig_out.save(str(out_path))

        import xml.etree.ElementTree as ET
        tree = ET.parse(str(out_path))
        root_svg = tree.getroot()
        svg_ns = None
        for prefix, uri in root_svg.attrib.items():
            if prefix.startswith("xmlns") and "svg" in uri.lower():
                svg_ns = uri
                break
        if svg_ns is None:
            svg_ns = "http://www.w3.org/2000/svg"
        ET.register_namespace("", svg_ns)

        root_svg.set("width", f"{total_w}pt")
        root_svg.set("height", f"{total_h}pt")
        root_svg.set("viewBox", f"0 0 {total_w} {total_h}")
        labels_group = ET.SubElement(root_svg, f"{{{svg_ns}}}g", {"id": "labels"})

        def create_text(x, y, text, font_size=label_font_size, anchor="middle", baseline="middle", bold=False):
            text_elem = ET.SubElement(labels_group, f"{{{svg_ns}}}text", {
                "x": str(x), "y": str(y),
                "font-family": "Arial, sans-serif", "font-size": str(font_size),
                "text-anchor": anchor, "dominant-baseline": baseline, "fill": "black"
            })
            if bold:
                text_elem.set("font-weight", "bold")
            text_elem.text = text
            return text_elem

        # 上方：温度标签
        for col_idx, temp in enumerate(temperatures):
            x_center = subplot_start_x + margin_x + col_idx * (panel_w + margin_x) + panel_w / 2
            y_pos = top_label_height - 8
            temp_str = str(int(temp)) if temp in (10.0, 20.0) else str(temp)
            create_text(x_center, y_pos, f"{temp_str} K", font_size=label_font_size)
        # 下方：横轴刻度（对数 1, 10, 100 nΩ）和横轴名称
        x_ticks = X_AXIS_TICKS_NOHM
        x_tick_log = [np.log10(t) for t in x_ticks]
        bottom_row_subplot_bottom = subplot_start_y + margin_y + (n_rows - 1) * (panel_h + margin_y) + panel_h
        for col_idx in range(n_cols):
            subplot_x_start = subplot_start_x + margin_x + col_idx * (panel_w + margin_x)
            subplot_x_width = panel_w
            for tick_val, tick_log in zip(x_ticks, x_tick_log):
                t_norm = (tick_log - X_AXIS_MIN_LOG) / (X_AXIS_MAX_LOG - X_AXIS_MIN_LOG) if (X_AXIS_MAX_LOG - X_AXIS_MIN_LOG) != 0 else 0
                t_norm = max(0.0, min(1.0, t_norm))
                tick_x_global = subplot_x_start + t_norm * subplot_x_width
                y_tick = bottom_row_subplot_bottom + x_axis_subplot_to_tick
                create_text(tick_x_global, y_tick, str(tick_val), font_size=label_font_size, anchor="middle", baseline="top")
        x_axis_label_x = subplot_start_x + subplot_area_w / 2
        x_axis_label_y = bottom_row_subplot_bottom + x_axis_subplot_to_tick + label_font_size + x_axis_tick_to_label
        create_text(x_axis_label_x, x_axis_label_y, to_title_case("Coil-to-coil joint resistance (nΩ)"), font_size=label_font_size, anchor="middle", baseline="top")
        # 左边：纵轴刻度和名称
        y_ticks = Y_AXIS_TICKS
        y_min, y_max = Y_AXIS_MIN, Y_AXIS_MAX
        y_range = y_max - y_min
        left_col_subplot_left = subplot_start_x + margin_x
        for row_idx in range(n_rows):
            subplot_y_start = subplot_start_y + margin_y + row_idx * (panel_h + margin_y)
            subplot_y_height = panel_h
            for tick_val in y_ticks:
                t_norm = (tick_val - y_min) / y_range if y_range > 0 else 0.0
                t_norm = max(0.0, min(1.0, t_norm))
                tick_y_from_bottom = t_norm * subplot_y_height
                tick_y_global = subplot_y_start + subplot_y_height - tick_y_from_bottom
                tick_y_global = max(subplot_y_start, min(subplot_y_start + subplot_y_height, tick_y_global))
                x_tick = left_col_subplot_left - y_axis_subplot_to_tick
                create_text(x_tick, tick_y_global, str(tick_val), font_size=label_font_size, anchor="end", baseline="middle")
        y_axis_label_x = left_col_subplot_left - y_axis_subplot_to_tick - label_font_size * 2 - y_axis_tick_to_label
        y_axis_label_y = subplot_start_y + subplot_area_h / 2
        y_label_elem = ET.SubElement(labels_group, f"{{{svg_ns}}}text", {
            "x": str(y_axis_label_x), "y": str(y_axis_label_y),
            "font-family": "Arial, sans-serif", "font-size": str(label_font_size),
            "text-anchor": "middle", "dominant-baseline": "middle", "fill": "black",
            "transform": f"rotate(-90 {y_axis_label_x} {y_axis_label_y})"
        })
        y_label_elem.text = to_title_case("Number of parallel tapes")
        # 右边：模式标签
        for row_idx, (_mode_key, mode_label) in enumerate(modes):
            x_pos = subplot_start_x + subplot_area_w + 16
            y_center = subplot_start_y + margin_y + row_idx * (panel_h + margin_y) + panel_h / 2
            create_text(x_pos, y_center, mode_label, font_size=label_font_size, anchor="start", baseline="middle")

        root_svg.remove(labels_group)
        root_svg.append(labels_group)
        tree.write(str(out_path), encoding="utf-8", xml_declaration=True)
        export_svg_to_png(out_path)
        print(f"制冷功率 MWe 九宫格拼接完成 ({cmap_name}): {out_path}")
        # 160 mm 投稿候选(3x3 版式)，见工作文档 §14.42
        _write_submission_candidate(out_path, "FigS10_cryo_power_grid", cmap_name)


def stitch_cryo_power_single_temp_svgs(temp, cmaps=None):
    """
    SI 图 S8/S9：固定温度(4.2K→S8, 10K→S9)下 pulse/dwell/static 三模式的单列拼图。
    = 3×3 网格(stitch_cryo_power_MWe_svgs)某一温度列单独成图；复用全部同款子图与样式，
    保证与 S10 视觉一致。布局：3 行(Pulse/Dwell/Static)×1 列；顶部温度标签、右侧模式标签、
    底部横轴=接头电阻(nΩ)、左侧纵轴=并绕根数；无 colorbar(等高线数字承载 MWe)。
    低温电功率仅随 (mode,T,Npw,R_joint) 变，与 HTS 价格情景无关 → 无 scenario 轴。
    """
    import xml.etree.ElementTree as ET
    modes = [("pulse", "Pulse"), ("dwell", "Dwell"), ("static", "Static")]
    temperatures = [temp]
    n_rows, n_cols = len(modes), 1
    label = f"{int(temp)}K" if temp in (10.0, 20.0) else f"{temp}K"
    cmap_list = cmaps if cmaps is not None else cfg.color_schemes

    for cmap_name in cmap_list:
        cryo_dir = FIGURE_OUTPUT_ROOT / cmap_name / cfg.PARASITIC_RATIO_OUTPUT_DIR / "cryo_power_MWe"
        panel_entries, panel_w, panel_h = [], None, None
        for row_idx, (mode_key, _l) in enumerate(modes):
            fname = cryo_dir / f"cryo_power_MWe_heatmap_{mode_key}_{label}.svg"
            if not fname.exists():
                continue
            fig = sg.fromfile(str(fname)); root = fig.getroot()
            w, h = _parse_size(fig.get_size()[0]), _parse_size(fig.get_size()[1])
            panel_w = w if panel_w is None else max(panel_w, w)
            panel_h = h if panel_h is None else max(panel_h, h)
            panel_entries.append((root, row_idx, 0))
        if not panel_entries or panel_w is None:
            print(f"[warning] 配色 {cmap_name} / {label}: 未找到足够子图，跳过。")
            continue

        margin_x, margin_y = panel_w * 0.1, panel_h * 0.1
        label_font_size = 22
        left_margin = 20
        x_axis_subplot_to_tick, x_axis_tick_to_label = 25, 15
        y_axis_subplot_to_tick, y_axis_tick_to_label = 15, 15
        top_label_height = label_font_size + 8
        bottom_label_height = x_axis_subplot_to_tick + label_font_size + x_axis_tick_to_label + label_font_size
        left_label_width = left_margin + y_axis_subplot_to_tick + label_font_size * 2 + y_axis_tick_to_label + label_font_size
        right_label_width = max(label_font_size * 2 + 16, RIGHT_LABEL_WIDTH_MIN)
        subplot_start_x, subplot_start_y = left_label_width, top_label_height
        subplot_area_w = margin_x * (n_cols + 1) + panel_w * n_cols
        subplot_area_h = margin_y * (n_rows + 1) + panel_h * n_rows
        total_w = subplot_area_w + left_label_width + right_label_width
        total_h = subplot_area_h + top_label_height + bottom_label_height

        fig_out = sg.SVGFigure(f"{total_w}pt", f"{total_h}pt")
        placed = []
        for root, row_idx, col_idx in panel_entries:
            x = subplot_start_x + margin_x + col_idx * (panel_w + margin_x)
            y = subplot_start_y + margin_y + row_idx * (panel_h + margin_y)
            root.moveto(x, y); placed.append(root)
        fig_out.append(placed)

        stitched_dir = cryo_dir / "stitched"; stitched_dir.mkdir(parents=True, exist_ok=True)
        out_path = stitched_dir / f"cryo_power_MWe_grid_3modes_{label}.svg"
        fig_out.save(str(out_path))

        tree = ET.parse(str(out_path)); root_svg = tree.getroot()
        svg_ns = "http://www.w3.org/2000/svg"
        for k, v in root_svg.attrib.items():
            if k.startswith("xmlns") and "svg" in v.lower(): svg_ns = v; break
        ET.register_namespace("", svg_ns)
        root_svg.set("width", f"{total_w}pt"); root_svg.set("height", f"{total_h}pt")
        root_svg.set("viewBox", f"0 0 {total_w} {total_h}")
        labels_group = ET.SubElement(root_svg, f"{{{svg_ns}}}g", {"id": "labels"})

        def create_text(x, y, text, anchor="middle", baseline="middle", rot=None):
            a = {"x": str(x), "y": str(y), "font-family": "Arial, sans-serif",
                 "font-size": str(label_font_size), "text-anchor": anchor,
                 "dominant-baseline": baseline, "fill": "black"}
            if rot is not None: a["transform"] = f"rotate(-90 {x} {y})"
            e = ET.SubElement(labels_group, f"{{{svg_ns}}}text", a); e.text = text; return e

        # 顶部温度标签
        temp_str = str(int(temp)) if temp in (10.0, 20.0) else str(temp)
        create_text(subplot_start_x + margin_x + panel_w / 2, top_label_height - 8, f"{temp_str} K")
        # 底部横轴刻度 + 名称
        bottom = subplot_start_y + margin_y + (n_rows - 1) * (panel_h + margin_y) + panel_h
        sx = subplot_start_x + margin_x
        for tv in X_AXIS_TICKS_NOHM:
            tn = max(0.0, min(1.0, (np.log10(tv) - X_AXIS_MIN_LOG) / (X_AXIS_MAX_LOG - X_AXIS_MIN_LOG)))
            create_text(sx + tn * panel_w, bottom + x_axis_subplot_to_tick, str(tv), baseline="top")
        create_text(subplot_start_x + subplot_area_w / 2,
                    bottom + x_axis_subplot_to_tick + label_font_size + x_axis_tick_to_label,
                    to_title_case("Coil-to-coil joint resistance (nΩ)"), baseline="top")
        # 左侧纵轴刻度 + 名称
        y_min, y_max = Y_AXIS_MIN, Y_AXIS_MAX; yr = y_max - y_min
        lx = subplot_start_x + margin_x
        for row_idx in range(n_rows):
            sy = subplot_start_y + margin_y + row_idx * (panel_h + margin_y)
            for tv in Y_AXIS_TICKS:
                tn = max(0.0, min(1.0, (tv - y_min) / yr))
                create_text(lx - y_axis_subplot_to_tick, sy + panel_h - tn * panel_h, str(tv), anchor="end")
        ylx = lx - y_axis_subplot_to_tick - label_font_size * 2 - y_axis_tick_to_label
        yly = subplot_start_y + subplot_area_h / 2
        create_text(ylx, yly, to_title_case("Number of parallel tapes"), rot=True)
        # 右侧模式标签
        for row_idx, (_k, ml) in enumerate(modes):
            yc = subplot_start_y + margin_y + row_idx * (panel_h + margin_y) + panel_h / 2
            create_text(subplot_start_x + subplot_area_w + 16, yc, ml, anchor="start")

        root_svg.remove(labels_group); root_svg.append(labels_group)
        tree.write(str(out_path), encoding="utf-8", xml_declaration=True)
        export_svg_to_png(out_path)
        print(f"制冷功率 MWe 单温({label})三模式拼接完成 ({cmap_name}): {out_path}")


def generate_readme():
    """
    生成README文档，记录所有组合图的生成条件、参数和子图来源。
    """
    from datetime import datetime
    current_date = datetime.now().strftime("%Y年%m月%d日")
    
    # 获取输出目录
    scan_base_dir = FIGURE_OUTPUT_ROOT
    af_base_dir = FIGURE_OUTPUT_ROOT / "AF_heatmaps_scenarios" / "figures"

    # README 文件路径（保存在主要输出目录）
    readme_path = scan_base_dir / "README.md"
    
    readme_content = f"""# 经济分析组合图说明文档

本文档记录了经济分析相关组合图的生成条件、参数和子图来源。

**文档生成日期：** {current_date}

## 生成脚本

所有组合图由 `9.0_stitch_delta_lcoe_min_svgs.py` 脚本生成。

---

## 输出文件列表

### 1. `delta_lcoe_min_heatmap_grid_S1-S3.svg`
**图片含义：**
- 显示不同技术情景（S1, S2, S3）和不同温度-制冷剂组合下的最小LCOE增量（delta LCOE min）热力图
- 横轴：线圈间接头电阻（Coil-to-coil Joint Resistance），对数刻度，范围 1-100 nΩ
- 纵轴：并绕根数（Number of Parallel Tapes），范围 2-200

**生成条件：**
- 技术情景：S1, S2, S3
- 温度-制冷剂组合：
  - 4.2K He
  - 10K He
  - 20K He
  - 20K H₂

**布局说明：**
- **行（从上到下）**：S1, S2, S3
- **列（从左到右）**：4.2K He, 10K He, 20K He, 20K H₂
- **上方标签**：每个子图对应的温度-制冷剂标签
- **下方标签**：横轴刻度（1, 10, 100 nΩ）和横轴名称 "Coil-to-coil Joint Resistance (nΩ)"
- **左边标签**：纵轴刻度（10, 50, 100, 200）和纵轴名称 "Number of Parallel Tapes"（旋转90度）
- **右边标签**：每个子图对应的情景标签（S1, S2, S3）

**子图文件来源：**
```
{cfg.ECONOMIC_OUTPUT_DIR}/{{配色方案}}/{cfg.PARASITIC_RATIO_OUTPUT_DIR}/{cfg.PARASITIC_RATIO_USD_DIR}/{{情景}}/delta_lcoe_min_heatmap_{{情景}}_{{温度}}K_{{制冷剂}}_{{配色方案}}.svg
```
例如：
- `{cfg.ECONOMIC_OUTPUT_DIR}/viridis/{cfg.PARASITIC_RATIO_OUTPUT_DIR}/{cfg.PARASITIC_RATIO_USD_DIR}/S1/delta_lcoe_min_heatmap_S1_4.2K_He_viridis.svg`
- `{cfg.ECONOMIC_OUTPUT_DIR}/cividis/{cfg.PARASITIC_RATIO_OUTPUT_DIR}/{cfg.PARASITIC_RATIO_USD_DIR}/S1/delta_lcoe_min_heatmap_S1_10.0K_He_cividis.svg`
- `{cfg.ECONOMIC_OUTPUT_DIR}/PuBuGn/{cfg.PARASITIC_RATIO_OUTPUT_DIR}/{cfg.PARASITIC_RATIO_USD_DIR}/S2/delta_lcoe_min_heatmap_S2_20.0K_H2_PuBuGn.svg`

**输出路径：**
```
{cfg.ECONOMIC_OUTPUT_DIR}/{{配色方案}}/{cfg.PARASITIC_RATIO_OUTPUT_DIR}/{cfg.PARASITIC_RATIO_USD_DIR}/stitched/delta_lcoe_min_heatmap_grid_S1-S3_{{配色方案}}.svg
```

---

### 2. `delta_lcoe_min_heatmap_grid_HTS_tape_sensitivity.svg`
**图片含义：**
- 显示HTS带材价格敏感性分析的最小LCOE增量热力图
- 对比不同HTS带材价格（100, 50, 10 constant-2025-US$ kA⁻¹ m⁻¹）对LCOE的影响
- 横轴：线圈间接头电阻（Coil-to-coil Joint Resistance），对数刻度，范围 1-100 nΩ
- 纵轴：并绕根数（Number of Parallel Tapes），范围 2-200

**生成条件：**
- 技术情景：S4, S5, S6（对应HTS带材价格：100, 50, 10 constant-2025-US$ kA⁻¹ m⁻¹）
- 温度-制冷剂组合：
  - 4.2K He
  - 10K He
  - 20K He
  - 20K H₂

**布局说明：**
- **行（从上到下）**：S4 (100 constant-2025-US$ kA⁻¹ m⁻¹), S5 (50 constant-2025-US$ kA⁻¹ m⁻¹), S6 (10 constant-2025-US$ kA⁻¹ m⁻¹)
- **列（从左到右）**：4.2K He, 10K He, 20K He, 20K H₂
- **上方标签**：每个子图对应的温度-制冷剂标签
- **下方标签**：横轴刻度（1, 10, 100 nΩ）和横轴名称 "Coil-to-coil Joint Resistance (nΩ)"
- **左边标签**：纵轴刻度（10, 50, 100, 200）和纵轴名称 "Number of Parallel Tapes"（旋转90度）
- **右边标签**：每个子图对应的HTS带材价格标签（100, 50, 10 constant-2025-US$ kA⁻¹ m⁻¹）

**子图文件来源：**
```
{cfg.ECONOMIC_OUTPUT_DIR}/{{配色方案}}/{cfg.PARASITIC_RATIO_OUTPUT_DIR}/{cfg.PARASITIC_RATIO_USD_DIR}/{{情景}}/delta_lcoe_min_heatmap_{{情景}}_{{温度}}K_{{制冷剂}}_{{配色方案}}.svg
```
例如：
- `{cfg.ECONOMIC_OUTPUT_DIR}/viridis/{cfg.PARASITIC_RATIO_OUTPUT_DIR}/{cfg.PARASITIC_RATIO_USD_DIR}/S4/delta_lcoe_min_heatmap_S4_4.2K_He_viridis.svg`
- `{cfg.ECONOMIC_OUTPUT_DIR}/cividis/{cfg.PARASITIC_RATIO_OUTPUT_DIR}/{cfg.PARASITIC_RATIO_USD_DIR}/S5/delta_lcoe_min_heatmap_S5_20.0K_He_cividis.svg`
- `{cfg.ECONOMIC_OUTPUT_DIR}/PuBuGn/{cfg.PARASITIC_RATIO_OUTPUT_DIR}/{cfg.PARASITIC_RATIO_USD_DIR}/S6/delta_lcoe_min_heatmap_S6_20.0K_H2_PuBuGn.svg`

**输出路径：**
```
{cfg.ECONOMIC_OUTPUT_DIR}/{{配色方案}}/{cfg.PARASITIC_RATIO_OUTPUT_DIR}/{cfg.PARASITIC_RATIO_USD_DIR}/stitched/delta_lcoe_min_heatmap_grid_HTS_tape_sensitivity_{{配色方案}}.svg
```

---

### 3. `parasitic_ratio_heatmap_grid_S1-S3.svg`
**图片含义：**
- 显示不同技术情景（S1, S2, S3）和不同温度下的寄生功耗占比（parasitic ratio）热力图
- 横轴：线圈间接头电阻（Coil-to-coil Joint Resistance），对数刻度，范围 1-100 nΩ
- 纵轴：并绕根数（Number of Parallel Tapes），范围 2-200
- 所有子图都使用He作为制冷剂

**生成条件：**
- 技术情景：S1, S2, S3
- 温度：4.2K, 10K, 20K（都是He）

**布局说明：**
- **行（从上到下）**：S1, S2, S3
- **列（从左到右）**：4.2K, 10K, 20K
- **上方标签**：每个子图对应的温度标签（4.2K, 10K, 20K）
- **下方标签**：横轴刻度（1, 10, 100 nΩ）和横轴名称 "Coil-to-coil Joint Resistance (nΩ)"
- **左边标签**：纵轴刻度（10, 50, 100, 200）和纵轴名称 "Number of Parallel Tapes"（旋转90度）
- **右边标签**：每个子图对应的情景标签（S1, S2, S3）

**子图文件来源：**
```
{cfg.ECONOMIC_OUTPUT_DIR}/{{配色方案}}/{cfg.PARASITIC_RATIO_OUTPUT_DIR}/{cfg.PARASITIC_RATIO_USD_DIR}/{{情景}}/parasitic_ratio_heatmap_{{情景}}_{{温度}}K_{{配色方案}}.svg
```
例如：
- `{cfg.ECONOMIC_OUTPUT_DIR}/viridis/{cfg.PARASITIC_RATIO_OUTPUT_DIR}/{cfg.PARASITIC_RATIO_USD_DIR}/S1/parasitic_ratio_heatmap_S1_4.2K_viridis.svg`
- `{cfg.ECONOMIC_OUTPUT_DIR}/cividis/{cfg.PARASITIC_RATIO_OUTPUT_DIR}/{cfg.PARASITIC_RATIO_USD_DIR}/S2/parasitic_ratio_heatmap_S2_10.0K_cividis.svg`
- `{cfg.ECONOMIC_OUTPUT_DIR}/PuBuGn/{cfg.PARASITIC_RATIO_OUTPUT_DIR}/{cfg.PARASITIC_RATIO_USD_DIR}/S3/parasitic_ratio_heatmap_S3_20.0K_PuBuGn.svg`

**输出路径：**
```
{cfg.ECONOMIC_OUTPUT_DIR}/{{配色方案}}/{cfg.PARASITIC_RATIO_OUTPUT_DIR}/{cfg.PARASITIC_RATIO_USD_DIR}/stitched/parasitic_ratio_heatmap_grid_S1-S3_{{配色方案}}.svg
```

---

### 4. `AF_heatmap_grid_S1-S3.svg`
**图片含义：**
- 显示不同技术情景（S1, S2, S3）和不同温度下的AF（Availability Factor，可用性因子）热力图
- 横轴：匝间电阻率（Turn-to-turn Resistivity），对数刻度，范围 10-1000 μΩ·cm²
- 纵轴：并绕根数（Number of Parallel Tapes），范围 1-20

**生成条件：**
- 技术情景：S1, S2, S3
- 温度：4.2K, 10K, 20K

**布局说明：**
- **行（从上到下）**：S1, S2, S3
- **列（从左到右）**：4.2K, 10K, 20K
- **上方标签**：每个子图对应的温度标签（4.2K, 10K, 20K）
- **下方标签**：横轴刻度（10, 100, 1000）和横轴名称 "Turn-to-turn Resistivity (μΩ·cm²)"
- **左边标签**：纵轴刻度（1, 5, 10, 20）和纵轴名称 "Number of Parallel Tapes"（旋转90度）
- **右边标签**：每个子图对应的情景标签（S1, S2, S3）

**子图文件来源：**
```
{cfg.ECONOMIC_OUTPUT_DIR}/{{配色方案}}/AF_heatmaps_scenarios/figures/{{情景}}/AF_heatmap_{{情景}}_Top_{{温度}}K.svg
```
例如：
- `{cfg.ECONOMIC_OUTPUT_DIR}/viridis/AF_heatmaps_scenarios/figures/S1/AF_heatmap_S1_Top_4.2K.svg`
- `{cfg.ECONOMIC_OUTPUT_DIR}/cividis/AF_heatmaps_scenarios/figures/S2/AF_heatmap_S2_Top_10.0K.svg`
- `{cfg.ECONOMIC_OUTPUT_DIR}/PuBuGn/AF_heatmaps_scenarios/figures/S3/AF_heatmap_S3_Top_20.0K.svg`

**输出路径：**
```
{cfg.ECONOMIC_OUTPUT_DIR}/{{配色方案}}/AF_heatmaps_scenarios/figures/AF_heatmap_grid_S1-S3.svg
```

---

## 技术情景说明

| 情景 | HTS带材价格 | 说明 |
|------|-------------|------|
| S1   | 100 constant-2025-US$ kA⁻¹ m⁻¹ | 主情景 |
| S2   | 50 constant-2025-US$ kA⁻¹ m⁻¹ | 主情景 |
| S3   | 10 constant-2025-US$ kA⁻¹ m⁻¹ | 主情景 |
| S4   | 100 constant-2025-US$ kA⁻¹ m⁻¹ | HTS 带材价格敏感性（其余同 S2） |
| S5   | 50 constant-2025-US$ kA⁻¹ m⁻¹ | HTS 带材价格敏感性（其余同 S2） |
| S6   | 10 constant-2025-US$ kA⁻¹ m⁻¹ | HTS 带材价格敏感性（其余同 S2） |

---

## 子图生成说明

所有子图由经济分析脚本（如 `8.0_run_economic_analysis_unified.py`）生成。

**子图类型：**
1. **delta_lcoe_min_heatmap**：最小LCOE增量热力图
2. **parasitic_ratio_heatmap**：寄生功耗占比热力图
3. **AF_heatmap**：可用性因子热力图

**子图文件命名规则：**
- `delta_lcoe_min_heatmap_{{情景}}_{{温度}}K_{{制冷剂}}.svg`
- `parasitic_ratio_heatmap_{{情景}}_{{温度}}K.svg`
- `AF_heatmap_{{情景}}_Top_{{温度}}K.svg`

---

## 文件路径结构

**注意：** 所有文件现在都按配色方案组织，每个配色方案有独立的文件夹。

```
{cfg.ECONOMIC_OUTPUT_DIR}/
├── {{配色方案}}/  # 例如：viridis, cividis, PuBuGn 等
│   ├── {cfg.PARASITIC_RATIO_OUTPUT_DIR}/
│   │   └── {cfg.PARASITIC_RATIO_USD_DIR}/
│   │       ├── stitched/
│   │       │   ├── delta_lcoe_min_heatmap_grid_S1-S3_{{配色方案}}.svg
│   │       │   ├── delta_lcoe_min_heatmap_grid_HTS_tape_sensitivity_{{配色方案}}.svg
│   │       │   └── parasitic_ratio_heatmap_grid_S1-S3_{{配色方案}}.svg
│   │       └── {{情景}}/
│   │           ├── delta_lcoe_min_heatmap_{{情景}}_{{温度}}K_{{制冷剂}}_{{配色方案}}.svg
│   │           └── parasitic_ratio_heatmap_{{情景}}_{{温度}}K_{{配色方案}}.svg
│   │
│   ├── AF_heatmaps_scenarios/
│   │   └── figures/
│   │       ├── AF_heatmap_grid_S1-S3.svg
│   │       └── {{情景}}/
│   │           └── AF_heatmap_{{情景}}_Top_{{温度}}K.svg
│   │
│   └── AF_ref_heatmaps_scenarios/
│       └── figures/
│           ├── AF_ref_heatmap_grid_S1-S3.svg
│           └── {{情景}}/
│               └── AF_ref_heatmap_{{情景}}_Top_{{温度}}K.svg
```

---

## 备注

1. 所有路径相对于项目根目录。
2. 子图文件必须存在才能生成组合图，缺失的子图会在控制台输出警告。
3. 组合图会自动添加坐标轴标签、刻度标签和情景标签。
4. 本文档由 `9.0_stitch_delta_lcoe_min_svgs.py` 脚本自动生成，每次运行脚本时会自动更新。
"""
    
    # 写入README文件
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"\n[OK] README文档已生成: {readme_path}")


if __name__ == "__main__":
    # 快速调试模式 FAST_FIG=1: 只拼 delta_LCOE / HTS价格 / 再循环 三张, 跳过 cryo / af / af_ref。
    _FAST = _os.environ.get("FAST_FIG", "").lower() in ("1", "true", "yes")
    stitch_delta_lcoe_min_svgs()         # Fig5
    stitch_hts_tape_sensitivity_svgs()   # Fig6
    stitch_parasitic_ratio_svgs()        # Fig4
    if not _FAST:
        stitch_cryo_power_MWe_svgs()             # S10: 3×3 模式×温度
        stitch_cryo_power_single_temp_svgs(4.2)  # S8: Top=4.2K 三模式单列
        stitch_cryo_power_single_temp_svgs(10.0) # S9: Top=10K 三模式单列
        stitch_af_heatmap_svgs()
        stitch_af_ref_heatmap_svgs()
        # 生成README文档
        print(f"\n生成README文档...")
        generate_readme()

