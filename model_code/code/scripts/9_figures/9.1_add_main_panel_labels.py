"""Normalize main-text panel labels to bold uppercase letters.

This publication post-processing step does not alter data, axes, contours, or
layout. It is idempotent and writes labelled master SVGs to code-model/figures.
"""

from __future__ import annotations

import copy
import re
import string
from pathlib import Path

from lxml import etree


ROOT = Path(__file__).resolve().parents[4]
MASTER = ROOT / "code-model" / "figures"
OUTPUTS = ROOT / "code-model" / "code" / "outputs" / "arc_16pancake_nuc600"
SVG_NS = "http://www.w3.org/2000/svg"
NS = {"svg": SVG_NS}

SOURCES = {
    "Fig1": MASTER / "Fig1_availability.svg",
    "Fig2": MASTER / "Fig2_heatload.svg",
    "Fig3": OUTPUTS / "figures" / "heatload" / "relative_heatload_grids"
    / "operation_comparison_20K_Rj_10vs100nOhm.svg",
    "Fig4": MASTER / "Fig4_recirc_fraction.svg",
    "Fig5": MASTER / "Fig5_deltaLCOE.svg",
    "Fig6": MASTER / "Fig6_HTS_price.svg",
}

TARGETS = {
    "Fig1": MASTER / "Fig1_availability.svg",
    "Fig2": MASTER / "Fig2_heatload.svg",
    "Fig3": MASTER / "Fig3_heatload_joint.svg",
    "Fig4": MASTER / "Fig4_recirc_fraction.svg",
    "Fig5": MASTER / "Fig5_deltaLCOE.svg",
    "Fig6": MASTER / "Fig6_HTS_price.svg",
}

GRID_ORIGINS = {
    "Fig1": [(x, y) for y in (84.0, 498.0, 912.0) for x in (170.0, 584.0, 998.0)],
    "Fig4": [(x, y) for y in (66.0, 462.0, 858.0) for x in (152.0, 548.0, 944.0)],
    "Fig5": [(x, y) for y in (66.0, 462.0, 858.0) for x in (152.0, 548.0, 944.0, 1340.0)],
    "Fig6": [(x, y) for y in (66.0, 462.0, 858.0) for x in (152.0, 548.0, 944.0, 1340.0)],
}


def text_content(node: etree._Element) -> str:
    return "".join(node.itertext()).strip()


def uppercase_existing_labels(root: etree._Element) -> tuple[int, int]:
    """Uppercase one-letter panel labels and report changed/total labels."""
    changed = 0
    total = 0
    for node in root.xpath(".//svg:text", namespaces=NS):
        value = text_content(node)
        if re.fullmatch(r"[A-Za-z]", value):
            total += 1
            if value.islower():
                node.text = value.upper()
                changed += 1
    return changed, total


def add_grid_labels(root: etree._Element, origins: list[tuple[float, float]]) -> None:
    for old in root.xpath("./svg:g[@id='main_panel_labels']", namespaces=NS):
        root.remove(old)
    group = etree.Element(f"{{{SVG_NS}}}g", id="main_panel_labels")
    for letter, (x0, y0) in zip(string.ascii_uppercase, origins):
        label = etree.SubElement(
            group,
            f"{{{SVG_NS}}}text",
            x=f"{x0 + 10:g}",
            y=f"{y0 + 32:g}",
            **{
                "font-family": "Arial, Helvetica, sans-serif",
                "font-size": "32",
                "font-weight": "bold",
                "fill": "#000000",
                "stroke": "#ffffff",
                "stroke-width": "3",
                "stroke-linejoin": "round",
                "paint-order": "stroke",
            },
        )
        label.text = letter
    root.append(group)


def process(stem: str) -> None:
    parser = etree.XMLParser(remove_blank_text=False, huge_tree=True)
    source = SOURCES[stem]
    if not source.is_file():
        raise FileNotFoundError(source)
    tree = etree.parse(str(source), parser)
    root = copy.deepcopy(tree.getroot())
    if stem in GRID_ORIGINS:
        add_grid_labels(root, GRID_ORIGINS[stem])
    else:
        expected = {"Fig2": 4, "Fig3": 2}[stem]
        changed, total = uppercase_existing_labels(root)
        if total != expected:
            raise RuntimeError(f"{stem}: expected {expected} panel labels, found {total}")
    target = TARGETS[stem]
    target.parent.mkdir(parents=True, exist_ok=True)
    etree.ElementTree(root).write(
        str(target), encoding="utf-8", xml_declaration=True, pretty_print=False
    )
    print(f"{stem}: {source.name} -> {target.name}")


def main() -> None:
    for stem in SOURCES:
        process(stem)


if __name__ == "__main__":
    main()
