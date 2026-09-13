"""Regression tests for upstream publication SVG finalization."""

from pathlib import Path
from tempfile import TemporaryDirectory
import re
import unittest

from lxml import etree

from fusion_tem.publication_svg import finalize_economic_heatmap_grid


SVG_NS = "http://www.w3.org/2000/svg"
NS = {"svg": SVG_NS}


class PublicationSvgTests(unittest.TestCase):
    def test_economic_grid_has_submission_geometry_and_typography(self):
        root = etree.Element(
            f"{{{SVG_NS}}}svg",
            width="1856pt",
            height="1338pt",
            viewBox="0 0 1856 1338",
        )
        artwork = etree.SubElement(root, f"{{{SVG_NS}}}g")
        for index in range(12):
            panel = etree.SubElement(
                artwork,
                f"{{{SVG_NS}}}g",
                transform=f"translate({index} {index}) scale(1 1)",
            )
            if index == 0:
                collection = etree.SubElement(
                    panel, f"{{{SVG_NS}}}g", id="PathCollection_1"
                )
                text = etree.SubElement(collection, f"{{{SVG_NS}}}g", id="text_1")
                etree.SubElement(
                    text,
                    f"{{{SVG_NS}}}path",
                    d="M 0 0 L 18 0 L 18 18 L 0 18 Z",
                )

        labels = etree.SubElement(root, f"{{{SVG_NS}}}g", id="labels")
        values = [
            "4.2 K He",
            "10 K He",
            "20 K He",
            "20 K H2",
            *(["1", "10", "100"] * 4),
            "Coil-to-coil joint resistance (nΩ)",
            *(["10", "50", "100", "200"] * 3),
            "Number of parallel tapes",
            "S1",
            "S2",
            "S3",
        ]
        y_values = (
            [22.0] * 4
            + [1243.0] * 12
            + [1280.0]
            + [409.7186, 337.3568, 246.9045, 66.0] * 3
            + [642.0]
            + [246.0, 642.0, 1038.0]
        )
        self.assertEqual(len(values), 33)
        for value, y in zip(values, y_values, strict=True):
            node = etree.SubElement(labels, f"{{{SVG_NS}}}text", x="0", y=f"{y:g}")
            node.text = value

        with TemporaryDirectory() as directory:
            path = Path(directory) / "Fig5.svg"
            etree.ElementTree(root).write(
                str(path), encoding="utf-8", xml_declaration=True
            )
            result = finalize_economic_heatmap_grid(path)
            final_root = etree.parse(str(path)).getroot()
            first_transform = final_root.xpath(".//svg:g[@id='text_1']", namespaces=NS)[
                0
            ].get("transform")
            finalize_economic_heatmap_grid(path)
            final_root = etree.parse(str(path)).getroot()
            second_transform = final_root.xpath(
                ".//svg:g[@id='text_1']", namespaces=NS
            )[0].get("transform")

        self.assertEqual(first_transform, second_transform)
        self.assertEqual(final_root.get("width"), "160mm")
        self.assertEqual(final_root.get("viewBox"), "0 0 2200 1600")
        self.assertEqual(final_root.get("data-font-standard-pt"), "7")
        self.assertEqual(final_root.get("data-font-contour-pt"), "6")
        self.assertEqual(final_root.get("data-font-panel-pt"), "9")
        panels = final_root.xpath("./svg:g[1]/svg:g", namespaces=NS)
        self.assertEqual(len(panels), 12)
        self.assertIn("translate(220 100)", panels[0].get("transform"))
        self.assertIn("translate(1540 1060)", panels[-1].get("transform"))
        letters = final_root.xpath(
            "./svg:g[@id='main_panel_labels']/svg:text", namespaces=NS
        )
        self.assertEqual([node.text for node in letters], list("ABCDEFGHIJKL"))
        self.assertTrue(
            all(node.get("data-final-font-size-pt") == "9" for node in letters)
        )
        contour = final_root.xpath(".//svg:g[@id='text_1']", namespaces=NS)
        self.assertEqual(contour[0].get("data-contour-font-size-pt"), "6")
        self.assertEqual(result["path_text_groups"]["contour"], 1)

    def test_submission_candidate_hooks_are_scoped_to_fig5_and_fig6(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts/9_figures/9.0_stitch_econimic_svgs.py"
        )
        source = script.read_text(encoding="utf-8")
        # Fig. 5 的图名是个表达式（FIG5_YLOG 时输出 Fig5-1，见 9.0 的同名开关），
        # 所以按调用点计数而非字面串：每张图仍应恰好一个候选图写出点。
        calls = re.findall(
            r"_write_submission_candidate\(\s*out_path,\s*(.+?),", source
        )
        self.assertEqual(sum("Fig5" in c for c in calls), 1, calls)
        self.assertEqual(sum("Fig6" in c for c in calls), 1, calls)


if __name__ == "__main__":
    unittest.main()
