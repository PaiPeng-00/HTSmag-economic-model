from pathlib import Path
import cairosvg

svg = Path("test.svg")
pdf = svg.with_suffix(".pdf")

cairosvg.svg2pdf(
    url=str(svg),
    write_to=str(pdf),
)

print("OK:", pdf)
