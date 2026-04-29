#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


W = 720.0
H = 255.0

COLORS = {
    "black": "#222222",
    "axis": "#333333",
    "grid": "#E6E6E6",
    "blue": "#3B6EA8",
    "orange": "#D55E00",
    "green": "#009E73",
    "green_light": "#E5F2ED",
    "gray": "#8A8F98",
    "gray_light": "#ECEDEF",
    "red": "#B03A2E",
}


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def hex_to_rgb(color: str) -> tuple[float, float, float]:
    named = {
        "white": "#FFFFFF",
        "black": "#000000",
    }
    color = named.get(color.lower(), color)
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def esc_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def esc_pdf(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class Svg:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def add(self, raw: str) -> None:
        self.parts.append(raw)

    def rect(self, x: float, y: float, w: float, h: float, fill: str = "none",
             stroke: str = "none", sw: float = 1.0, opacity: float | None = None) -> None:
        extra = "" if opacity is None else f' opacity="{opacity:.3f}"'
        self.add(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw:.2f}"{extra}/>'
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, stroke: str,
             sw: float = 1.0, dash: str | None = None) -> None:
        extra = "" if dash is None else f' stroke-dasharray="{dash}"'
        self.add(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{stroke}" stroke-width="{sw:.2f}"{extra}/>'
        )

    def circle(self, x: float, y: float, r: float, fill: str,
               stroke: str = "none", sw: float = 1.0, opacity: float = 1.0) -> None:
        self.add(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw:.2f}" opacity="{opacity:.3f}"/>'
        )

    def cross(self, x: float, y: float, r: float, stroke: str,
              sw: float = 1.4, opacity: float = 1.0) -> None:
        self.add(
            f'<line x1="{x-r:.2f}" y1="{y-r:.2f}" x2="{x+r:.2f}" y2="{y+r:.2f}" '
            f'stroke="{stroke}" stroke-width="{sw:.2f}" opacity="{opacity:.3f}" '
            f'stroke-linecap="round"/>'
        )
        self.add(
            f'<line x1="{x-r:.2f}" y1="{y+r:.2f}" x2="{x+r:.2f}" y2="{y-r:.2f}" '
            f'stroke="{stroke}" stroke-width="{sw:.2f}" opacity="{opacity:.3f}" '
            f'stroke-linecap="round"/>'
        )

    def text(self, x: float, y: float, text: str, size: float = 10.0,
             fill: str = COLORS["black"], anchor: str = "start",
             weight: str = "normal") -> None:
        self.add(
            f'<text x="{x:.2f}" y="{y:.2f}" font-family="Helvetica, Arial, Nimbus Sans, sans-serif" '
            f'font-size="{size:.2f}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{esc_xml(text)}</text>'
        )

    def save(self, path: Path) -> None:
        header = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="7.2in" height="2.55in" '
            f'viewBox="0 0 {W:.0f} {H:.0f}">\n'
            '<rect width="100%" height="100%" fill="white"/>\n'
        )
        path.write_text(header + "\n".join(self.parts) + "\n</svg>\n", encoding="utf-8")


class Pdf:
    def __init__(self) -> None:
        self.ops: list[str] = []

    def y(self, y_top: float) -> float:
        return H - y_top

    def color(self, color: str, stroke: bool = False) -> None:
        r, g, b = hex_to_rgb(color)
        self.ops.append(f"{r:.4f} {g:.4f} {b:.4f} {'RG' if stroke else 'rg'}")

    def rect(self, x: float, y: float, w: float, h: float, fill: str = "none",
             stroke: str = "none", sw: float = 1.0) -> None:
        y_pdf = self.y(y + h)
        if fill != "none" and stroke != "none":
            self.color(fill, stroke=False)
            self.color(stroke, stroke=True)
            self.ops.append(f"{sw:.2f} w {x:.2f} {y_pdf:.2f} {w:.2f} {h:.2f} re B")
        elif fill != "none":
            self.color(fill, stroke=False)
            self.ops.append(f"{x:.2f} {y_pdf:.2f} {w:.2f} {h:.2f} re f")
        elif stroke != "none":
            self.color(stroke, stroke=True)
            self.ops.append(f"{sw:.2f} w {x:.2f} {y_pdf:.2f} {w:.2f} {h:.2f} re S")

    def line(self, x1: float, y1: float, x2: float, y2: float, stroke: str,
             sw: float = 1.0, dash: bool = False) -> None:
        self.color(stroke, stroke=True)
        if dash:
            self.ops.append("[3 2] 0 d")
        self.ops.append(
            f"{sw:.2f} w {x1:.2f} {self.y(y1):.2f} m {x2:.2f} {self.y(y2):.2f} l S"
        )
        if dash:
            self.ops.append("[] 0 d")

    def circle(self, x: float, y: float, r: float, fill: str,
               stroke: str = "none", sw: float = 1.0) -> None:
        k = 0.5522847498
        cy = self.y(y)
        c = k * r
        self.color(fill, stroke=False)
        if stroke != "none":
            self.color(stroke, stroke=True)
        op = "B" if stroke != "none" else "f"
        self.ops.append(
            f"{sw:.2f} w "
            f"{x + r:.2f} {cy:.2f} m "
            f"{x + r:.2f} {cy + c:.2f} {x + c:.2f} {cy + r:.2f} {x:.2f} {cy + r:.2f} c "
            f"{x - c:.2f} {cy + r:.2f} {x - r:.2f} {cy + c:.2f} {x - r:.2f} {cy:.2f} c "
            f"{x - r:.2f} {cy - c:.2f} {x - c:.2f} {cy - r:.2f} {x:.2f} {cy - r:.2f} c "
            f"{x + c:.2f} {cy - r:.2f} {x + r:.2f} {cy - c:.2f} {x + r:.2f} {cy:.2f} c "
            f"h {op}"
        )

    def cross(self, x: float, y: float, r: float, stroke: str,
              sw: float = 1.4) -> None:
        self.color(stroke, stroke=True)
        self.ops.append(
            f"{sw:.2f} w {x-r:.2f} {self.y(y-r):.2f} m {x+r:.2f} {self.y(y+r):.2f} l S"
        )
        self.ops.append(
            f"{sw:.2f} w {x-r:.2f} {self.y(y+r):.2f} m {x+r:.2f} {self.y(y-r):.2f} l S"
        )

    def text(self, x: float, y: float, text: str, size: float = 10.0,
             fill: str = COLORS["black"], anchor: str = "start",
             weight: str = "normal") -> None:
        width = 0.47 * size * len(text)
        tx = x
        if anchor == "middle":
            tx -= width / 2.0
        elif anchor == "end":
            tx -= width
        self.color(fill, stroke=False)
        font = "F2" if weight == "bold" else "F1"
        self.ops.append(
            f"BT /{font} {size:.2f} Tf {tx:.2f} {self.y(y):.2f} Td ({esc_pdf(text)}) Tj ET"
        )

    def save(self, path: Path) -> None:
        stream = "\n".join(self.ops).encode("latin-1")
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W:.0f} {H:.0f}] "
                f"/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> "
                f"/Contents 6 0 R >>"
            ).encode("latin-1"),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
            b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
        ]
        chunks = [b"%PDF-1.4\n"]
        offsets = [0]
        for i, obj in enumerate(objects, start=1):
            offsets.append(sum(len(c) for c in chunks))
            chunks.append(f"{i} 0 obj\n".encode("ascii") + obj + b"\nendobj\n")
        xref = sum(len(c) for c in chunks)
        chunks.append(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        chunks.append(b"0000000000 65535 f \n")
        for off in offsets[1:]:
            chunks.append(f"{off:010d} 00000 n \n".encode("ascii"))
        chunks.append(
            f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n".encode("ascii")
        )
        path.write_bytes(b"".join(chunks))


class Canvas:
    def __init__(self) -> None:
        self.svg = Svg()
        self.pdf = Pdf()

    def rect(self, *args: Any, **kwargs: Any) -> None:
        self.svg.rect(*args, **kwargs)
        kwargs.pop("opacity", None)
        self.pdf.rect(*args, **kwargs)

    def line(self, *args: Any, **kwargs: Any) -> None:
        self.svg.line(*args, **kwargs)
        dash = kwargs.pop("dash", None)
        self.pdf.line(*args, dash=bool(dash), **kwargs)

    def circle(self, *args: Any, **kwargs: Any) -> None:
        self.svg.circle(*args, **kwargs)
        kwargs.pop("opacity", None)
        self.pdf.circle(*args, **kwargs)

    def cross(self, *args: Any, **kwargs: Any) -> None:
        self.svg.cross(*args, **kwargs)
        kwargs.pop("opacity", None)
        self.pdf.cross(*args, **kwargs)

    def text(self, *args: Any, **kwargs: Any) -> None:
        self.svg.text(*args, **kwargs)
        self.pdf.text(*args, **kwargs)


def draw_figure(policy: dict[str, Any], out_svg: Path, out_pdf: Path) -> None:
    observations_all = policy["analysis"]["observations"]
    observations = [
        obs for obs in observations_all
        if obs.get("metadata", {}).get("candidate_graph_allowed") is True
    ]
    n_total = len(observations_all)
    n = len(observations)
    wins = sum(1 for obs in observations if bool(obs.get("useful")))
    losses = n - wins
    correct = sum(1 for obs in observations if bool(obs.get("correct")))

    c = Canvas()
    px, py, pw, ph = 75.0, 28.0, 575.0, 175.0
    c.rect(px, py, pw, ph, fill="white", stroke=COLORS["axis"], sw=0.8)

    xmin, xmax = math.log10(64), math.log10(4096)
    ymin, ymax = 0.82, 1.62

    def sx(tokens: float) -> float:
        return px + (math.log10(max(64.0, min(4096.0, tokens))) - xmin) / (xmax - xmin) * pw

    def sy(speedup: float) -> float:
        return py + (ymax - speedup) / (ymax - ymin) * ph

    y_one = sy(1.0)
    c.rect(px, py, pw, y_one - py, fill="#EAF2FB", stroke="none")
    c.rect(px, y_one, pw, py + ph - y_one, fill="#FBEDE5", stroke="none")
    c.rect(px, py, pw, ph, fill="none", stroke=COLORS["axis"], sw=0.8)

    for ytick in [0.8, 1.0, 1.2, 1.4, 1.6]:
        y = sy(ytick)
        c.line(px, y, px + pw, y, COLORS["grid"], sw=0.6)
        c.text(px - 10, y + 6.0, f"{ytick:.1f}", size=16.0, anchor="end")
    c.line(px, y_one, px + pw, y_one, COLORS["red"], sw=1.1, dash="4,3")
    c.text(px + pw - 5, y_one - 8, "fallback parity", size=16.0, fill=COLORS["red"], anchor="end")

    for xtick in [64, 256, 1024, 4096]:
        x = sx(xtick)
        c.line(x, py, x, py + ph, COLORS["grid"], sw=0.6)
        c.text(x, py + ph + 21, str(xtick), size=16.0, anchor="middle")

    # Points: only rows that the static candidate policy would graph-replay.
    for obs in observations:
        speed = float(obs["fallback_ms"]) / float(obs["graph_ms"])
        x = sx(float(obs["tokens"]))
        y = sy(speed)
        if speed >= 1.0:
            c.cross(x, y, 3.1, COLORS["blue"], sw=1.35)
        else:
            c.cross(x, y, 3.1, COLORS["orange"], sw=1.35)

    c.text(px + pw / 2, py + ph + 45, "input tokens (log scale)", size=18.0, anchor="middle")

    # Legend.
    lx, ly = px + 14, py + 17
    c.cross(lx, ly, 3.1, COLORS["blue"], sw=1.35)
    c.text(lx + 12, ly + 6, "faster replay", size=16.0)
    c.cross(lx + 145, ly, 3.1, COLORS["orange"], sw=1.35)
    c.text(lx + 157, ly + 6, "slower replay", size=16.0)

    c.text(px + pw - 8, y_one + 25, f"{losses}/{n} graph replays are slower", size=16.0,
           fill=COLORS["orange"], anchor="end", weight="bold")

    out_svg.parent.mkdir(parents=True, exist_ok=True)
    c.svg.save(out_svg)
    c.pdf.save(out_pdf)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        default="results/runtime_policy_vllm_qwen3_32b_64_residual_learned_from_explore.json",
    )
    parser.add_argument("--out-svg", default="figures/figure1_motivation.svg")
    parser.add_argument("--out-pdf", default="figures/figure1_motivation.pdf")
    args = parser.parse_args()
    draw_figure(load_json(args.policy), Path(args.out_svg), Path(args.out_pdf))
    print(args.out_svg)
    print(args.out_pdf)


if __name__ == "__main__":
    main()
