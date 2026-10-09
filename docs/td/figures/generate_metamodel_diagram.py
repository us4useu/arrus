#!/usr/bin/env python3
"""Generate a UML-style class diagram of a YAML metamodel.

Reads a metamodel YAML (default: ndarray_metamodel.yaml), emits Graphviz DOT,
renders to PNG/SVG. Also prints a plain-text summary usable in the design doc.

Supported YAML shape (same for L1 graph metamodel and L3 domain model):
    node_kinds:
      NodeName:
        kind: abstract | concrete
        extends: OtherNode     # optional
        doc: ...
        attributes:
          - {name, type, required, unit, doc}
    edge_kinds:
      EdgeName:
        from: NodeName
        to:   NodeName
        from_cardinality: [lo, hi | "*"]
        to_cardinality:   [lo, hi | "*"]
        ordered: bool          # optional
        named_constraints: [...]  # optional
        attributes:            # optional
          - {name, type, ...}
    global_constraints:
      ConstraintName:
        doc: ...

Requires: pyyaml, graphviz `dot` on PATH.

Usage:
    python3 generate_metamodel_diagram.py [YAML_PATH]

Defaults to ndarray_metamodel.yaml (L1 graph metamodel).
"""

from __future__ import annotations

import argparse
import subprocess
import textwrap
from pathlib import Path
from typing import Mapping

import yaml

HERE = Path(__file__).resolve().parent

# Fill colors keyed to node names. Anything not listed falls back to a neutral
# tint. Chosen to sit next to the fig 3 palette without clashing.
NODE_COLOR = {
    # L1 graph metamodel
    "Op":               "#FFE7CC",   # light orange
    "Variable":         "#F9E7A0",   # gold
    "TriggerSource":    "#F2C4C4",   # rose
    "Value":            "#D8DEF5",   # indigo tint
    "Device":           "#C8E7E8",   # teal
    "Stream":           "#D9D9D9",   # gray

    # L3 ultrasound domain model
    "DomainDevice":     "#E5EFF7",   # light blue
    "UltrasoundDevice": "#D4E6F5",
    "Us4R":             "#C8E7E8",
    "Us4OEM":           "#DFF0E1",
    "File":             "#EDE7F6",
    "CPU":              "#D8F5D8",
    "GPU":              "#FFE0B2",
    "Probe":            "#FBE7A0",
    "ProbeModel":       "#FFF3B0",
    "Lens":             "#FFF0F0",
    "MatchingLayer":    "#FFF0F0",
    "Backplane":        "#ECEFF1",
    "GpuSettings":      "#F5F5F5",
    "DeviceId":         "#F5F5F5",
    "ProbeModelId":     "#F5F5F5",
    "HVPSMeasurement":  "#F5F5F5",
    "Variant":          "#F5F5F5",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_card(card) -> str:
    lo, hi = card
    if lo == hi:
        return str(lo)
    if hi == "*":
        return f"{lo}..*"
    return f"{lo}..{hi}"


def _esc(s: str) -> str:
    """Escape characters that need HTML-escaping in Graphviz HTML labels."""
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
    )


def _wrap(text: str, width: int = 42) -> str:
    text = " ".join(text.split())
    return "<BR/>".join(_esc(line) for line in textwrap.wrap(text, width=width))


def _fmt_attribute(attr: Mapping) -> str:
    """Render one attribute in UML notation:
        + name : Type [unit] {required?}
    """
    name = attr["name"]
    tp   = attr.get("type", "?")
    unit = attr.get("unit")
    required = attr.get("required", False)
    parts = [f"<B>{_esc(name)}</B> : {_esc(str(tp))}"]
    if unit:
        parts.append(f'<FONT POINT-SIZE="7">[{_esc(str(unit))}]</FONT>')
    marker = "+" if required else "-"
    return f'{marker}&nbsp;' + " ".join(parts)


def _first_sentence(doc: str, max_chars: int = 110) -> str:
    """Return the first sentence of the docstring, truncated to max_chars."""
    text = " ".join(doc.split())
    for sep in [". ", "; "]:
        idx = text.find(sep)
        if 0 < idx < max_chars:
            return text[: idx + 1]
    return text[:max_chars] + ("…" if len(text) > max_chars else "")


# ---------------------------------------------------------------------------
# Node label (UML class box)
# ---------------------------------------------------------------------------

def _node_label(name: str, spec: Mapping) -> str:
    stereotype = f"«{spec['kind']}»"
    attrs = spec.get("attributes") or []

    # Compose the HTML table for the node.
    rows = [
        f'<TR><TD ALIGN="CENTER"><FONT POINT-SIZE="10">{stereotype}</FONT></TD></TR>',
        f'<TR><TD ALIGN="CENTER"><B><FONT POINT-SIZE="13">{_esc(name)}</FONT></B></TD></TR>',
    ]

    if attrs:
        # UML attribute compartment: HR separator + one row per attribute
        rows.append('<HR/>')
        for attr in attrs:
            rows.append(
                f'<TR><TD ALIGN="LEFT">'
                f'<FONT POINT-SIZE="9">{_fmt_attribute(attr)}</FONT>'
                f'</TD></TR>'
            )

    doc = spec.get("doc", "")
    if doc:
        # Short doc footer (italic)
        rows.append('<HR/>')
        rows.append(
            f'<TR><TD ALIGN="LEFT">'
            f'<FONT POINT-SIZE="8"><I>{_esc(_first_sentence(doc))}</I></FONT>'
            f'</TD></TR>'
        )

    body = "".join(rows)
    return (
        f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="2">'
        f'{body}'
        f'</TABLE>>'
    )


def _edge_label(name: str, spec: Mapping) -> str:
    attrs = spec.get("attributes") or []
    marker = " *" if spec.get("ordered") or spec.get("named_constraints") else ""
    lines = [f'<B>{_esc(name)}</B>{marker}']
    for attr in attrs:
        lines.append(
            f'<FONT POINT-SIZE="8">{_fmt_attribute(attr)}</FONT>'
        )
    body = "<BR/>".join(lines)
    return f'<{body}>'


# ---------------------------------------------------------------------------
# DOT rendering
# ---------------------------------------------------------------------------

def render_dot(metamodel: Mapping, title: str | None = None) -> str:
    lines: list[str] = [
        "digraph metamodel {",
        '  graph [rankdir=LR, splines=spline, nodesep=0.55, ranksep=1.0,',
        '         bgcolor="white", fontname="Helvetica", pad=0.3];',
        '  node  [shape=box, style="rounded,filled", fontname="Helvetica",',
        '         margin="0.12,0.08"];',
        '  edge  [fontname="Helvetica", fontsize=9, color="black"];',
    ]
    if title:
        lines.append(f'  labelloc="t"; label=<<B>{_esc(title)}</B>>;')

    lines += ["", "  // ---------------- Node kinds ----------------"]
    for name, spec in metamodel["node_kinds"].items():
        color = NODE_COLOR.get(name, "#F5F5F5")
        style = 'rounded,filled'
        if spec.get("kind") == "abstract":
            style += ",dashed"
        label = _node_label(name, spec)
        lines.append(
            f'  {name} [label={label}, style="{style}", fillcolor="{color}"];'
        )

    lines += ["", "  // ---------------- Inheritance ----------------"]
    for name, spec in metamodel["node_kinds"].items():
        parent = spec.get("extends")
        if parent:
            lines.append(
                f'  {name} -> {parent} '
                f'[arrowhead=onormal, style=dashed, penwidth=0.8, minlen=2];'
            )

    lines += ["", "  // ---------------- Associations (edge kinds) ----------------"]
    for edge_name, edge_spec in (metamodel.get("edge_kinds") or {}).items():
        src = edge_spec["from"]
        tgt = edge_spec["to"]
        head_card = _fmt_card(edge_spec["to_cardinality"])
        tail_card = _fmt_card(edge_spec["from_cardinality"])
        label = _edge_label(edge_name, edge_spec)
        lines.append(
            f'  {src} -> {tgt} '
            f'[label={label}, headlabel="{head_card}", taillabel="{tail_card}", '
            f'labeldistance=2.4, labelangle=25, arrowhead=vee, '
            f'penwidth=1.0, color="black"];'
        )

    # Legend as a subgraph cluster at the top.
    lines += ["", "  // ---------------- Legend ----------------",
              '  subgraph cluster_legend {',
              '    label=<<B>Legend</B>>; fontname="Helvetica"; fontsize=10;',
              '    style="rounded,filled"; fillcolor="#FAFAFA"; color="#BDBDBD";',
              '    labelloc="t";',
              '    node [shape=plaintext, margin=0];',
              '    legend [label=<'
              '<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2">'
              '<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9">'
              '<B>«abstract»</B>&nbsp;dashed box &mdash; abstract kind'
              '</FONT></TD></TR>'
              '<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9">'
              '<B>«concrete»</B>&nbsp;solid box &mdash; concrete kind'
              '</FONT></TD></TR>'
              '<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9">'
              'triangle arrow &mdash; UML generalization (extends)'
              '</FONT></TD></TR>'
              '<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9">'
              'vee arrow &mdash; association, labelled with '
              '<B>name</B> and endpoint <I>cardinality</I>'
              '</FONT></TD></TR>'
              '<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9">'
              '<B>*</B> after edge name &mdash; ordered or has named constraints'
              '</FONT></TD></TR>'
              '<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9">'
              'attributes shown UML-style: <B>+</B> required, <B>-</B> optional'
              '</FONT></TD></TR>'
              '</TABLE>>];',
              '  }',
              "}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def render_summary(metamodel: Mapping) -> str:
    lines = [f"## Metamodel auto-summary", ""]
    lines.append("### Node kinds")
    lines.append("")
    lines.append("| Kind | Category | Extends | Attributes | Purpose |")
    lines.append("|------|----------|---------|------------|---------|")
    for name, spec in metamodel["node_kinds"].items():
        parent = spec.get("extends", "")
        attrs = spec.get("attributes") or []
        attr_str = ", ".join(f"`{a['name']}`" for a in attrs) or "—"
        doc = " ".join(spec.get("doc", "").split())
        lines.append(
            f"| `{name}` | {spec['kind']} | `{parent}` | {attr_str} | {doc} |"
        )
    lines.append("")

    if metamodel.get("edge_kinds"):
        lines.append("### Edge kinds")
        lines.append("")
        lines.append("| Edge | From | To | Src card. | Tgt card. | Ordered | Attributes | Constraints |")
        lines.append("|------|------|----|-----------|-----------|---------|------------|-------------|")
        for name, spec in metamodel["edge_kinds"].items():
            attrs = spec.get("attributes") or []
            attr_str = ", ".join(f"`{a['name']}`" for a in attrs) or "—"
            constraints = ", ".join(spec.get("named_constraints", [])) or "—"
            lines.append(
                f"| `{name}` | `{spec['from']}` | `{spec['to']}` | "
                f"{_fmt_card(spec['from_cardinality'])} | "
                f"{_fmt_card(spec['to_cardinality'])} | "
                f"{'yes' if spec.get('ordered') else 'no'} | {attr_str} | "
                f"{constraints} |"
            )
        lines.append("")

    if metamodel.get("global_constraints"):
        lines.append("### Global constraints")
        lines.append("")
        for name, spec in metamodel["global_constraints"].items():
            doc = " ".join(spec.get("doc", "").split())
            lines.append(f"- **`{name}`** — {doc}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build(yaml_path: Path, title: str | None = None) -> None:
    stem = yaml_path.stem                     # e.g. "ndarray_metamodel"
    dot_path = HERE / f"{stem}.dot"
    png_path = HERE / f"{stem}.png"
    svg_path = HERE / f"{stem}.svg"

    metamodel = yaml.safe_load(yaml_path.read_text())
    dot_src = render_dot(metamodel, title=title)
    dot_path.write_text(dot_src)
    print(f"wrote {dot_path}")

    subprocess.run(["dot", "-Tpng", str(dot_path), "-o", str(png_path)], check=True)
    print(f"wrote {png_path}")
    subprocess.run(["dot", "-Tsvg", str(dot_path), "-o", str(svg_path)], check=True)
    print(f"wrote {svg_path}")

    print()
    print(render_summary(metamodel))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("yaml", nargs="?", default=str(HERE / "ndarray_metamodel.yaml"),
                    help="Path to the metamodel YAML (default: ndarray_metamodel.yaml)")
    ap.add_argument("--title", default=None,
                    help="Optional title placed above the diagram")
    args = ap.parse_args()
    build(Path(args.yaml), title=args.title)


if __name__ == "__main__":
    main()
