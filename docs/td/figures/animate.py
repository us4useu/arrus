#!/usr/bin/env python3
"""Generate two animated GIFs illustrating dynamic ultrasound scheme update:

  1. Variable rebind: set_variable("sos", ...) propagating as an impulse
     from the Variable node to the TX/RX delays and to the beamformer.
  2. Structural rewrite: session.compile(g.copy_with(edits={"n_tx_rx": 2}))
     reducing an explicit 4-TX/RX sequence to 2.

Both animations use the same base graph (a "fig 3 sequence-expanded" variant of
docs/td/figures/ndarray_fig3_trigger_buffers.tex) with 4 explicit tx_rx ops
inside Region A, chained by precedence edges.

Requires: pdflatex, pdftoppm, ImageMagick's `convert`, gifsicle (optional).
Outputs:
    ndarray_anim1_variable_rebind.gif
    ndarray_anim2_structural_rewrite.gif
"""

from __future__ import annotations

import shutil
import subprocess
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Frame configuration
# ---------------------------------------------------------------------------

# Highlight kinds control the visual treatment of a node or edge on a frame.
# We keep them as plain strings and map them to TikZ style overrides in
# `_style_override`.  Adding a new kind is a matter of extending the mapping.
NODE_HIGHLIGHTS = {
    "impulse":   "line width=1.2pt, draw=arrusVariable!90!black, fill=arrusVariable!45",
    "active":    "line width=1.2pt, draw=arrusGPU!90!black,   fill=arrusGPU!45",
    "dirty":     "line width=1.2pt, draw=arrusUs4R!90!black,  fill=arrusUs4R!45",
    "removed":   "line width=1.0pt, draw=red!70!black, densely dashed, fill=red!10, "
                 "text=black!45",
    "gone":      "opacity=0.15",
}

EDGE_HIGHLIGHTS = {
    "impulse":   "very thick, draw=arrusVariable!90!black",
    "active":    "very thick, draw=arrusGPU!90!black",
    "dirty":     "very thick, draw=arrusUs4R!90!black",
    "removed":   "thick, draw=red!70!black, densely dashed",
    "gone":      "opacity=0.10",
}


@dataclass
class Frame:
    """A single animation frame."""
    caption: str = ""
    highlighted_nodes: Mapping[str, str] = field(default_factory=dict)
    highlighted_edges: Mapping[tuple[str, str], str] = field(default_factory=dict)
    hidden_nodes: frozenset[str] = frozenset()
    hidden_edges: frozenset[tuple[str, str]] = frozenset()


def _style_override(name: str, kind: str, table: Mapping[str, str]) -> str:
    style = table.get(kind, "")
    return style


# ---------------------------------------------------------------------------
# TikZ source generation
# ---------------------------------------------------------------------------

# The graph body is templated by a Frame.  Nodes and edges are declared
# uniformly with an `append style` slot that lets a Frame paint any of them
# with a highlight-kind style.  Hidden nodes get `opacity=0` so their layout
# space is preserved (avoids jitter across frames).

# List of tx_rx op ids in the acquisition sequence.
TXRX_IDS = ["tx_rx_0", "tx_rx_1", "tx_rx_2", "tx_rx_3"]


def _node_style(name: str, base: str, frame: Frame) -> str:
    parts = [base]
    if name in frame.hidden_nodes:
        parts.append("opacity=0")
    elif name in frame.highlighted_nodes:
        parts.append(_style_override(name, frame.highlighted_nodes[name],
                                     NODE_HIGHLIGHTS))
    return ", ".join(p for p in parts if p)


def _edge_style(key: tuple[str, str], base: str, frame: Frame) -> str:
    parts = [base]
    if key in frame.hidden_edges:
        parts.append("opacity=0")
    elif key in frame.highlighted_edges:
        parts.append(_style_override(key, frame.highlighted_edges[key],
                                     EDGE_HIGHLIGHTS))
    return ", ".join(p for p in parts if p)


def render_tex(frame: Frame) -> str:
    """Build the standalone TikZ document for one animation frame."""

    # ---- nodes ----
    trig_style   = _node_style("trig",   "op/trigger",   frame)
    triglbl_hide = "opacity=0" if "trig" in frame.hidden_nodes else ""

    tx_style = {i: _node_style(i, "op/us4r", frame) for i in TXRX_IDS}

    buf1_style   = _node_style("buf1",   "op/buffer",    frame)
    buf2_style   = _node_style("buf2",   "op/buffer",    frame)
    buf3_style   = _node_style("buf3",   "op/buffer",    frame)

    bf_style     = _node_style("bf",     "op/gpu",       frame)
    bmode_style  = _node_style("bmode",  "op/gpu",       frame)
    outB_style   = _node_style("outB",   "op/output",    frame)
    dop_style    = _node_style("dop",    "op/gpu",       frame)
    outD_style   = _node_style("outD",   "op/output",    frame)

    sos_style    = _node_style("sos",    "op/variable",  frame)

    # ---- edges ----
    def e(a, b, base="edge/intra"):
        return _edge_style((a, b), base, frame)

    caption = frame.caption.replace("_", r"\_")

    return textwrap.dedent(rf"""
    \documentclass[tikz,border=6pt]{{standalone}}
    \input{{ndarray_common}}
    \begin{{document}}
    \begin{{tikzpicture}}[node distance=6mm and 10mm]

      %% Caption (top of frame) -------------------------------------------------
      \node[font=\small\itshape, text=black!70, anchor=north west]
          (caption) at (0, 5.8) {{{caption}}};

      %% Region A: 4 explicit TX/RX ops (stacked) + trigger source -------------
      \node[{trig_style}]                                    (trig) at (0, 3.4) {{clk}};
      \node[below=0.3mm of trig, font=\scriptsize\itshape,
            text=arrusTrigger!80!black, {triglbl_hide}]      (triglbl) {{internal(prf)}};

      \node[{tx_style["tx_rx_0"]}, right=14mm of trig, yshift=-2mm] (tx_rx_0)
          {{tx\_rx\_0}};
      \node[{tx_style["tx_rx_1"]}, below=of tx_rx_0]         (tx_rx_1) {{tx\_rx\_1}};
      \node[{tx_style["tx_rx_2"]}, below=of tx_rx_1]         (tx_rx_2) {{tx\_rx\_2}};
      \node[{tx_style["tx_rx_3"]}, below=of tx_rx_2]         (tx_rx_3) {{tx\_rx\_3}};

      %% Buffers ---------------------------------------------------------------
      \node[{buf1_style}, right=14mm of tx_rx_1, yshift=-3mm] (buf1)
          {{device ring buffer\\\scriptsize (Us4R DDR)}};
      \node[{buf2_style}, right=of buf1] (buf2)
          {{host ring buffer\\\scriptsize (Us4R $\to$ GPU)}};

      %% Region B: per-frame GPU processing ------------------------------------
      \node[{bf_style}, right=of buf2] (bf) {{Beamforming\\\scriptsize\itshape (GPU)}};
      \node[{bmode_style}, right=of bf, yshift=14mm] (bmode) {{B-mode reconstruction}};
      \node[{outB_style}, right=of bmode] (outB) {{B-mode image}};

      %% IQ buffer + Region C ---------------------------------------------------
      \node[{buf3_style}, right=of bf, yshift=-18mm] (buf3)
          {{IQ ring buffer\\\scriptsize (GPU)}};
      \node[{dop_style}, right=of buf3] (dop) {{Color Doppler}};
      \node[{outD_style}, right=of dop] (outD) {{Doppler image}};

      %% Variable ("sos") -------------------------------------------------------
      \node[{sos_style}, above=8mm of buf1] (sos) {{Variable(``sos'')}};

      %% Trigger edge ----------------------------------------------------------
      \draw[edge/trigger, {e("trig", "tx_rx_0", "edge/trigger")}]
          (trig.east) -- (tx_rx_0.west);

      %% Precedence edges between TX/RX ops (dotted precedence) ---------------
      \draw[-{{Latex[length=1.6mm]}}, thick, dotted, draw=arrusUs4R!70!black,
            {e("tx_rx_0", "tx_rx_1", "")}]
          (tx_rx_0.south) -- (tx_rx_1.north);
      \draw[-{{Latex[length=1.6mm]}}, thick, dotted, draw=arrusUs4R!70!black,
            {e("tx_rx_1", "tx_rx_2", "")}]
          (tx_rx_1.south) -- (tx_rx_2.north);
      \draw[-{{Latex[length=1.6mm]}}, thick, dotted, draw=arrusUs4R!70!black,
            {e("tx_rx_2", "tx_rx_3", "")}]
          (tx_rx_2.south) -- (tx_rx_3.north);

      %% Data edges: each TX/RX writes into device ring buffer ----------------
      \draw[edge/stream, {e("tx_rx_0", "buf1", "edge/stream")}]
          (tx_rx_0.east) -| ([xshift=-3mm]buf1.north) -- (buf1.north);
      \draw[edge/stream, {e("tx_rx_1", "buf1", "edge/stream")}]
          (tx_rx_1.east) -- ([xshift=-3mm,yshift=2mm]buf1.west) -- (buf1.west);
      \draw[edge/stream, {e("tx_rx_2", "buf1", "edge/stream")}]
          (tx_rx_2.east) -- ([xshift=-3mm,yshift=-2mm]buf1.west) -- (buf1.west);
      \draw[edge/stream, {e("tx_rx_3", "buf1", "edge/stream")}]
          (tx_rx_3.east) -| ([xshift=-3mm]buf1.south) -- (buf1.south);

      %% Data edges: buf1 -> buf2 -> bf ---------------------------------------
      \draw[edge/stream, {e("buf1", "buf2", "edge/stream")}] (buf1.east) -- (buf2.west);
      \draw[edge/stream, {e("buf2", "bf",   "edge/stream")}] (buf2.east) -- (bf.west);

      %% Data edges within Region B (per-frame, intra-region) ----------------
      \draw[edge/intra,  {e("bf", "bmode",   "edge/intra")}]
          (bf.east) -- ++(4mm,0) |- (bmode.west);
      \draw[edge/intra,  {e("bmode", "outB", "edge/intra")}] (bmode.east) -- (outB.west);

      %% Data edges to Doppler (cross-region, N-frame batch) ------------------
      \draw[edge/stream, {e("bf", "buf3",   "edge/stream")}]
          (bf.east) -- ++(4mm,0) |- (buf3.west);
      \draw[edge/stream, {e("buf3", "dop",  "edge/stream")}] (buf3.east) -- (dop.west);
      \draw[edge/intra,  {e("dop", "outD",  "edge/intra")}]  (dop.east)  -- (outD.west);

      %% Variable ("sos") edges: fan-out to every tx_rx delays + bf sos -------
      \draw[edge/variable, {e("sos", "tx_rx_0", "edge/variable")}]
          (sos.west) -- ++(-8mm,0) |- (tx_rx_0.west);
      \draw[edge/variable, {e("sos", "tx_rx_1", "edge/variable")}]
          (sos.west) -- ++(-8mm,0) |- (tx_rx_1.west);
      \draw[edge/variable, {e("sos", "tx_rx_2", "edge/variable")}]
          (sos.west) -- ++(-8mm,0) |- (tx_rx_2.west);
      \draw[edge/variable, {e("sos", "tx_rx_3", "edge/variable")}]
          (sos.west) -- ++(-8mm,0) |- (tx_rx_3.west);
      \draw[edge/variable, {e("sos", "bf",      "edge/variable")}]
          (sos.east) -- ++(26mm,0) -- ++(0,-9mm) -| (bf.north);

    \end{{tikzpicture}}
    \end{{document}}
    """).strip()


# ---------------------------------------------------------------------------
# Build one frame: TikZ -> PDF -> PNG
# ---------------------------------------------------------------------------

def build_frame(tex_src: str, work: Path, frame_idx: int, dpi: int = 130) -> Path:
    """Compile one frame's TikZ source and return the resulting PNG path."""
    stem = f"frame_{frame_idx:03d}"
    tex_path = work / f"{stem}.tex"
    tex_path.write_text(tex_src)
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
         f"-output-directory={work}", tex_path.name],
        cwd=HERE, check=True, capture_output=True,
    )
    pdf_path = work / f"{stem}.pdf"
    png_stem = work / stem
    subprocess.run(
        ["pdftoppm", "-r", str(dpi), "-png",
         str(pdf_path), str(png_stem)],
        check=True, capture_output=True,
    )
    # pdftoppm emits <stem>-1.png for single-page PDFs.
    png_path = work / f"{stem}-1.png"
    return png_path


# ---------------------------------------------------------------------------
# Assemble frames into an animated GIF
# ---------------------------------------------------------------------------

def build_gif(frames: Iterable[Path], output: Path, frame_ms: int = 700,
              loop_pause_ms: int = 1500):
    """Combine PNG frames into a looping GIF.

    The last frame is held longer via a duplicate with a bigger delay so the
    viewer can inspect the terminal state before the loop restarts.
    """
    frames = list(frames)
    # Per-frame delay for ImageMagick's `convert` is in units of 1/100 s.
    delay = max(1, frame_ms // 10)
    last_delay = max(1, loop_pause_ms // 10)
    cmd = ["convert", "-loop", "0", "-dispose", "previous"]
    for frame in frames[:-1]:
        cmd += ["-delay", str(delay), str(frame)]
    cmd += ["-delay", str(last_delay), str(frames[-1]), str(output)]
    subprocess.run(cmd, check=True)
    if shutil.which("gifsicle"):
        subprocess.run(
            ["gifsicle", "-O3", "--colors", "128", "-o", str(output), str(output)],
            check=True,
        )


# ---------------------------------------------------------------------------
# Animation 1: Variable rebind (SoS impulse propagation)
# ---------------------------------------------------------------------------

def animation_variable_rebind() -> list[Frame]:
    """Build the frame list for the SoS-variable-rebind animation."""

    txrx = TXRX_IDS
    v_to_tx = [("sos", tid) for tid in txrx]
    v_to_bf = ("sos", "bf")

    return [
        # 0: steady state
        Frame(caption="steady state --- acquisition running, sos = 1540 m/s"),
        # 1: user calls set_variable — SoS lights up
        Frame(
            caption="session.set_variable('sos', 1500.0) --- rebind begins",
            highlighted_nodes={"sos": "impulse"},
        ),
        # 2: impulse begins to travel down every def-use edge from sos
        Frame(
            caption="impulse propagates along def-use edges (BFS dirty set)",
            highlighted_nodes={"sos": "impulse"},
            highlighted_edges={e: "impulse" for e in v_to_tx + [v_to_bf]},
        ),
        # 3: consumers become dirty
        Frame(
            caption="dirty set = { tx_rx_0..3, beamform } --- host recomputes delays",
            highlighted_nodes={
                "sos": "impulse",
                **{t: "dirty" for t in txrx},
                "bf": "active",
            },
            highlighted_edges={e: "impulse" for e in v_to_tx + [v_to_bf]},
        ),
        # 4: partial device rewrite — buf1 flashes as descriptor delays region
        # is patched (double-buffered; acquisition never pauses)
        Frame(
            caption="Us4RBackend.update(...) --- partial PCIe write of delays "
                    "(double-buffered, no acquisition pause)",
            highlighted_nodes={
                "sos": "impulse",
                **{t: "dirty" for t in txrx},
                "buf1": "dirty",
                "bf": "active",
            },
            highlighted_edges={e: "impulse" for e in v_to_tx + [v_to_bf]},
        ),
        # 5: next TX/RX cycle uses new delays
        Frame(
            caption="next TX/RX cycle uses new delays; beamformer reads sos=1500",
            highlighted_nodes={
                **{t: "active" for t in txrx},
                "bf": "active",
            },
        ),
        # 6: downstream processing sees updated frames
        Frame(
            caption="downstream regions consume updated frames (bmode, doppler)",
            highlighted_nodes={
                "bf": "active",
                "bmode": "active",
                "buf3": "active",
                "dop": "active",
            },
        ),
        # 7: back to steady state (with new sos value)
        Frame(caption="steady state --- acquisition continues, sos = 1500 m/s"),
    ]


# ---------------------------------------------------------------------------
# Animation 2: Structural rewrite (sequence reduction 4 -> 2)
# ---------------------------------------------------------------------------

def animation_structural_rewrite() -> list[Frame]:
    """Build the frame list for the copy_with sequence-reduction animation."""

    dropped_nodes = {"tx_rx_2", "tx_rx_3"}
    dropped_edges = frozenset({
        ("tx_rx_1", "tx_rx_2"),
        ("tx_rx_2", "tx_rx_3"),
        ("tx_rx_2", "buf1"),
        ("tx_rx_3", "buf1"),
        ("sos", "tx_rx_2"),
        ("sos", "tx_rx_3"),
    })

    return [
        # 0: steady state, 4 TX/RXs in sequence
        Frame(caption="steady state --- 4-TX/RX LinSequence running"),
        # 1: user issues copy_with — mark the doomed ops
        Frame(
            caption="new_g = g.copy_with(edits={'n_tx_rx': 2}) "
                    "--- structural edit begins",
            highlighted_nodes={"tx_rx_2": "removed", "tx_rx_3": "removed"},
            highlighted_edges={
                ("tx_rx_1", "tx_rx_2"): "removed",
                ("tx_rx_2", "tx_rx_3"): "removed",
                ("tx_rx_2", "buf1"): "removed",
                ("tx_rx_3", "buf1"): "removed",
                ("sos", "tx_rx_2"): "removed",
                ("sos", "tx_rx_3"): "removed",
            },
        ),
        # 2: acquisition stops for reupload (Region A dims)
        Frame(
            caption="session.compile(new_g, ...) --- old CompiledGraph.stop()",
            highlighted_nodes={
                "trig": "removed",
                **{t: "removed" for t in TXRX_IDS},
                "tx_rx_2": "removed",
                "tx_rx_3": "removed",
                "buf1": "removed",
            },
        ),
        # 3: dropped ops disappear entirely
        Frame(
            caption="new revision produced (persistent DAG; shared sub-graph "
                    "reused); recompile + re-upload in progress",
            hidden_nodes=frozenset(dropped_nodes),
            hidden_edges=dropped_edges,
            highlighted_nodes={"buf1": "dirty"},
        ),
        # 4: reupload complete, new (shorter) descriptor in device memory
        Frame(
            caption="descriptor re-uploaded --- 2-TX/RX LinSequence "
                    "(higher achievable PRF)",
            hidden_nodes=frozenset(dropped_nodes),
            hidden_edges=dropped_edges,
            highlighted_nodes={
                "tx_rx_0": "active",
                "tx_rx_1": "active",
                "buf1": "active",
            },
        ),
        # 5: acquisition restarts on new sequence
        Frame(
            caption="new_compiled.start() --- acquisition resumes",
            hidden_nodes=frozenset(dropped_nodes),
            hidden_edges=dropped_edges,
            highlighted_nodes={
                "trig": "impulse",
                "tx_rx_0": "active",
                "tx_rx_1": "active",
                "bf": "active",
            },
        ),
        # 6: downstream regions resume
        Frame(
            caption="downstream regions resume; new revision recorded in "
                    "GraphHistory",
            hidden_nodes=frozenset(dropped_nodes),
            hidden_edges=dropped_edges,
            highlighted_nodes={
                "bf": "active", "bmode": "active", "outB": "active",
                "buf3": "active", "dop": "active", "outD": "active",
            },
        ),
        # 7: new steady state
        Frame(
            caption="steady state --- 2-TX/RX LinSequence running",
            hidden_nodes=frozenset(dropped_nodes),
            hidden_edges=dropped_edges,
        ),
    ]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_animation(frames: list[Frame], name: str, work_root: Path,
                    frame_ms: int = 900):
    work = work_root / name
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    png_paths: list[Path] = []
    for idx, frame in enumerate(frames):
        tex_src = render_tex(frame)
        png_paths.append(build_frame(tex_src, work, idx))
    gif_path = HERE / f"{name}.gif"
    build_gif(png_paths, gif_path, frame_ms=frame_ms)
    # Retain the last PNG as a static preview alongside the GIF.
    shutil.copy(png_paths[-1], HERE / f"{name}_final.png")
    return gif_path


def main():
    work_root = HERE / "_anim_work"
    if work_root.exists():
        shutil.rmtree(work_root)

    gif1 = build_animation(
        animation_variable_rebind(),
        "ndarray_anim1_variable_rebind",
        work_root,
        frame_ms=900,
    )
    print(f"wrote {gif1}")

    gif2 = build_animation(
        animation_structural_rewrite(),
        "ndarray_anim2_structural_rewrite",
        work_root,
        frame_ms=1100,
    )
    print(f"wrote {gif2}")

    # Clean up intermediate build directory (comment out if you want to inspect
    # individual frames).
    shutil.rmtree(work_root)


if __name__ == "__main__":
    main()
