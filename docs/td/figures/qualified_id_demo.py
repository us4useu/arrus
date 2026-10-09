#!/usr/bin/env python3
"""Demonstrate derived path-like IDs over composite Ops and Devices.

Scratch module illustrating the hybrid model described in the design doc:
- composites are first-class nodes (contains_op / contains_device edges),
- qualified path-like IDs like "Us4R:0/OEM:2/Channel:5" are derived from the
  containment tree, not stored,
- lookup by path splits on "/" and traverses.

Not production code -- this is a proof of shape. The real implementation
lives in the graph library (once built). Run:

    python3 qualified_id_demo.py

to see qualified IDs and lookups for a small example graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Minimal in-memory graph representation. Composite membership is stored as a
# parent pointer on each node; roots have parent=None. This is enough to
# derive qualified IDs and answer "find by path" queries.
# ---------------------------------------------------------------------------

@dataclass
class Node:
    local_name: str                       # e.g., "OEM:2", "rxbf_lin"
    parent: Optional["Node"] = None       # composite parent, or None for root
    children: list["Node"] = field(default_factory=list)

    def add_child(self, child: "Node") -> "Node":
        child.parent = self
        self.children.append(child)
        return child

    def qualified_id(self, sep: str = "/") -> str:
        """Walk up the parent chain to build the path-like ID."""
        parts: list[str] = []
        cur: Optional[Node] = self
        while cur is not None:
            parts.append(cur.local_name)
            cur = cur.parent
        return sep.join(reversed(parts))

    def find(self, path: str, sep: str = "/") -> Optional["Node"]:
        """Look up a descendant (or self) by path relative to this node."""
        if not path:
            return self
        head, _, tail = path.partition(sep)
        for child in self.children:
            if child.local_name == head:
                return child.find(tail, sep=sep) if tail else child
        return None


class Op(Node):
    """A node in the operation graph. May be a leaf op or a composite."""
    def is_composite(self) -> bool:
        return bool(self.children)


class Device(Node):
    """A placement target. May be a leaf device or a composite."""
    def is_composite(self) -> bool:
        return bool(self.children)


# ---------------------------------------------------------------------------
# Demo: a nested Us4R device tree and a nested B-mode processing composite.
# ---------------------------------------------------------------------------

def build_example():
    # --- Device tree ---------------------------------------------------------
    # Us4R:0 contains 4 OEMs, each containing 32 channels. Only leaves
    # (channels) have concrete addresses; composites (OEMs, Us4R) are first-
    # class nodes with their own identity.
    us4r0 = Device("Us4R:0")
    for oem in range(4):
        oem_node = us4r0.add_child(Device(f"OEM:{oem}"))
        for ch in range(32):
            oem_node.add_child(Device(f"Channel:{ch}"))

    # A separate GPU. Leaf; no children.
    gpu0 = Device("GPU:0")

    # --- Op tree -------------------------------------------------------------
    # A "bmode_pipeline" composite Op exposing an interface to the outer
    # graph. Inside: a chain of processing steps. In a real graph the leaves
    # would be operand/result-connected via Value edges (not modelled here --
    # this demo focuses on containment / qualified IDs).
    imaging = Op("imaging_pipeline")
    bmode   = imaging.add_child(Op("bmode"))
    for step in ["remap", "transpose", "bandpass", "iq_demod",
                 "decimate", "rxbf_lin", "envelope", "log_compress"]:
        bmode.add_child(Op(step))
    doppler = imaging.add_child(Op("color_doppler"))
    for step in ["wall_filter", "autocorr", "velocity", "power",
                 "threshold", "colormap"]:
        doppler.add_child(Op(step))

    return us4r0, gpu0, imaging


def _print(label: str, value):
    print(f"  {label:<38} -> {value}")


def main():
    us4r0, gpu0, imaging = build_example()

    print("Qualified IDs (derived by walking the parent chain):")
    _print("us4r0.qualified_id()", us4r0.qualified_id())
    ch5 = us4r0.children[2].children[5]
    _print("Us4R:0 / OEM:2 / Channel:5", ch5.qualified_id())
    _print("gpu0.qualified_id()", gpu0.qualified_id())

    rxbf = imaging.find("bmode/rxbf_lin")
    _print("imaging.find('bmode/rxbf_lin')", rxbf.qualified_id())
    wall = imaging.find("color_doppler/wall_filter")
    _print("imaging.find('color_doppler/wall_filter')", wall.qualified_id())
    print()

    print("Path lookup:")
    hit = us4r0.find("OEM:2/Channel:5")
    _print("us4r0.find('OEM:2/Channel:5')",
           f"{hit.qualified_id()}  (found: {hit is ch5})")
    miss = us4r0.find("OEM:99/Channel:0")
    _print("us4r0.find('OEM:99/Channel:0')", f"{miss}  (missing -> None)")
    print()

    print("Composite check + interface reasoning:")
    _print("us4r0.is_composite()",   us4r0.is_composite())
    _print("ch5.is_composite()",     ch5.is_composite())
    _print("imaging.is_composite()", imaging.is_composite())
    _print("rxbf.is_composite()",    rxbf.is_composite())
    print()

    print("Renaming a composite renames all descendants' qualified IDs")
    print("(the composite has one identity; children reference it by node):")
    us4r0.local_name = "Us4R:5"
    _print("after us4r0.local_name = 'Us4R:5', ch5 is now",
           ch5.qualified_id())


if __name__ == "__main__":
    main()
