"""
Merge two kalman_process_noise top-15 dumps into one side-by-side markdown doc.
Input: path to a .txt with blocks 'with .01' / 'with .0025' and --- division --- sections.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

Row = Tuple[int, float, str, str]  # rank, elo, fighter_id, name


def parse_section(text: str) -> Dict[str, List[Row]]:
    divisions: Dict[str, List[Row]] = {}
    current: str | None = None
    for line in text.splitlines():
        m_div = re.match(r"^---\s+(.+?)\s+---\s*$", line.strip())
        if m_div:
            current = m_div.group(1).strip()
            divisions[current] = []
            continue
        if not current:
            continue
        m_row = re.match(
            r"^\s*(\d+)\s+([\d.]+)\s+([0-9a-f]{16})\s+(.+?)\s*$",
            line,
            re.I,
        )
        if m_row:
            rnk = int(m_row.group(1))
            elo = float(m_row.group(2))
            fid = m_row.group(3)
            name = m_row.group(4).strip()
            divisions[current].append((rnk, elo, fid, name))
    return divisions


def label_division(key: str) -> str:
    if key == "catch_weight":
        return "Catch weight (mixed)"
    if key.startswith("w_"):
        return key.replace("w_", "").replace("_", " ").title() + " (women)"
    return key.replace("_", " ").title() + " (men)"


def main() -> None:
    src = Path(__file__).resolve().parent.parent / "docs" / "k sensitivity comparison top 15.txt"
    if len(sys.argv) > 1:
        src = Path(sys.argv[1])
    raw = src.read_text(encoding="utf-8")
    parts = raw.split("with .0025")
    if len(parts) != 2:
        print("Expected exactly one 'with .0025' split", file=sys.stderr)
        sys.exit(1)
    part_a = parts[0]
    part_b = "with .0025" + parts[1]
    # First block: from first 'with .01' or start to before 'with .0025'
    idx = part_a.find("with .01")
    if idx >= 0:
        part_a = part_a[idx:]
    a_text = part_a.split("Top 15", 1)[-1] if "Top 15" in part_a else part_a
    b_text = part_b.split("Top 15", 1)[-1] if "Top 15" in part_b else part_b

    div_a = parse_section(a_text)
    div_b = parse_section(b_text)

    order = [
        "flyweight",
        "bantamweight",
        "featherweight",
        "lightweight",
        "welterweight",
        "middleweight",
        "light_heavyweight",
        "heavyweight",
        "w_strawweight",
        "w_flyweight",
        "w_bantamweight",
        "w_featherweight",
        "catch_weight",
    ]

    lines: List[str] = []
    lines.append("# Kalman process noise — top 15 side by side")
    lines.append("")
    lines.append(
        "Source: merged from `docs/k sensitivity comparison top 15.txt`. "
        "**Left:** `kalman_process_noise = 0.01`. "
        "**Right:** `kalman_process_noise = 0.0025`. "
        "As-of date in source: 2026-04-19."
    )
    lines.append("")

    for key in order:
        rows_a = div_a.get(key, [])
        rows_b = div_b.get(key, [])
        if not rows_a and not rows_b:
            continue
        lines.append(f"## {label_division(key)}")
        lines.append("")
        lines.append(
            "Left three columns: `kalman_process_noise = 0.01`. "
            "Right three: `0.0025`."
        )
        lines.append("")
        lines.append("| # | Elo | Name | # | Elo | Name |")
        lines.append("|:-:|---:|---|:-:|---:|---|")
        for i in range(max(len(rows_a), len(rows_b))):
            if i < len(rows_a) and i < len(rows_b):
                rnk_a, elo_a, _, name_a = rows_a[i]
                rnk_b, elo_b, _, name_b = rows_b[i]
                lines.append(
                    f"| {rnk_a} | {elo_a:.1f} | {name_a} | {rnk_b} | {elo_b:.1f} | {name_b} |"
                )
            elif i < len(rows_a):
                rnk_a, elo_a, _, name_a = rows_a[i]
                lines.append(f"| {rnk_a} | {elo_a:.1f} | {name_a} | | | |")
            else:
                rnk_b, elo_b, _, name_b = rows_b[i]
                lines.append(f"| | | | {rnk_b} | {elo_b:.1f} | {name_b} |")
        lines.append("")

    out = src.parent / "k-sensitivity-top15-side-by-side.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
