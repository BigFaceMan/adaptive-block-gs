#!/usr/bin/env python
import argparse
import re
from pathlib import Path

from count_ply_points import count_ply_points


ITERATION_RE = re.compile(r"^iteration_(\d+)$")


def iteration_from_ply(path):
    match = ITERATION_RE.match(path.parent.name)
    if not match:
        return -1
    return int(match.group(1))


def find_iteration_plys(model_root):
    point_cloud_root = model_root / "point_cloud"
    if not point_cloud_root.is_dir():
        return []
    return [
        path
        for path in point_cloud_root.glob("iteration_*/point_cloud.ply")
        if path.is_file() and iteration_from_ply(path) >= 0
    ]


def final_ply_for_experiment(exp_dir):
    search_roots = (
        ("root", exp_dir),
        ("merged", exp_dir / "merged"),
    )
    for source, model_root in search_roots:
        candidates = find_iteration_plys(model_root)
        if candidates:
            final_ply = max(candidates, key=lambda path: (iteration_from_ply(path), path.stat().st_mtime))
            return source, iteration_from_ply(final_ply), final_ply
    return None


def block_final_plys(exp_dir):
    blocks_root = exp_dir / "blocks"
    if not blocks_root.is_dir():
        return []

    rows = []
    for block_dir in sorted(path for path in blocks_root.iterdir() if path.is_dir()):
        candidates = find_iteration_plys(block_dir)
        if not candidates:
            continue
        final_ply = max(candidates, key=lambda path: (iteration_from_ply(path), path.stat().st_mtime))
        rows.append((f"block:{block_dir.name}", iteration_from_ply(final_ply), final_ply))
    return rows


def iter_rows(output_root, include_blocks=False, show_missing=False):
    output_root = Path(output_root)
    if not output_root.is_dir():
        raise FileNotFoundError(f"Output root not found: {output_root}")

    for exp_dir in sorted(path for path in output_root.iterdir() if path.is_dir()):
        result = final_ply_for_experiment(exp_dir)
        if result is None:
            if show_missing:
                yield (exp_dir.name, "missing", "", "", "")
            continue

        source, iteration, ply_path = result
        yield (exp_dir.name, source, str(iteration), str(count_ply_points(ply_path)), str(ply_path))

        if include_blocks:
            for block_source, block_iteration, block_ply in block_final_plys(exp_dir):
                yield (
                    exp_dir.name,
                    block_source,
                    str(block_iteration),
                    str(count_ply_points(block_ply)),
                    str(block_ply),
                )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count final 3DGS PLY points for each experiment under an output directory."
    )
    parser.add_argument(
        "output_root",
        nargs="?",
        default="output",
        help="Output directory containing experiment folders. Default: output",
    )
    parser.add_argument(
        "--include-blocks",
        action="store_true",
        help="Also print each block's final PLY when an experiment has blocks/",
    )
    parser.add_argument(
        "--show-missing",
        action="store_true",
        help="Print experiment directories where no root or merged final PLY is found",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Do not print the header",
    )
    parser.add_argument(
        "--tsv",
        action="store_true",
        help="Print tab-separated output instead of an aligned table",
    )
    return parser.parse_args()


def print_tsv(rows, no_header=False):
    if not no_header:
        print("experiment\tsource\titeration\tpoints\tply_path")
    for row in rows:
        print("\t".join(row))


def print_aligned(rows, no_header=False):
    headers = ("experiment", "source", "iteration", "points", "ply_path")
    table = rows if no_header else [headers, *rows]
    if not table:
        return

    widths = [0] * len(headers)
    for row in table:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    for row_idx, row in enumerate(table):
        print("  ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))
        if row_idx == 0 and not no_header:
            print("  ".join("-" * width for width in widths))


def main():
    args = parse_args()
    rows = list(iter_rows(args.output_root, args.include_blocks, args.show_missing))
    if args.tsv:
        print_tsv(rows, args.no_header)
    else:
        print_aligned(rows, args.no_header)


if __name__ == "__main__":
    main()
