#!/usr/bin/env python
import argparse
from pathlib import Path


def count_ply_points(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"PLY file not found: {path}")

    vertex_count = None
    with path.open("rb") as f:
        first_line = f.readline().decode("ascii", errors="replace").strip()
        if first_line != "ply":
            raise ValueError(f"Not a PLY file: {path}")

        for raw_line in f:
            line = raw_line.decode("ascii", errors="replace").strip()
            if line == "end_header":
                break

            parts = line.split()
            if len(parts) == 3 and parts[0] == "element" and parts[1] == "vertex":
                try:
                    vertex_count = int(parts[2])
                except ValueError as exc:
                    raise ValueError(f"Invalid vertex count in PLY header: {path}") from exc
        else:
            raise ValueError(f"PLY header missing end_header: {path}")

    if vertex_count is None:
        raise ValueError(f"PLY header missing 'element vertex' entry: {path}")
    return vertex_count


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print the number of points in one or more 3DGS PLY files."
    )
    parser.add_argument("ply", nargs="+", help="Path(s) to .ply file(s)")
    parser.add_argument(
        "--total",
        action="store_true",
        help="Also print the total count when multiple PLY files are provided",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    counts = [(ply, count_ply_points(ply)) for ply in args.ply]

    if len(counts) == 1:
        print(counts[0][1])
        return

    for ply, count in counts:
        print(f"{ply}\t{count}")

    if args.total:
        print(f"total\t{sum(count for _, count in counts)}")


if __name__ == "__main__":
    main()
