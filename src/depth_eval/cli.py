"""Command line entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

from depth_eval.evaluate import run_evaluation
from depth_eval.generate import generate_dataset


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="depth-eval", description="单目深度与立体匹配统一评测")
    sub = parser.add_subparsers(dest="cmd", required=True)

    generate = sub.add_parser("generate", help="生成可复现的评测夹具")
    generate.add_argument("--out", type=Path, required=True)

    evaluate = sub.add_parser("evaluate", help="按统一协议比较单目与立体输出")
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--out", type=Path, required=True)
    evaluate.add_argument("--align", choices=["median", "least_squares"], default="median")

    args = parser.parse_args(argv)
    if args.cmd == "generate":
        paths = generate_dataset(args.out)
        print(f"wrote {len(paths)} scenes to {args.out}")
        return
    result = run_evaluation(args.data, args.out, align=args.align)
    print("\n".join(result["conclusion"]))
    print(f"report: {args.out / 'report.md'}")


if __name__ == "__main__":
    main()
