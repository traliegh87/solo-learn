# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse
import os
from pathlib import Path

import pandas as pd


def build_class_folders(
    df: pd.DataFrame,
    out_dir: str,
    path_col: str,
    label_col: str,
    split_name: str,
) -> None:
    """Symlinks each image referenced in df into out_dir/<label>/<basename>.

    Args:
        df (pd.DataFrame): rows for a single split.
        out_dir (str): destination root (e.g. ./datasets/vtcv_pretrain/train).
        path_col (str): column holding the absolute image path.
        label_col (str): column holding the class label.
        split_name (str): human-readable split name, used only for logging.
    """

    out_root = Path(out_dir)
    linked, missing = 0, 0

    for row in df.itertuples(index=False):
        src = Path(getattr(row, path_col))
        if not src.is_file():
            missing += 1
            continue

        label = str(getattr(row, label_col))
        dest_dir = out_root / label
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest = dest_dir / src.name
        if dest.exists() or dest.is_symlink():
            continue

        os.symlink(src, dest)
        linked += 1

    num_classes = sum(1 for entry in os.scandir(out_root) if entry.is_dir()) if out_root.is_dir() else 0
    print(f"[{split_name}] linked {linked} images into {num_classes} class folders under {out_root}")
    if missing:
        print(f"[{split_name}] skipped {missing} rows: source image not found on disk")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="path to existing_data.csv")
    parser.add_argument("--path-col", type=str, default="Id", help="column with the absolute image path")
    parser.add_argument("--label-col", type=str, default="y", help="column with the class label")
    parser.add_argument("--split-col", type=str, default="Split", help="column with the train/val split")
    parser.add_argument("--train-split-value", type=str, default="Train")
    parser.add_argument("--val-split-value", type=str, default=None, help="e.g. Val; omit to skip val-out")
    parser.add_argument("--train-out", type=str, required=True)
    parser.add_argument("--val-out", type=str, default=None)
    parser.add_argument(
        "--expected-num-classes",
        type=int,
        default=None,
        help="if set, warn when the linked train set doesn't match this class count",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    train_df = df[df[args.split_col] == args.train_split_value]
    build_class_folders(train_df, args.train_out, args.path_col, args.label_col, "train")

    if args.expected_num_classes is not None:
        num_classes = sum(1 for entry in os.scandir(args.train_out) if entry.is_dir())
        if num_classes != args.expected_num_classes:
            print(
                f"WARNING: expected {args.expected_num_classes} classes, "
                f"but linked {num_classes} class folders under {args.train_out}"
            )

    if args.val_split_value is not None and args.val_out is not None:
        val_df = df[df[args.split_col] == args.val_split_value]
        build_class_folders(val_df, args.val_out, args.path_col, args.label_col, "val")
