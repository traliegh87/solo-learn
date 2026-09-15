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


def find_collisions(
    df: pd.DataFrame,
    specimen_col: str,
    view_col: str,
    tray_col: str,
    path_col: str,
) -> pd.Series:
    """Flags rows whose Specimen_Id doesn't uniquely identify one image.

    Specimen_Id is expected to be unique per specimen, so (Specimen_Id, View)
    should resolve to exactly one image path, and a given Specimen_Id should
    never appear under more than one Tray_Id. Either condition failing means
    the sheet has two different images sharing a Specimen_Id -- a real data
    problem, not just a filesystem naming issue.
    """

    same_view_diff_image = df.groupby([specimen_col, view_col])[path_col].transform("nunique") > 1
    same_specimen_diff_tray = df.groupby(specimen_col)[tray_col].transform("nunique") > 1
    return same_view_diff_image | same_specimen_diff_tray


def report_collisions(df: pd.DataFrame, specimen_col: str, tray_col: str, path_col: str, split_name: str) -> None:
    for specimen_id, group in df.groupby(specimen_col):
        trays = sorted(group[tray_col].astype(str).unique())
        paths = sorted(group[path_col].astype(str).unique())
        print(f"[{split_name}] COLLISION {specimen_col}={specimen_id!r}: trays={trays} paths={paths}")


def link_rows(df: pd.DataFrame, path_col: str, dest_for_row) -> tuple:
    """Symlinks each row's source image to dest_for_row(row). Returns (linked, duplicates)."""

    linked, duplicates = 0, 0
    for row in df.itertuples(index=False):
        src = Path(getattr(row, path_col))
        dest = dest_for_row(row)
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.is_symlink():
            duplicates += 1
            continue

        os.symlink(src, dest)
        linked += 1

    return linked, duplicates


def build_split(
    df: pd.DataFrame,
    out_dir: str,
    collisions_dir: str,
    path_col: str,
    label_col: str,
    specimen_col: str,
    view_col: str,
    tray_col: str,
    split_name: str,
) -> None:
    existing_df = df[df[path_col].apply(lambda p: Path(p).is_file())]
    missing = len(df) - len(existing_df)

    collision_mask = find_collisions(existing_df, specimen_col, view_col, tray_col, path_col)
    collided_df = existing_df[collision_mask]
    clean_df = existing_df[~collision_mask]

    if len(collided_df):
        report_collisions(collided_df, specimen_col, tray_col, path_col, split_name)

        def collision_dest(row):
            suffix = Path(getattr(row, path_col)).suffix
            specimen, tray, view = getattr(row, specimen_col), getattr(row, tray_col), getattr(row, view_col)
            return Path(collisions_dir) / str(specimen) / f"{tray}_{view}{suffix}"

        linked, duplicates = link_rows(collided_df, path_col, collision_dest)
        print(
            f"[{split_name}] WARNING: {len(collided_df)} rows across "
            f"{collided_df[specimen_col].nunique()} colliding {specimen_col} values -- "
            f"linked {linked} images (skipped {duplicates} exact dups) into {collisions_dir}"
        )

    def clean_dest(row):
        suffix = Path(getattr(row, path_col)).suffix
        label, specimen, view = getattr(row, label_col), getattr(row, specimen_col), getattr(row, view_col)
        return Path(out_dir) / str(label) / f"{specimen}_{view}{suffix}"

    linked, duplicates = link_rows(clean_df, path_col, clean_dest)
    out_root = Path(out_dir)
    num_classes = sum(1 for entry in os.scandir(out_root) if entry.is_dir()) if out_root.is_dir() else 0
    print(f"[{split_name}] linked {linked} images into {num_classes} class folders under {out_root}")
    if duplicates:
        print(f"[{split_name}] skipped {duplicates} duplicate rows (same source image seen twice)")
    if missing:
        print(f"[{split_name}] skipped {missing} rows: source image not found on disk")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="path to existing_data.csv")
    parser.add_argument("--path-col", type=str, default="Id", help="column with the absolute image path")
    parser.add_argument("--label-col", type=str, default="y", help="column with the class label")
    parser.add_argument("--specimen-col", type=str, default="Specimen_Id")
    parser.add_argument("--view-col", type=str, default="View")
    parser.add_argument("--tray-col", type=str, default="Tray_Id")
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
    build_split(
        train_df,
        args.train_out,
        args.train_out.rstrip("/\\") + "_collisions",
        args.path_col,
        args.label_col,
        args.specimen_col,
        args.view_col,
        args.tray_col,
        "train",
    )

    if args.expected_num_classes is not None:
        num_classes = sum(1 for entry in os.scandir(args.train_out) if entry.is_dir())
        if num_classes != args.expected_num_classes:
            print(
                f"WARNING: expected {args.expected_num_classes} classes, "
                f"but linked {num_classes} class folders under {args.train_out}"
            )

    if args.val_split_value is not None and args.val_out is not None:
        val_df = df[df[args.split_col] == args.val_split_value]
        build_split(
            val_df,
            args.val_out,
            args.val_out.rstrip("/\\") + "_collisions",
            args.path_col,
            args.label_col,
            args.specimen_col,
            args.view_col,
            args.tray_col,
            "val",
        )
