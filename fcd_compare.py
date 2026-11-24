#!/usr/bin/env python3
"""Compare two ruth FCD HDF5 histories as SETS of records rather than as files.

h5diff compares row i against row i, so a pure reordering reads as millions of
differences. The distributed advance is allowed to emit a step's FCDs in a
different order than serial (the acceptance criterion is vehicle positions, not
row order), so the question h5diff cannot answer is the one that matters: are
these the same records written in a different order, or different records?

Sorting both files by every column answers it. If the sorted arrays are equal
the contents are identical and only the order differs; if they are not, the
per-column and per-vehicle breakdown says what actually moved.

    python3 fcd_compare.py fcd_history-part0000.h5 fcd_history-part0000.h5.serial
"""

import argparse
import sys

import h5py
import numpy as np


def load(path, dataset):
    with h5py.File(path, "r") as handle:
        if dataset not in handle:
            sys.exit(f"{path}: no dataset {dataset!r} (has: {list(handle.keys())})")
        return handle[dataset][:]


def canonical(array):
    """Sorted by every column, so equal multisets compare equal.

    lexsort takes the PRIMARY key last, hence the reversed field order.
    """
    keys = [array[name] for name in reversed(array.dtype.names)]
    return array[np.lexsort(keys)]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--dataset", default="fcd")
    parser.add_argument("--samples", type=int, default=5,
                        help="rows of each disagreement to print (default 5)")
    args = parser.parse_args()

    left = load(args.left, args.dataset)
    right = load(args.right, args.dataset)

    print(f"left : {len(left):>10} rows  {args.left}")
    print(f"right: {len(right):>10} rows  {args.right}")
    if left.dtype != right.dtype:
        sys.exit(f"dtypes differ:\n  {left.dtype}\n  {right.dtype}")

    if len(left) != len(right):
        print(f"\nROW COUNT DIFFERS by {len(left) - len(right):+d} -- not a reordering.")

    # How much of the difference is order alone?
    if len(left) == len(right):
        same_position = int((left == right).sum())
        print(f"\nrows identical in place : {same_position} / {len(left)} "
              f"({100.0 * same_position / max(len(left), 1):.2f}%)")

    sorted_left, sorted_right = canonical(left), canonical(right)

    if len(left) == len(right) and np.array_equal(sorted_left, sorted_right):
        print("\nSORTED CONTENTS ARE IDENTICAL -> the files hold the same records; "
              "only the row order differs.")
        return 0

    print("\nSORTED CONTENTS DIFFER -> the records themselves are not the same.")

    if len(sorted_left) == len(sorted_right):
        print("\nper-column disagreement (after sorting):")
        for name in left.dtype.names:
            differing = int((sorted_left[name] != sorted_right[name]).sum())
            print(f"  {name:<16} {differing:>10}")

    # Which vehicles are involved, by comparing each side's record multiset.
    left_ids = np.unique(left["vehicle_id"])
    right_ids = np.unique(right["vehicle_id"])
    only_left = np.setdiff1d(left_ids, right_ids)
    only_right = np.setdiff1d(right_ids, left_ids)
    if len(only_left) or len(only_right):
        print(f"\nvehicles only in left : {len(only_left)} {only_left[:10]}")
        print(f"vehicles only in right: {len(only_right)} {only_right[:10]}")

    # Row counts per vehicle: a vehicle emitting a different NUMBER of samples is
    # a much stronger signal than one whose values moved.
    common = np.intersect1d(left_ids, right_ids)
    left_counts = dict(zip(*np.unique(left["vehicle_id"], return_counts=True)))
    right_counts = dict(zip(*np.unique(right["vehicle_id"], return_counts=True)))
    mismatched = [v for v in common if left_counts[v] != right_counts[v]]
    print(f"\nvehicles with a different sample COUNT: {len(mismatched)}")
    for vehicle in mismatched[:args.samples]:
        print(f"  vehicle {vehicle}: left={left_counts[vehicle]} right={right_counts[vehicle]}")

    if len(sorted_left) == len(sorted_right):
        differing = np.nonzero(sorted_left != sorted_right)[0]
        print(f"\nfirst {min(args.samples, len(differing))} differing rows (sorted order):")
        for i in differing[:args.samples]:
            print(f"  left : {sorted_left[i]}")
            print(f"  right: {sorted_right[i]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
