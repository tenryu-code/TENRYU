#!/usr/bin/env python3
"""Compare TENRYU owned-slab dumps across rank counts (MPI M18 gate tool).

Usage:
  compare_owned_dump.py <ref_prefix> <test_prefix> [--rtol 0.0]

<prefix>_r<rank>.bin files are produced by TENRYU_MPI_DUMP_OWNED=<prefix>.
The reference is normally a P=1 run (single file _r0.bin holding the full
arrays); each test rank's owned slice is compared against the same flat
index range of the reference field. rtol 0.0 = bitwise.

Field order (must match driver.cpp dump block): rho, ee, ei, Te, Ti, mass,
vol, x_r, v_r, x_z, v_z, rad_E. Cell fields slice [cell_begin, cell_end);
node fields [node_begin, node_end); rad_E is cell-major group-minor so its
slice is the cell window scaled by n_groups (1D verified; re-verify layout
before trusting rad_E for 2D).
"""

import glob
import struct
import sys

import numpy as np

FIELDS = [
    ("rho", "cell"), ("ee", "cell"), ("ei", "cell"), ("Te", "cell"),
    ("Ti", "cell"), ("mass", "cell"), ("vol", "cell"),
    ("x_r", "node"), ("v_r", "node"), ("x_z", "node"), ("v_z", "node"),
    ("rad_E", "rad"),
]


def read_dump(path):
    with open(path, "rb") as fh:
        header = struct.unpack("<6i", fh.read(24))
        rank, n_ranks, cb, ce, nb, ne = header
        fields = []
        for _name, _kind in FIELDS:
            (count,) = struct.unpack("<i", fh.read(4))
            data = np.frombuffer(fh.read(8 * count), dtype="<f8") if count else np.array([])
            fields.append(data)
    return {"rank": rank, "n_ranks": n_ranks, "cell": (cb, ce),
            "node": (nb, ne), "fields": fields}


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    ref_prefix, test_prefix = sys.argv[1], sys.argv[2]
    rtol = 0.0
    if "--rtol" in sys.argv:
        rtol = float(sys.argv[sys.argv.index("--rtol") + 1])

    ref = read_dump(f"{ref_prefix}_r0.bin")
    assert ref["n_ranks"] == 1, "reference dump must be a P=1 run"

    test_paths = sorted(glob.glob(f"{test_prefix}_r*.bin"))
    assert test_paths, f"no test dumps match {test_prefix}_r*.bin"

    n_cells_ref = ref["cell"][1] - ref["cell"][0]
    worst = {}
    failures = 0
    for path in test_paths:
        t = read_dump(path)
        cb, ce = t["cell"]
        nb, ne = t["node"]
        for fi, (name, kind) in enumerate(FIELDS):
            ref_arr = ref["fields"][fi]
            test_arr = t["fields"][fi]
            if len(test_arr) == 0:
                continue
            if len(ref_arr) == 0:
                print(f"FAIL {path} {name}: test has data, reference empty")
                failures += 1
                continue
            if kind == "cell":
                lo, hi = cb, ce
            elif kind == "node":
                lo, hi = nb, ne
            else:  # rad: cell window scaled by groups
                groups = len(ref_arr) // max(n_cells_ref, 1)
                lo, hi = cb * groups, ce * groups
            ref_slice = ref_arr[lo:hi]
            if len(ref_slice) != len(test_arr):
                print(f"FAIL {path} {name}: length {len(test_arr)} vs ref {len(ref_slice)}")
                failures += 1
                continue
            if rtol == 0.0:
                mismatch = np.count_nonzero(
                    ref_slice.view(np.uint64) != test_arr.view(np.uint64))
                if mismatch:
                    idx = np.nonzero(ref_slice.view(np.uint64) != test_arr.view(np.uint64))[0]
                    k = idx[0]
                    print(f"FAIL {path} {name}: {mismatch}/{len(test_arr)} bitwise "
                          f"mismatches, first at local {k} (global {lo + k}): "
                          f"ref={ref_slice[k]!r} test={test_arr[k]!r}")
                    failures += 1
                continue
            # Field-max-normalized deviation: near-zero entries of a field
            # with O(1) dynamic range must not dominate the relative metric.
            scale = max(float(np.max(np.abs(ref_arr))) if len(ref_arr) else 0.0, 1e-300)
            rel = float(np.max(np.abs(test_arr - ref_slice))) / scale if len(ref_slice) else 0.0
            worst[name] = max(worst.get(name, 0.0), rel)
            if rel > rtol:
                print(f"FAIL {path} {name}: max rel diff {rel:.3e} > rtol {rtol:.1e}")
                failures += 1

    if rtol > 0.0:
        for name, rel in sorted(worst.items()):
            print(f"  {name}: max rel diff {rel:.3e}")
    if failures == 0:
        print(f"PASS: {len(test_paths)} rank dump(s) match reference "
              f"({'bitwise' if rtol == 0.0 else f'rtol {rtol:g}'})")
        return 0
    print(f"FAILURES: {failures}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
