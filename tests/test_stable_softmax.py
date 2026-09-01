import numpy as np
import torch
import subprocess
import os

BIN = "./stable_softmax"

def run(S):
    rows, cols = S.shape
    os.makedirs("tmp", exist_ok=True)
    S.astype(np.float32).tofile("tmp/s.bin")
    subprocess.run([BIN, str(rows), str(cols), "tmp/s.bin", "tmp/out"], check=True)
    Pu = np.fromfile("tmp/out_unstable.bin", dtype=np.float32).reshape(rows, cols)
    Ps = np.fromfile("tmp/out_stable.bin", dtype=np.float32).reshape(rows, cols)
    return Pu, Ps

def test_normal_values_match_torch():
    rng = np.random.default_rng(0)
    S = rng.standard_normal((16, 32)).astype(np.float32) * 3.0
    Pu, Ps = run(S)
    ref = torch.softmax(torch.tensor(S), dim=-1).numpy()

    err = np.abs(Ps - ref).max()
    print(f"[normal values] stable max abs error vs torch: {err:.3e}")
    assert err < 1e-5
    assert np.abs(Pu - ref).max() < 1e-4, "unstable should still be fine for small values"
    print("  PASS (both agree on well-behaved inputs)")

def test_extreme_values_break_unstable():
    rng = np.random.default_rng(1)
    S = rng.standard_normal((4, 16)).astype(np.float32)
    S[0, 3] = 1000.0     # will overflow expf in the unstable kernel
    S[1, 0] = -1000.0    # will underflow -- fine either way, included for contrast
    Pu, Ps = run(S)
    ref = torch.softmax(torch.tensor(S), dim=-1).numpy()

    row0_unstable_broken = not np.all(np.isfinite(Pu[0]))
    row0_stable_ok = np.all(np.isfinite(Ps[0])) and abs(Ps[0].sum() - 1.0) < 1e-4
    row0_stable_matches_ref = np.abs(Ps[0] - ref[0]).max() < 1e-4

    print(f"[extreme values] unstable row0 finite? {not row0_unstable_broken}  "
          f"(expect False -- this IS the bug)")
    print(f"[extreme values] stable row0 finite and sums to 1? {row0_stable_ok}")
    print(f"[extreme values] stable row0 matches torch: {row0_stable_matches_ref}  "
          f"max_abs_err={np.abs(Ps[0]-ref[0]).max():.3e}")

    assert row0_unstable_broken, "expected the unstable kernel to overflow here"
    assert row0_stable_ok
    assert row0_stable_matches_ref
    print("  PASS (demonstrated the failure AND the fix)")

if __name__ == "__main__":
    test_normal_values_match_torch()
    test_extreme_values_break_unstable()
    print("ALL PASS")
