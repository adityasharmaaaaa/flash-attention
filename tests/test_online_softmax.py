import numpy as np
import torch
import subprocess
import os

BIN = "./online_softmax"

def run(S):
    rows, cols = S.shape
    os.makedirs("tmp", exist_ok=True)
    S.astype(np.float32).tofile("tmp/s.bin")
    subprocess.run([BIN, str(rows), str(cols), "tmp/s.bin", "tmp/p.bin"], check=True)
    return np.fromfile("tmp/p.bin", dtype=np.float32).reshape(rows, cols)

def check(name, S):
    P = run(S)
    ref = torch.softmax(torch.tensor(S), dim=-1).numpy()
    err = np.abs(P - ref).max()
    print(f"[{name}] max abs error vs stable softmax: {err:.3e}")
    assert err < 1e-4, f"{name} failed"
    print("  PASS")

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # 1. plain random rows, cols not a multiple of ROW_TILE (see Colab command below)
    check("random, N=37", rng.standard_normal((8, 37)).astype(np.float32) * 2.0)

    # 2. the exact scenario from the Stage 4 -> 5 handoff question:
    #    a large value arrives in a LATER tile, forcing the running max to update
    #    mid-row. If the rescale correction is wrong, this row breaks first.
    S = rng.standard_normal((1, 40)).astype(np.float32)
    S[0, 35] = 50.0   # deliberately placed near the END, well past the first few tiles
    check("late-arriving max at col 35", S)

    # 3. large value in the FIRST tile instead, for contrast -- this one would still
    #    pass even with a buggy recurrence that only handled the "easy" case
    S2 = rng.standard_normal((1, 40)).astype(np.float32)
    S2[0, 1] = 50.0
    check("early max at col 1 (control case)", S2)

    print("ALL PASS")
