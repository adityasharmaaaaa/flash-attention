import numpy as np
import torch
import subprocess
import math
import os

BIN = "./naive_forward"

def run_test(N, D, seed=0):
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((N, D)).astype(np.float32)
    K = rng.standard_normal((N, D)).astype(np.float32)
    V = rng.standard_normal((N, D)).astype(np.float32)

    os.makedirs("tmp", exist_ok=True)
    Q.tofile("tmp/q.bin")
    K.tofile("tmp/k.bin")
    V.tofile("tmp/v.bin")

    subprocess.run(
        [BIN, str(N), str(D), "tmp/q.bin", "tmp/k.bin", "tmp/v.bin", "tmp/o.bin"],
        check=True,
    )

    O_gpu = np.fromfile("tmp/o.bin", dtype=np.float32).reshape(N, D)

    Qt, Kt, Vt = torch.tensor(Q), torch.tensor(K), torch.tensor(V)
    scores = Qt @ Kt.T / math.sqrt(D)
    P = torch.softmax(scores, dim=-1)
    O_ref = (P @ Vt).numpy()

    abs_err = np.abs(O_gpu - O_ref)
    rel_err = abs_err / (np.abs(O_ref) + 1e-6)

    print(f"N={N:5d} D={D:4d} | max_abs={abs_err.max():.3e} "
          f"mean_abs={abs_err.mean():.3e} max_rel={rel_err.max():.3e}")
    assert abs_err.max() < 1e-3, f"Mismatch too large for N={N}, D={D}"

if __name__ == "__main__":
    for N, D in [(4, 8), (37, 16), (64, 16), (128, 32), (256, 64)]:
        run_test(N, D)
    print("ALL PASS")
