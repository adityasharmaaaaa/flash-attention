import numpy as np
import torch
import subprocess
import math
import os

BIN = "./naive_backward"

def run_test(N, D, seed=0):
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((N, D)).astype(np.float32)
    K = rng.standard_normal((N, D)).astype(np.float32)
    V = rng.standard_normal((N, D)).astype(np.float32)
    dO = rng.standard_normal((N, D)).astype(np.float32)

    os.makedirs("tmp", exist_ok=True)
    Q.tofile("tmp/q.bin"); K.tofile("tmp/k.bin"); V.tofile("tmp/v.bin")
    dO.tofile("tmp/do.bin")

    subprocess.run(
        [BIN, str(N), str(D),
         "tmp/q.bin", "tmp/k.bin", "tmp/v.bin", "tmp/do.bin",
         "tmp/dq.bin", "tmp/dk.bin", "tmp/dv.bin"],
        check=True,
    )

    dQ_gpu = np.fromfile("tmp/dq.bin", dtype=np.float32).reshape(N, D)
    dK_gpu = np.fromfile("tmp/dk.bin", dtype=np.float32).reshape(N, D)
    dV_gpu = np.fromfile("tmp/dv.bin", dtype=np.float32).reshape(N, D)

    Qt = torch.tensor(Q, requires_grad=True)
    Kt = torch.tensor(K, requires_grad=True)
    Vt = torch.tensor(V, requires_grad=True)
    dOt = torch.tensor(dO)

    scores = Qt @ Kt.T / math.sqrt(D)
    P = torch.softmax(scores, dim=-1)
    O = P @ Vt
    O.backward(dOt)  # feeds dO in as the upstream gradient directly

    def report(name, gpu, ref):
        err = np.abs(gpu - ref)
        print(f"  {name}: max_abs={err.max():.3e} mean_abs={err.mean():.3e}")
        assert err.max() < 1e-3, f"{name} mismatch too large"

    print(f"N={N} D={D}")
    report("dQ", dQ_gpu, Qt.grad.numpy())
    report("dK", dK_gpu, Kt.grad.numpy())
    report("dV", dV_gpu, Vt.grad.numpy())

if __name__ == "__main__":
    for N, D in [(4, 8), (37, 16), (64, 16), (128, 32)]:
        run_test(N, D)
    print("ALL PASS")
