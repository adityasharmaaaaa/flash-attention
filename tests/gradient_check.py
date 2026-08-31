"""
Finite-difference gradient check.

Unlike test_backward.py (which trusts PyTorch's own autograd math), this
script checks our CUDA *forward* and *backward* binaries against each
other, using nothing but the definition of a derivative. It perturbs a
single element of Q, K, or V, reruns the actual naive_forward.cu binary,
and compares (L(x+eps) - L(x-eps)) / (2*eps) to what naive_backward.cu
computed analytically.

L is defined as L = sum(O * dO_fixed) for a fixed random dO, which makes
dL/dO = dO_fixed by construction -- exactly the upstream gradient our
backward kernels expect.
"""
import numpy as np
import subprocess
import os

FWD_BIN = "./naive_forward"
BWD_BIN = "./naive_backward"
EPS = 1e-2          # deliberately not tiny -- see note in Stage 2 docs
TOL = 2e-2          # loose tolerance appropriate for fp32 + this eps


def forward(Q, K, V, N, D):
    Q.tofile("tmp/q.bin"); K.tofile("tmp/k.bin"); V.tofile("tmp/v.bin")
    subprocess.run([FWD_BIN, str(N), str(D),
                    "tmp/q.bin", "tmp/k.bin", "tmp/v.bin", "tmp/o.bin"], check=True)
    return np.fromfile("tmp/o.bin", dtype=np.float32).reshape(N, D)


def loss(O, dO_fixed):
    return float(np.sum(O * dO_fixed))


def numerical_grad(tensor_name, Q, K, V, dO_fixed, N, D, i, j):
    tensors = {"Q": Q, "K": K, "V": V}
    t = tensors[tensor_name]

    orig = t[i, j]
    t[i, j] = orig + EPS
    L_plus = loss(forward(Q, K, V, N, D), dO_fixed)

    t[i, j] = orig - EPS
    L_minus = loss(forward(Q, K, V, N, D), dO_fixed)

    t[i, j] = orig  # restore
    return (L_plus - L_minus) / (2 * EPS)


def run_check(N=4, D=4, seed=1, n_samples=6):
    os.makedirs("tmp", exist_ok=True)
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((N, D)).astype(np.float32)
    K = rng.standard_normal((N, D)).astype(np.float32)
    V = rng.standard_normal((N, D)).astype(np.float32)
    dO_fixed = rng.standard_normal((N, D)).astype(np.float32)

    # Run the analytic backward ONCE, at the unperturbed Q,K,V, and load the
    # results into memory now -- before the perturbation loop below starts
    # overwriting tmp/q.bin etc. via forward().
    Q.tofile("tmp/q.bin"); K.tofile("tmp/k.bin"); V.tofile("tmp/v.bin")
    dO_fixed.tofile("tmp/do.bin")
    subprocess.run([BWD_BIN, str(N), str(D),
                    "tmp/q.bin", "tmp/k.bin", "tmp/v.bin", "tmp/do.bin",
                    "tmp/dq.bin", "tmp/dk.bin", "tmp/dv.bin"], check=True)

    analytic = {
        "Q": np.fromfile("tmp/dq.bin", dtype=np.float32).reshape(N, D),
        "K": np.fromfile("tmp/dk.bin", dtype=np.float32).reshape(N, D),
        "V": np.fromfile("tmp/dv.bin", dtype=np.float32).reshape(N, D),
    }

    rng2 = np.random.default_rng(seed + 100)
    all_ok = True
    for name in ["Q", "K", "V"]:
        for _ in range(n_samples):
            i = rng2.integers(0, N)
            j = rng2.integers(0, D)
            num = numerical_grad(name, Q, K, V, dO_fixed, N, D, i, j)
            ana = analytic[name][i, j]
            diff = abs(num - ana)
            ok = diff < TOL
            all_ok &= ok
            status = "OK " if ok else "FAIL"
            print(f"  [{status}] d{name}[{i},{j}]  numeric={num:+.5f}  "
                  f"analytic={ana:+.5f}  diff={diff:.2e}")

    print("ALL PASS" if all_ok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    run_check()
