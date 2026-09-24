"""Generate the persisted RD1D train/test datasets used in production.

Heat samples are synthesized on demand by :class:`datasets.diffusion.DiffusionDataset`
and therefore do not have a generation command.
"""
from __future__ import annotations

import argparse
import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="datasets/data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nproc", type=int, default=64)
    args = parser.parse_args()

    from datasets.generate_RD1d_data import run_parallel

    os.makedirs(args.root, exist_ok=True)
    run_parallel(
        root=args.root, N_ic=80, N_bc=80, nproc=args.nproc, seed=args.seed,
        filename="RD_neumann_train",
    )
    run_parallel(
        root=args.root, N_ic=16, N_bc=16, nproc=args.nproc, seed=0,
        filename="RD_neumann_test",
    )


if __name__ == "__main__":
    main()
