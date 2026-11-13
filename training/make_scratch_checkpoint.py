"""Write a randomly initialized DispatchNet actor checkpoint for from-scratch training.

The architecture is the paper's DispatchNet (hidden size 128,
two recurrent layers, two Transformer layers) unless overridden.  The file is
consumed by ``training.train_dispatch_gpu --init-checkpoint``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces

from algorithms.actor_critic import Actor
from algorithms.utils.action_space import MultiDiscrete
from evaluation.dispatchppo_adapter import _actor_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, required=True)
    parser.add_argument("--num-uavs", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--recurrent-n", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    actor_args = _actor_args(
        {"hidden_size": args.hidden_size, "recurrent_N": args.recurrent_n},
        epsilon_ic=0.0,
    )
    obs_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(3, args.grid_size, args.grid_size), dtype=np.float32
    )
    action_space = MultiDiscrete([[0, 8]] * args.num_uavs)
    actor = Actor(actor_args, obs_space, action_space, device=torch.device("cpu"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": actor.state_dict(),
            "training": {
                "update": 0,
                "phase": "scratch",
                "setting": {"grid_size": args.grid_size, "num_uavs": args.num_uavs},
                "note": "random initialization for from-scratch training",
                "seed": args.seed,
            },
        },
        args.output,
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
