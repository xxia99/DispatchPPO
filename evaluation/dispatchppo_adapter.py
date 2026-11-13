"""Checkpoint adapter for batched DispatchPPO execution."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from algorithms.actor_critic import Actor
from algorithms.utils.action_space import MultiDiscrete
from gymnasium import spaces

from algorithms.utils.decode_order import (
    DECODE_ORDERS,
    build_decode_permutation,
)
from evaluation.gpu_env import GpuUAVCentralEnv


def _unwrap_state_dict(obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ("state_dict", "actor_state_dict", "model"):
            value = obj.get(key)
            if isinstance(value, dict):
                obj = value
                break
    if not isinstance(obj, dict):
        raise TypeError("checkpoint does not contain a state dictionary")
    result = {}
    for key, value in obj.items():
        if not torch.is_tensor(value):
            continue
        for prefix in ("module.", "actor."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        result[key] = value
    return result


def infer_actor_metadata(state: dict[str, torch.Tensor]) -> dict[str, int]:
    conv_key = "base.cnn.cnn.4.weight"
    if conv_key not in state:
        raise ValueError(f"checkpoint is not a centralized DispatchNet actor: missing {conv_key}")
    hidden = int(state[conv_key].shape[0])
    recurrent_layers = sum(
        key.startswith("rnn.rnn.weight_ih_l") for key in state
    )
    transformer_layers = max(
        [
            int(key.split(".")[4]) + 1
            for key in state
            if key.startswith("encoder.encoder.transformer.layers.")
        ]
        or [2]
    )
    return {
        "hidden_size": hidden,
        "recurrent_N": max(recurrent_layers, 1),
        "transformer_layers": transformer_layers,
    }


def _actor_args(metadata: dict[str, int], epsilon_ic: float) -> Namespace:
    return Namespace(
        hidden_size=metadata["hidden_size"],
        gain=0.01,
        use_orthogonal=True,
        use_policy_active_masks=False,
        use_naive_recurrent_policy=True,
        use_recurrent_policy=False,
        recurrent_N=metadata["recurrent_N"],
        explore_eps=0.0,
        epsilon_ic=float(epsilon_ic),
        use_gnn=False,
        use_ReLU=True,
        beta_mix=1.0,
        delta0=0.0,
    )


class DispatchPPOAdapter:
    """Load one centralized actor and execute Stage-II collision-safe actions."""

    def __init__(
        self,
        checkpoint: str | Path,
        grid_size: int,
        num_uavs: int,
        device: torch.device,
        epsilon_ic: float = 0.01,
        beta_mix: float = 1.0,
        delta0: float = 0.0,
        stochastic: bool = True,
        decode_order: str = "fixed",
    ):
        self.path = Path(checkpoint)
        self.device = device
        self.G = int(grid_size)
        self.N = int(num_uavs)
        self.epsilon_ic = float(epsilon_ic)
        self.beta_mix = float(beta_mix)
        self.delta0 = float(delta0)
        self.stochastic = bool(stochastic)
        if decode_order not in DECODE_ORDERS:
            raise ValueError(
                f"decode_order must be one of {DECODE_ORDERS}, "
                f"got {decode_order!r}"
            )
        self.decode_order = str(decode_order)

        payload = torch.load(self.path, map_location="cpu")
        state = _unwrap_state_dict(payload)
        metadata = infer_actor_metadata(state)
        args = _actor_args(metadata, epsilon_ic)
        self.obs_channels = int(state["base.cnn.cnn.0.weight"].shape[1])
        if self.obs_channels != 3:
            raise ValueError(
                f"checkpoint expects {self.obs_channels} observation channels; "
                "DispatchNet uses the three channels [CUSUM, uncertainty, occupancy]"
            )
        obs_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_channels, self.G, self.G),
            dtype=np.float32,
        )
        action_space = MultiDiscrete([[0, 8]] * self.N)
        self.actor = Actor(args, obs_space, action_space, device=device)
        # The per-UAV heads ``act.action_outs.*`` are constructed per fleet size but never used by the
        # shared decoder, so a checkpoint transfers across fleet sizes: drop them from the loaded
        # state and from the compatibility check.
        state = {k: v for k, v in state.items() if ".action_outs." not in k}
        missing, unexpected = self.actor.load_state_dict(state, strict=False)
        missing = [key for key in missing if ".action_outs." not in key]
        if missing or unexpected:
            raise RuntimeError(
                f"incompatible DispatchPPO checkpoint {self.path}: "
                f"missing={missing}, unexpected={unexpected}"
            )
        self.actor.eval()
        self.hidden_size = metadata["hidden_size"]
        self.recurrent_N = metadata["recurrent_N"]
        self.rnn_state: torch.Tensor | None = None
        self.previous_statistic: torch.Tensor | None = None
        self.statistic_increment: torch.Tensor | None = None
        self.last_decode_permutation: torch.Tensor | None = None

    def reset(self, num_envs: int) -> None:
        self.rnn_state = torch.zeros(
            (num_envs, self.recurrent_N, self.hidden_size),
            dtype=torch.float32,
            device=self.device,
        )
        self.previous_statistic = torch.zeros(
            (num_envs, self.G * self.G),
            dtype=torch.float32,
            device=self.device,
        )
        self.statistic_increment = torch.zeros_like(
            self.previous_statistic
        )
        self.last_decode_permutation = None

    def _actor_logits(
        self,
        actor: Actor,
        rnn_state: torch.Tensor,
        obs: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = actor.base(obs)
        if actor._use_naive_recurrent_policy or actor._use_recurrent_policy:
            masks = torch.ones(
                (obs.shape[0], 1),
                dtype=torch.float32,
                device=self.device,
            )
            _, rnn_state = actor.rnn(features, rnn_state, masks)
        encoded = actor.encoder(features, positions)
        return actor.act.decoder(encoded), rnn_state

    @torch.no_grad()
    def act(self, env: GpuUAVCentralEnv) -> torch.Tensor:
        if self.rnn_state is None or self.rnn_state.shape[0] != env.E:
            self.reset(env.E)
        assert self.rnn_state is not None
        assert self.previous_statistic is not None
        assert self.statistic_increment is not None
        torch.sub(
            env.S,
            self.previous_statistic,
            out=self.statistic_increment,
        )
        self.previous_statistic.copy_(env.S)
        obs = env.build_dispatch_obs()
        positions = env.position_xy()
        logits, self.rnn_state = self._actor_logits(
            self.actor,
            self.rnn_state,
            obs,
            positions,
        )
        return self._stage_two_actions(env, logits)

    def _stage_two_actions(
        self,
        env: GpuUAVCentralEnv,
        logits: torch.Tensor,
        decode_order: str | None = None,
    ) -> torch.Tensor:
        candidates, in_bounds = env.candidate_cells()
        actions = torch.empty((env.E, env.N), dtype=torch.long, device=env.device)
        assigned = torch.zeros((env.E, env.K), dtype=torch.bool, device=env.device)
        batch = torch.arange(env.E, device=env.device)
        permutation = build_decode_permutation(
            env.position_xy(),
            env.S.view(env.E, env.G, env.G),
            self.decode_order if decode_order is None else decode_order,
        )
        self.last_decode_permutation = permutation

        for k in range(env.N):
            agent_idx = permutation[:, k]
            cands = candidates[batch, agent_idx]
            bounds = in_bounds[batch, agent_idx]
            # Moves off the grid are infeasible.
            invalid = ~bounds
            invalid |= torch.gather(assigned, 1, cands)
            if k < env.N - 1:
                future_idx = permutation[:, k + 1 :]
                future = torch.gather(env.pos, 1, future_idx)
                invalid |= (
                    cands.unsqueeze(2) == future.unsqueeze(1)
                ).any(dim=2)
            valid = ~invalid
            adjusted_logits = logits[batch, agent_idx]
            policy = torch.softmax(
                adjusted_logits.masked_fill(invalid, -torch.inf),
                dim=1,
            )
            uniform = valid.float() / valid.float().sum(dim=1, keepdim=True).clamp_min(1.0)

            current = env.pos[batch, agent_idx]
            own = cands == current.unsqueeze(1)
            stay = valid & own
            nonstay = valid & ~own
            w_cur = env.S[batch, current]
            w_dest = env.S[batch.unsqueeze(1), cands]
            max_nonstay = w_dest.masked_fill(~nonstay, -torch.inf).max(
                dim=1
            ).values
            gap = w_cur - max_nonstay
            q = torch.sigmoid(self.beta_mix * (gap - self.delta0))

            stay_policy = policy * stay.float()
            stay_policy = stay_policy / stay_policy.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0e-12)
            move_policy = policy * nonstay.float()
            move_policy = move_policy / move_policy.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0e-12)
            locked = (
                q.unsqueeze(1) * stay_policy
                + (1.0 - q).unsqueeze(1) * move_policy
            )
            mixed = (
                (1.0 - self.epsilon_ic) * locked
                + self.epsilon_ic * uniform
            )
            mixed = mixed * valid.float()
            mixed = mixed / mixed.sum(dim=1, keepdim=True).clamp_min(1.0e-12)

            if self.stochastic:
                action_i = torch.multinomial(mixed, 1).squeeze(1)
            else:
                action_i = mixed.argmax(dim=1)
            cell_i = cands[batch, action_i]
            actions[batch, agent_idx] = action_i
            assigned[batch, cell_i] = True
        return actions.unsqueeze(-1)


def masked_dispatch_actions(
    env: GpuUAVCentralEnv,
    logits: torch.Tensor,
    *,
    actions: torch.Tensor | None = None,
    deterministic: bool = False,
    epsilon_ic: float = 0.0,
    beta_mix: float = 1.0,
    delta0: float = 0.0,
    decode_order: str = "fixed",
    probs_out: list | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample or score a collision-feasible autoregressive joint action.

    Used by the trainer: returns ``(actions [E,N,1], log_probs [E,N],
    entropies [E,N])`` of the executed policy (stay bias and epsilon_IC mixture).
    ``probs_out``: when a list is given, the per-slot execution probabilities
    [E, 9] are appended in decoding order."""

    if logits.shape != (env.E, env.N, 9):
        raise ValueError(
            f"logits must have shape {(env.E, env.N, 9)}, got {tuple(logits.shape)}"
        )
    if actions is not None:
        actions = actions.squeeze(-1).long()
        if actions.shape != (env.E, env.N):
            raise ValueError(
                f"actions must have shape {(env.E, env.N)}, got {tuple(actions.shape)}"
            )

    candidates, in_bounds = env.candidate_cells()
    chosen = torch.empty((env.E, env.N), dtype=torch.long, device=env.device)
    log_probs = torch.empty((env.E, env.N), dtype=torch.float32, device=env.device)
    entropies = torch.empty_like(log_probs)
    assigned = torch.zeros((env.E, env.K), dtype=torch.bool, device=env.device)
    batch = torch.arange(env.E, device=env.device)
    decode_permutation = build_decode_permutation(
        env.position_xy(),
        env.S.view(env.E, env.G, env.G),
        decode_order,
    )

    for decode_index in range(env.N):
        agent = decode_permutation[:, decode_index]
        agent_candidates = candidates[batch, agent]
        agent_in_bounds = in_bounds[batch, agent]
        invalid = ~agent_in_bounds
        invalid |= torch.gather(assigned, 1, agent_candidates)
        if decode_index < env.N - 1:
            future_agent = decode_permutation[:, decode_index + 1 :]
            future = torch.gather(env.pos, 1, future_agent)
            invalid |= (
                agent_candidates.unsqueeze(2) == future.unsqueeze(1)
            ).any(dim=2)
        valid = ~invalid
        if not bool(valid.any(dim=1).all().item()):
            raise RuntimeError("DispatchNet action mask has an empty feasible set")

        masked_logits = logits[batch, agent].masked_fill(invalid, -1.0e9)
        policy_probs = torch.softmax(masked_logits, dim=1)
        current = env.pos[batch, agent]
        own = agent_candidates == current.unsqueeze(1)
        stay = valid & own
        nonstay = valid & ~own
        stat_current = env.S[batch, current]
        stat_destination = env.S[
            batch.unsqueeze(1), agent_candidates
        ]
        max_nonstay = stat_destination.masked_fill(
            ~nonstay, -torch.inf
        ).max(dim=1).values
        gap = stat_current - max_nonstay
        stay_mass = torch.sigmoid(
            float(beta_mix) * (gap - float(delta0))
        )
        # The stay set is exactly the current cell, so the renormalized
        # stay law is one-hot; the move law is the policy renormalized over
        # the non-stay actions, computed in logit space so that a policy
        # that puts almost no mass on the moves does not produce a ratio
        # of two tiny numbers (whose gradient explodes).
        stay_policy = stay.float()
        move_logits = masked_logits.masked_fill(~nonstay, -1.0e9)
        has_move = nonstay.any(dim=1, keepdim=True)
        move_policy = torch.where(
            has_move,
            torch.softmax(move_logits, dim=1),
            torch.zeros_like(policy_probs),
        ) * nonstay.float()
        locked = (
            stay_mass.unsqueeze(1) * stay_policy
            + (1.0 - stay_mass).unsqueeze(1) * move_policy
        )
        uniform = valid.float() / valid.float().sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        execution_probs = (
            (1.0 - float(epsilon_ic)) * locked
            + float(epsilon_ic) * uniform
        )
        execution_probs = execution_probs * valid.float()
        execution_probs = execution_probs / execution_probs.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0e-12)

        distribution = torch.distributions.Categorical(
            probs=execution_probs
        )
        if actions is None:
            action = (
                distribution.probs.argmax(dim=1)
                if deterministic
                else distribution.sample()
            )
        else:
            action = actions[batch, agent]
            target_valid = valid[batch, action]
            if not bool(target_valid.all().item()):
                raise ValueError("target actions violate the DispatchNet feasibility mask")

        cell = agent_candidates[batch, action]
        chosen[batch, agent] = action
        log_probs[batch, agent] = distribution.log_prob(action)
        entropies[batch, agent] = distribution.entropy()
        assigned[batch, cell] = True
        if probs_out is not None:
            probs_out.append(execution_probs)

    return chosen.unsqueeze(-1), log_probs, entropies
