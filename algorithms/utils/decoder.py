from .distributions import Bernoulli, Categorical, DiagGaussian
from .decode_order import build_decode_permutation
import torch
import torch.nn as nn

class ACTLayer(nn.Module):
    """
    MLP Module to compute actions.
    """

    def __init__(self, action_space, inputs_dim, use_orthogonal, gain,
                 n_heads=4, n_layers=2, device=None, grid_size=None, args=None):
        super(ACTLayer, self).__init__()
        self.mixed_action = False
        self.multi_discrete = True
        self.action_type = action_space.__class__.__name__
        self.device = device
        self.grid_size = grid_size
        self.args = args
        self.n_dirs = 9
        # Decoder for each UAV (predict direction logits)
        self.decoder = nn.Sequential(
            nn.Linear(inputs_dim, inputs_dim),
            nn.ReLU(),
            nn.Linear(inputs_dim, self.n_dirs),
        )

        action_dims = action_space.high - action_space.low + 1
        self.action_outs = nn.ModuleList([Categorical(self.n_dirs, self.n_dirs, True, 0.01) for _ in action_dims])

    @staticmethod
    def _get_arg(args, name: str, default):
        if args is None:
            return default
        # Namespace-like
        if hasattr(args, name):
            return getattr(args, name)
        # Dict-like
        if isinstance(args, dict) and name in args:
            return args[name]
        return default

    def _prepare_w_map(self, w_map: torch.Tensor, B: int, device):
        """Return w_map as [B, grid, grid] float tensor on device."""
        if w_map is None:
            return None
        if not torch.is_tensor(w_map):
            raise TypeError("w_map must be a torch.Tensor or None")

        w = w_map.to(device)
        if w.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            w = w.float()

        if w.dim() == 3:
            # [B, I, J]
            return w
        if w.dim() == 2:
            # [B, K] with K = grid*grid
            return w.view(B, self.grid_size, self.grid_size)
        raise ValueError("w_map must have shape [B, I, J] or [B, I*J]")

    def forward(self, x, available_actions=None, deterministic=False, uav_positions=None,
                epsilon: float = 0.0, epsilon_ic: float = 0.0, w_map: torch.Tensor = None,
                beta_mix: float = None, delta0: float = None,
                decode_order: str = None):
        """
        Batched, vectorized forward. All batch samples are processed together.

        Uniform-mixture Policy Execution (optional):
        - If w_map is provided, executes the stay-gated lock policy with a uniform exploration floor.
        - If w_map is None, falls back to simple (policy, uniform) mixture using epsilon.

        Args:
            x: [B, N, d] decoder input embeddings.
            uav_positions: [B, N, 2] integer grid coordinates.
            epsilon: exploration-floor coefficient epsilon_IC.
            w_map: [B, I, J] or [B, I*J] two-sided CUSUM map (unclipped) used by the stay gate.
            beta_mix, delta0: stay gate parameters.
            decode_order: fixed index order, fresh per-batch random order, or
                descending current-cell CUSUM order.
        """
        B, N, _ = uav_positions.shape
        device = self.device
        # Uniform-mixture Policy Execution parameters
        epsilon_ic = float(epsilon_ic)
        beta_mix = float(self._get_arg(self.args, "beta_mix", 1.0) if beta_mix is None else beta_mix)
        delta0 = float(self._get_arg(self.args, "delta0", 0.0) if delta0 is None else delta0)
        w_grid = self._prepare_w_map(w_map, B, device)
        decode_order = (
            self._get_arg(self.args, "decode_order", "fixed")
            if decode_order is None
            else decode_order
        )
        permutation = build_decode_permutation(
            uav_positions,
            w_grid,
            str(decode_order),
        )

        # Step 1: Compute all logits for all UAVs in all batches at once
        logits = self.decoder(x)  # [B, N, n_dirs]

        # Step 2: Prepare candidate moves for all [B, N, n_dirs, 2]
        # [N, n_dirs, 2] -> [1, N, n_dirs, 2] -> broadcast to [B, N, n_dirs, 2]
        cur_pos = uav_positions.unsqueeze(2).expand(B, N, self.n_dirs, 2)  # [B, N, n_dirs, 2]
        directions = torch.tensor([
            [-1, -1], [-1, 0], [-1, 1],
            [0, -1], [0, 0], [0, 1],
            [1, -1], [1, 0], [1, 1]
        ], dtype=torch.long, device=device)
        directions = directions.view(1, 1, self.n_dirs, 2)
        candidates = cur_pos + directions  # [B, N, n_dirs, 2]
        candidates = candidates.clamp(0, self.grid_size - 1)

        # Step 3: Build masks to enforce collision-free moves, sequentially per UAV within each batch
        # This is still sequential in N, but batched in B!
        assigned_positions = torch.full((B, self.grid_size, self.grid_size), False, dtype=torch.bool, device=device)
        actions = torch.zeros(B, N, dtype=torch.long, device=device)
        action_log_probs = torch.zeros(B, N, device=device)
        new_positions = torch.zeros(B, N, 2, dtype=torch.long, device=device)

        batch_idx = torch.arange(B, device=device)

        for k in range(N):
            agent_idx = permutation[:, k]
            # Candidate positions for the kth-decoded UAV: [B, n_dirs, 2]
            cands = candidates[batch_idx, agent_idx]

            # Positions already assigned by previous UAVs
            mask_assigned = assigned_positions[
                batch_idx.unsqueeze(1),
                cands[..., 0].long(),
                cands[..., 1].long(),
            ]

            # Current positions of the remaining UAVs that haven't moved yet
            if k < N - 1:
                future_idx = permutation[:, k + 1 :]
                future_pos = torch.gather(
                    uav_positions,
                    1,
                    future_idx.unsqueeze(-1).expand(-1, -1, 2),
                )
                mask_future = (
                    (cands.unsqueeze(2) == future_pos.unsqueeze(1)).all(dim=-1).any(dim=-1)
                )
            else:
                mask_future = torch.zeros(B, self.n_dirs, dtype=torch.bool, device=device)

            mask = mask_assigned | mask_future

            # Mask logits
            masked_logits = logits[batch_idx, agent_idx].masked_fill(mask, -1e9)

            # Masked categorical distribution (tilde pi)
            probs = torch.softmax(masked_logits, dim=-1)
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)

            valid_mask = (~mask).float()
            uniform_probs = valid_mask / (valid_mask.sum(dim=-1, keepdim=True) + 1e-8)

            # Uniform-mixture Policy Execution: lock policy + uniform floor
            if w_grid is None:
                # print("Evaluating with simple uniform mixture policy (no w_map provided)")
                mixed_probs = (1 - epsilon) * probs + epsilon * uniform_probs
                mixed_probs = mixed_probs / (mixed_probs.sum(dim=-1, keepdim=True) + 1e-8)
            else:
                # print("Evaluating with stay-gated lock policy (w_map provided)")
                cur_k = uav_positions[batch_idx, agent_idx]
                dest_is_current = (cands == cur_k.unsqueeze(1)).all(dim=-1)  # [B, n_dirs]

                feasible = ~mask
                stay_like = feasible & dest_is_current
                nonstay = feasible & (~dest_is_current)

                # Local gap: w(cur) - max_{feasible dest != cur} w(dest)
                w_cur = w_grid[batch_idx, cur_k[:, 0].long(), cur_k[:, 1].long()]  # [B]
                w_dest = w_grid[batch_idx.unsqueeze(1), cands[..., 0].long(), cands[..., 1].long()]  # [B, n_dirs]
                w_nonstay = w_dest.masked_fill(~nonstay, float("-inf"))
                w_max = torch.max(w_nonstay, dim=-1).values  # [B]
                delta_loc = w_cur - w_max
                q = torch.sigmoid(beta_mix * (delta_loc - delta0))  # [B]

                # Distribute q over all actions whose destination is the current cell
                probs_stay = probs * stay_like.float()
                probs_nonstay = probs * nonstay.float()

                sum_stay = probs_stay.sum(dim=-1, keepdim=True) + 1e-8
                sum_nonstay = probs_nonstay.sum(dim=-1, keepdim=True) + 1e-8
                stay_norm = probs_stay / sum_stay
                nonstay_norm = probs_nonstay / sum_nonstay

                lock_probs = q.unsqueeze(-1) * stay_norm + (1.0 - q).unsqueeze(-1) * nonstay_norm
                mixed_probs = (1.0 - epsilon_ic) * lock_probs + epsilon_ic * uniform_probs
                mixed_probs = mixed_probs / (mixed_probs.sum(dim=-1, keepdim=True) + 1e-8)
                # print("mixed_probs:", mixed_probs)

            # if deterministic:
            #     action_out = mixed_probs.argmax(dim=-1)
            # else:
            action_out = torch.multinomial(mixed_probs, 1).squeeze(-1)

            actions[batch_idx, agent_idx] = action_out
            # log prob
            action_log_probs[batch_idx, agent_idx] = torch.log(
                mixed_probs.gather(1, action_out.unsqueeze(-1)).squeeze(-1)
                + 1e-8
            )
            # Save new position
            new_pos = cands[batch_idx, action_out]
            new_positions[batch_idx, agent_idx] = new_pos

            # Mark assigned positions in batch
            assigned_positions[
                batch_idx,
                new_pos[:, 0].long(),
                new_pos[:, 1].long(),
            ] = True

        return actions, action_log_probs

    def evaluate_actions(self, x, actions, available_actions=None, active_masks=None, uav_positions=None,
                         epsilon: float = 0.0, epsilon_ic: float = 0.0, w_map: torch.Tensor = None, beta_mix: float = None, delta0: float = None):
        B, N, _ = uav_positions.shape
        device = x.device

        # Uniform-mixture Policy Execution parameters
        epsilon_ic = float(epsilon_ic)
        beta_mix = float(self._get_arg(self.args, "beta_mix", 1.0) if beta_mix is None else beta_mix)
        delta0 = float(self._get_arg(self.args, "delta0", 0.0) if delta0 is None else delta0)
        w_grid = self._prepare_w_map(w_map, B, device)

        logits = self.decoder(x)  # [B, N, n_dirs]

        # [B, N, n_dirs, 2]
        cur_pos = uav_positions.unsqueeze(2).expand(B, N, self.n_dirs, 2)

        directions = torch.tensor([
            [-1, -1], [-1, 0], [-1, 1],
            [0, -1], [0, 0], [0, 1],
            [1, -1], [1, 0], [1, 1]
        ], dtype=torch.long, device=device)
        directions = directions.view(1, 1, self.n_dirs, 2)

        candidates = cur_pos + directions
        candidates = candidates.clamp(0, self.grid_size - 1)

        assigned_positions = torch.full((B, self.grid_size, self.grid_size), False, dtype=torch.bool, device=device)
        action_log_probs = torch.zeros(B, N, device=device)
        entropies = torch.zeros(B, N, device=device)

        batch_range = torch.arange(B, device=device).unsqueeze(1)
        batch_idx = torch.arange(B, device=device)

        for i in range(N):
            cands = candidates[:, i]
            mask_assigned = assigned_positions[batch_range, cands[..., 0].long(), cands[..., 1].long()]

            if i < N - 1:
                future_pos = uav_positions[:, i + 1 :, :]
                mask_future = (
                    (cands.unsqueeze(2) == future_pos.unsqueeze(1)).all(dim=-1).any(dim=-1)
                )
            else:
                mask_future = torch.zeros(B, self.n_dirs, dtype=torch.bool, device=device)

            mask = mask_assigned | mask_future
            masked_logits = logits[:, i, :].masked_fill(mask, -1e9)
            probs = torch.softmax(masked_logits, dim=-1)
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)

            valid_mask = (~mask).float()
            uniform_probs = valid_mask / (valid_mask.sum(dim=-1, keepdim=True) + 1e-8)

            # Uniform-mixture Policy Execution: lock policy + uniform floor
            if w_grid is None:
                # print("Evaluating with simple uniform mixture policy (no w_map provided)")
                mixed_probs = (1 - epsilon) * probs + epsilon * uniform_probs
                mixed_probs = mixed_probs / (mixed_probs.sum(dim=-1, keepdim=True) + 1e-8)
            else:
                # print("Evaluating with stay-gated lock policy (w_map provided)")
                cur_k = uav_positions[:, i, :]  # [B, 2]
                dest_is_current = (cands == cur_k.unsqueeze(1)).all(dim=-1)  # [B, n_dirs]

                feasible = ~mask
                stay_like = feasible & dest_is_current
                nonstay = feasible & (~dest_is_current)

                w_cur = w_grid[batch_idx, cur_k[:, 0].long(), cur_k[:, 1].long()]  # [B]
                w_dest = w_grid[batch_idx.unsqueeze(1), cands[..., 0].long(), cands[..., 1].long()]  # [B, n_dirs]
                w_nonstay = w_dest.masked_fill(~nonstay, float("-inf"))
                w_max = torch.max(w_nonstay, dim=-1).values  # [B]
                delta_loc = w_cur - w_max
                q = torch.sigmoid(beta_mix * (delta_loc - delta0))  # [B]

                probs_stay = probs * stay_like.float()
                probs_nonstay = probs * nonstay.float()

                sum_stay = probs_stay.sum(dim=-1, keepdim=True) + 1e-8
                sum_nonstay = probs_nonstay.sum(dim=-1, keepdim=True) + 1e-8
                stay_norm = probs_stay / sum_stay
                nonstay_norm = probs_nonstay / sum_nonstay

                lock_probs = q.unsqueeze(-1) * stay_norm + (1.0 - q).unsqueeze(-1) * nonstay_norm
                mixed_probs = (1.0 - epsilon_ic) * lock_probs + epsilon_ic * uniform_probs
                mixed_probs = mixed_probs / (mixed_probs.sum(dim=-1, keepdim=True) + 1e-8)

            action_idx = actions[:, i].long()
            log_prob = torch.log(mixed_probs.gather(1, action_idx.unsqueeze(-1)).squeeze(-1) + 1e-8)
            action_log_probs[:, i] = log_prob

            entropy = -torch.sum(mixed_probs * torch.log(mixed_probs + 1e-8), dim=-1)
            entropies[:, i] = entropy

            new_pos = cands[torch.arange(B), action_idx, :]
            assigned_positions[torch.arange(B), new_pos[:, 0].long(), new_pos[:, 1].long()] = True

        # Optionally mask out inactive agents
        if active_masks is not None:
            dist_entropy = (entropies * active_masks).sum() / (active_masks.sum() + 1e-8)
        else:
            dist_entropy = entropies.mean()

        return action_log_probs, dist_entropy
