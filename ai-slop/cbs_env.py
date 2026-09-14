"""Thin wrapper around the `cyberbattle` pip package (Microsoft CyberBattleSim
lineage) that makes it usable as a plain fixed-shape (obs, discrete-action)
environment for a Dreamer-style world model + REINFORCE actor-critic.

This is a genuinely SEPARATE system from the flow-traffic RSSM built for the
main SIH26153 pipeline -- CyberBattleSim's state is a symbolic graph of node
properties/credentials, not network flow statistics, and there is no way to
connect the two. It exists to demonstrate learning a world model AND a
policy inside it, on an environment that actually has action-conditioned
dynamics (unlike the passive flow-capture datasets, which have none).

Two package bugs worked around here, both Python/NumPy-version
incompatibilities in the pip release, not anything about the environment's
own logic: `np.infty` (removed in NumPy 2.0) and a `random.sample` on a
dict-keys view (only triggered by the package's OTHER sample environments,
avoided here by importing only the Chain env's module directly instead of
going through the package's `register_env()`/`gym.make()` path).
"""
from __future__ import annotations

import numpy as np

if not hasattr(np, "infty"):
    np.infty = np.inf

from cyberbattle.env.samples.CyberBattleChain import new_environment  # noqa: E402
from cyberbattle.env.env_generation.cyber_env import CyberBattleEnv  # noqa: E402

MAX_NODE_COUNT = 12
MAX_TOTAL_CREDENTIALS = 12


class _NoOpDefender:
    """The package's default scripted defender (ScanAndReimageCompromiseMachines)
    hits a genuine typo bug in this pip release (`MachineStatus.RUNNIG` vs
    the actual `RUNING`) the moment it runs. Not RL-controllable anyway (see
    module docstring) -- a no-op stands in cleanly rather than patching a
    shared site-packages file for a bug unrelated to what this wrapper needs."""

    def step(self, env, actuator, iteration_count, log):
        return log


def build_raw_env(size: int = 4) -> CyberBattleEnv:
    env_def = new_environment(size=size)
    return CyberBattleEnv(initial_environment=env_def, maximum_node_count=MAX_NODE_COUNT,
                          maximum_total_credentials=MAX_TOTAL_CREDENTIALS,
                          defender_agent=_NoOpDefender())


class ActionEnumeration:
    """Flattens the three DiscriminatedUnion action types (local_vulnerability,
    remote_vulnerability, connect) into ONE discrete index space, derived
    from the actual action_mask shapes at construction time -- so this
    always matches whatever bounds the env was actually built with."""

    def __init__(self, mask_shapes: dict[str, tuple]):
        self.shapes = mask_shapes
        self.ranges = {}
        offset = 0
        for name, shape in mask_shapes.items():
            n = int(np.prod(shape))
            self.ranges[name] = (offset, offset + n, shape)
            offset += n
        self.n_actions = offset

    def index_to_action(self, idx: int):
        for name, (lo, hi, shape) in self.ranges.items():
            if lo <= idx < hi:
                local = idx - lo
                params = np.array(np.unravel_index(local, shape))
                return {name: params}
        raise IndexError(idx)

    def mask_to_valid_indices(self, mask: dict) -> np.ndarray:
        valid = []
        for name, (lo, hi, shape) in self.ranges.items():
            flat = np.asarray(mask[name]).reshape(-1)
            valid.extend((lo + np.where(flat > 0)[0]).tolist())
        return np.array(valid, dtype=np.int64)

    def mask_vector(self, mask: dict) -> np.ndarray:
        v = np.zeros(self.n_actions, dtype=np.float32)
        valid = self.mask_to_valid_indices(mask)
        v[valid] = 1.0
        return v


def flatten_obs(obs, action_enum: ActionEnumeration,
                max_nodes: int = MAX_NODE_COUNT, max_creds: int = MAX_TOTAL_CREDENTIALS) -> np.ndarray:
    """Pads every variable-length field (grows as the attacker discovers
    more nodes/credentials over an episode) to a FIXED size, so the world
    model always sees a constant-dimension vector."""
    def pad2d(arr, rows, cols):
        arr = np.asarray(arr, dtype=np.float32).reshape(-1, cols) if np.asarray(arr).size else np.zeros((0, cols))
        out = np.full((rows, cols), -1.0, dtype=np.float32)
        out[:min(len(arr), rows)] = arr[:rows]
        return out.reshape(-1)

    def pad1d(arr, n):
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        out = np.full((n,), -1.0, dtype=np.float32)
        out[:min(len(arr), n)] = arr[:n]
        return out

    scalars = np.array([
        obs.newly_discovered_nodes_count, obs.lateral_move, obs.customer_data_found,
        obs.probe_result, obs.escalation, obs.credential_cache_length, obs.discovered_node_count,
    ], dtype=np.float32)

    prop_cols = np.asarray(obs.discovered_nodes_properties).shape[-1] if np.asarray(obs.discovered_nodes_properties).size else 1
    parts = [
        scalars,
        pad2d(obs.discovered_nodes_properties, max_nodes, prop_cols),
        pad1d(obs.nodes_privilegelevel, max_nodes),
        pad2d(obs.leaked_credentials, 4, 4),
        pad2d(obs.credentials_cache_matrix, max_creds, 2),
    ]
    return np.concatenate(parts)


class CBSWrapper:
    def __init__(self, size: int = 4, max_steps: int = 60):
        self.env = build_raw_env(size)
        self.max_steps = max_steps
        obs = self._reset_raw()
        self.action_enum = ActionEnumeration({k: np.asarray(v).shape for k, v in obs.action_mask.items()})
        self.obs_dim = flatten_obs(obs, self.action_enum).shape[0]
        self.n_actions = self.action_enum.n_actions
        self._last_raw_obs = obs
        self._steps = 0

    def _reset_raw(self):
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        return obs

    def reset(self) -> np.ndarray:
        self._steps = 0
        obs = self._reset_raw()
        self._last_raw_obs = obs
        return flatten_obs(obs, self.action_enum)

    def valid_action_mask(self) -> np.ndarray:
        return self.action_enum.mask_vector(self._last_raw_obs.action_mask)

    def step(self, action_idx: int):
        action = self.action_enum.index_to_action(int(action_idx))
        _, obs, reward, done, _ = self.env.step(action, "")
        self._last_raw_obs = obs
        self._steps += 1
        truncated = self._steps >= self.max_steps
        return flatten_obs(obs, self.action_enum), float(reward), bool(done or truncated)
