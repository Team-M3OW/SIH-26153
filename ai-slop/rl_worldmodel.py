"""Dreamer-style world model + REINFORCE actor-critic for CyberBattleSim
(via cbs_env.py). A genuinely separate system from the main SIH26153 flow-
traffic RSSM -- see cbs_env.py's docstring for why they can't be connected.

Unlike the passive flow-capture datasets, this environment has real
action-conditioned dynamics (you can actually try an action and observe
its effect), which is exactly what a world model needs to be useful for
learning a policy. Recipe:

  1. Collect random-valid-action rollouts from the real environment.
  2. Train an action-conditioned RSSM (same free-bits KL as wm/rssm.py) on
     those rollouts: reconstruct the next observation, predict reward,
     predict episode-continuation.
  3. Train an actor (categorical over the ~1400-5500 discrete actions) +
     critic entirely inside the learned world model's IMAGINED rollouts --
     REINFORCE with a value baseline. The world model ALSO learns a
     mask_head predicting, from its own latent state, which actions are
     valid (supervised by the real env.valid_action_mask() collected
     alongside every rollout step). That predicted mask is applied to the
     actor's logits during imagination too, so training-time action
     selection is consistent with real-environment evaluation, which masks
     to the TRUE valid set. Previously imagination used the raw unmasked
     logits (the world model was trained only on valid transitions, so it
     had no way to represent invalid-action outcomes at all) -- this let
     the actor learn to chase whatever the model over-predicted for
     actions it never saw, which the real env's masking then blocked at
     eval time. Measured effect of this exact mismatch: at size=10
     (12 nodes, 5532 actions), the pre-fix policy scored WORSE than random
     in the real environment (383 vs 480 mean episode reward) despite
     imagined return climbing throughout training -- a textbook world-
     model-exploitation failure, not a scale problem.
  4. Evaluate the trained actor in the REAL environment (masked to valid
     actions), against a random-policy baseline.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "wm"))
from rssm import free_bits_kl  # noqa: E402
from cbs_env import CBSWrapper  # noqa: E402

CKPT_DIR = "/media/kavinder/hdd2/sih26153-processed/checkpoints"


# ------------------------------------------------------------- world model ---

class CBSWorldModel(nn.Module):
    def __init__(self, obs_dim, n_actions, d_h=64, d_z=32, d_embed=64, d_action=32, d_hidden=128):
        super().__init__()
        self.d_h, self.d_z = d_h, d_z
        self.encoder = nn.Sequential(nn.Linear(obs_dim, d_hidden), nn.LayerNorm(d_hidden),
                                     nn.GELU(), nn.Linear(d_hidden, d_embed))
        self.action_embed = nn.Embedding(n_actions, d_action)
        self.gru = nn.GRUCell(d_z + d_action, d_h)
        self.prior_net = nn.Sequential(nn.Linear(d_h, d_hidden), nn.GELU(), nn.Linear(d_hidden, 2 * d_z))
        self.posterior_net = nn.Sequential(nn.Linear(d_h + d_embed, d_hidden), nn.GELU(),
                                           nn.Linear(d_hidden, 2 * d_z))
        self.decoder = nn.Sequential(nn.Linear(d_h + d_z, d_hidden), nn.GELU(), nn.Linear(d_hidden, obs_dim))
        self.reward_head = nn.Sequential(nn.Linear(d_h + d_z, d_hidden), nn.GELU(), nn.Linear(d_hidden, 1))
        self.continue_head = nn.Sequential(nn.Linear(d_h + d_z, d_hidden), nn.GELU(), nn.Linear(d_hidden, 1))
        self.mask_head = nn.Sequential(nn.Linear(d_h + d_z, d_hidden), nn.GELU(), nn.Linear(d_hidden, n_actions))
        self.obs_logvar = nn.Parameter(torch.zeros(obs_dim))

    def _split(self, out):
        mu, logvar = out.chunk(2, dim=-1)
        return mu, logvar.clamp(-8.0, 4.0)

    @staticmethod
    def rsample(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, obs_seq, action_seq):
        """Teacher-forced. obs_seq: (B,T,obs_dim). action_seq: (B,T) int64,
        action_seq[:,t] is the action taken AFTER observing obs_seq[:,t]."""
        b, t, _ = obs_seq.shape
        device = obs_seq.device
        e = self.encoder(obs_seq)
        h = torch.zeros(b, self.d_h, device=device)
        z = torch.zeros(b, self.d_z, device=device)

        states, post_mus, post_lvs, prior_mus, prior_lvs = [], [], [], [], []
        for i in range(t):
            post_mu, post_lv = self._split(self.posterior_net(torch.cat([h, e[:, i]], -1)))
            z = self.rsample(post_mu, post_lv)
            states.append(torch.cat([h, z], -1))
            post_mus.append(post_mu); post_lvs.append(post_lv)
            if i < t - 1:
                a_emb = self.action_embed(action_seq[:, i])
                h_next = self.gru(torch.cat([z, a_emb], -1), h)
                prior_mu, prior_lv = self._split(self.prior_net(h_next))
                prior_mus.append(prior_mu); prior_lvs.append(prior_lv)
                h = h_next
        states = torch.stack(states, 1)
        return states, torch.stack(post_mus, 1), torch.stack(post_lvs, 1), \
            torch.stack(prior_mus, 1), torch.stack(prior_lvs, 1)

    def imagine_step(self, h, z, action_idx):
        a_emb = self.action_embed(action_idx)
        h = self.gru(torch.cat([z, a_emb], -1), h)
        prior_mu, prior_lv = self._split(self.prior_net(h))
        z = self.rsample(prior_mu, prior_lv)
        return h, z

    def decode(self, states):
        return self.decoder(states)

    def reward(self, states):
        return self.reward_head(states).squeeze(-1)

    def cont_prob(self, states):
        return torch.sigmoid(self.continue_head(states)).squeeze(-1)

    def mask_logits(self, states):
        return self.mask_head(states)


# ------------------------------------------------------------ actor/critic ---

class Actor(nn.Module):
    def __init__(self, d_latent, n_actions, d_hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_latent, d_hidden), nn.GELU(),
                                 nn.Linear(d_hidden, d_hidden), nn.GELU(),
                                 nn.Linear(d_hidden, n_actions))

    def forward(self, state):
        return self.net(state)   # logits


class Critic(nn.Module):
    def __init__(self, d_latent, d_hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_latent, d_hidden), nn.GELU(),
                                 nn.Linear(d_hidden, d_hidden), nn.GELU(),
                                 nn.Linear(d_hidden, 1))

    def forward(self, state):
        return self.net(state).squeeze(-1)


# ------------------------------------------------------------------- data ---

def collect_random_episodes(env: CBSWrapper, n_episodes: int, max_len: int):
    """Now also collects the REAL per-step action-validity mask (env.valid_action_mask(),
    called before the action is chosen) alongside every transition, so the world model
    can learn to predict it -- this is what fixes the imagination/real-eval mask
    mismatch. Stored as bool to keep the (n_episodes, max_len, n_actions) buffer small
    (~1 byte/entry instead of 4)."""
    obs_buf, act_buf, rew_buf, active_buf, valid_mask_buf = [], [], [], [], []
    for _ in range(n_episodes):
        obs = env.reset()
        obs_seq = [obs]
        act_seq, rew_seq, active_seq, valid_mask_seq = [], [], [], []
        for t in range(max_len):
            mask = env.valid_action_mask()
            valid = np.where(mask > 0)[0]
            a = int(np.random.choice(valid))
            next_obs, r, done = env.step(a)
            act_seq.append(a); rew_seq.append(r); active_seq.append(1.0)
            valid_mask_seq.append(mask.astype(bool))
            obs_seq.append(next_obs)
            if done:
                break
        n_actions = valid_mask_seq[0].shape[0] if valid_mask_seq else env.n_actions
        # pad to max_len+1 obs / max_len act
        while len(act_seq) < max_len:
            obs_seq.append(obs_seq[-1]); act_seq.append(0); rew_seq.append(0.0); active_seq.append(0.0)
            valid_mask_seq.append(np.zeros(n_actions, dtype=bool))
        obs_buf.append(np.stack(obs_seq[:max_len + 1]))
        act_buf.append(np.array(act_seq[:max_len]))
        rew_buf.append(np.array(rew_seq[:max_len], dtype=np.float32))
        active_buf.append(np.array(active_seq[:max_len], dtype=np.float32))
        valid_mask_buf.append(np.stack(valid_mask_seq[:max_len]))
    return (np.stack(obs_buf), np.stack(act_buf), np.stack(rew_buf), np.stack(active_buf),
           np.stack(valid_mask_buf))


# ------------------------------------------------------------------ train ---

def train_world_model(model, obs, act, rew, active, valid_mask, device, epochs=60, batch_size=32, lr=3e-4):
    obs_t = torch.from_numpy(obs).float().to(device)
    act_t = torch.from_numpy(act).long().to(device)
    rew_t = torch.from_numpy(rew).float().to(device)
    active_t = torch.from_numpy(active).float().to(device)
    # kept on CPU as bool; only the current minibatch is materialized as float on GPU
    valid_mask_cpu = torch.from_numpy(valid_mask)
    n = obs_t.shape[0]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss, total_mask_acc, n_batches = 0.0, 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            o, a, r, m = obs_t[idx], act_t[idx], rew_t[idx], active_t[idx]
            vm = valid_mask_cpu[idx].to(device).float()
            states, post_mu, post_lv, prior_mu, prior_lv = model(o, a)
            recon = model.decode(states)
            obs_nll = 0.5 * (((o - recon) ** 2) / model.obs_logvar.exp() + model.obs_logvar).sum(-1)
            l_dyn, l_rep = free_bits_kl(post_mu[:, 1:], post_lv[:, 1:], prior_mu, prior_lv)
            r_pred = model.reward(states[:, 1:])
            reward_mse = (r_pred - r) ** 2
            mask_pred_logits = model.mask_logits(states[:, :-1])   # aligned with the action taken FROM this state
            mask_bce = nn.functional.binary_cross_entropy_with_logits(
                mask_pred_logits, vm, reduction="none").mean(-1)
            m_full = torch.cat([torch.ones_like(m[:, :1]), m], dim=1)
            loss = (obs_nll * m_full).sum() / m_full.sum() \
                + ((l_dyn + l_rep) * m).sum() / m.sum().clamp(min=1) \
                + (reward_mse * m).sum() / m.sum().clamp(min=1) \
                + (mask_bce * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            with torch.no_grad():
                acc = (((mask_pred_logits > 0).float() == vm).float() * m.unsqueeze(-1)).sum() \
                    / (m.sum() * vm.shape[-1]).clamp(min=1)
            total_mask_acc += acc.item(); n_batches += 1
        if (epoch + 1) % 10 == 0:
            print(f"  wm epoch {epoch+1}/{epochs}  loss={total_loss:.2f}  "
                 f"mask_acc={total_mask_acc/max(n_batches,1):.4f}", flush=True)


def lambda_returns(rewards, values, bootstrap, gamma, lam):
    """Dreamer's TD(lambda) target: R_t = r_t + gamma*((1-lam)*V(s_{t+1}) +
    lam*R_{t+1}), bootstrapped with the critic's OWN estimate at the
    imagination horizon (R_H = bootstrap) rather than truncating to zero --
    this is what actually lets a longer horizon pay off; a plain discounted
    sum with no bootstrap (the previous version) throws away the critic's
    opinion about everything past the horizon."""
    horizon = rewards.shape[1]
    returns = torch.zeros_like(rewards)
    next_return = bootstrap
    for t in reversed(range(horizon)):
        next_value = values[:, t + 1] if t + 1 < horizon else bootstrap
        next_return = rewards[:, t] + gamma * ((1 - lam) * next_value + lam * next_return)
        returns[:, t] = next_return
    return returns


def _apply_predicted_mask(logits, mask_logits):
    """Masks actor logits to predicted-valid actions, falling back to the
    unmasked logits for any row where the mask head predicts NOTHING valid
    (degenerate early-training case) rather than producing an all -1e9 row."""
    pred_valid = mask_logits > 0
    row_has_valid = pred_valid.any(dim=-1, keepdim=True)
    effective_valid = torch.where(row_has_valid, pred_valid, torch.ones_like(pred_valid))
    return logits.masked_fill(~effective_valid, -1e9)


def train_actor_critic(model, actor, critic, obs, valid_mask, device, imagine_horizon=20,
                       iterations=600, batch_size=32, lr=1e-4, gamma=0.97, lam=0.95):
    """Actor logits are masked at every imagined step -- using the REAL
    collected mask for step 0 (ground truth, since state 0 is a real
    encoded observation), and the world model's OWN mask_head prediction
    for every subsequent imagined step (no ground truth exists there).
    This is the fix for the imagination/real-eval action-mask mismatch:
    previously the actor trained against the full unmasked action space in
    imagination, then got masked to the true valid set only at real-
    environment evaluation time -- a train/eval mismatch that let it learn
    to chase world-model errors on actions it was never masked away from."""
    obs_t = torch.from_numpy(obs).float().to(device)
    valid_mask_cpu = torch.from_numpy(valid_mask)   # bool, CPU-resident (see train_world_model)
    n = obs_t.shape[0]
    opt_a = torch.optim.AdamW(actor.parameters(), lr=lr)
    opt_c = torch.optim.AdamW(critic.parameters(), lr=lr)

    for it in range(iterations):
        idx = torch.randint(0, n, (batch_size,))
        o0 = obs_t[idx, 0]
        mask0 = valid_mask_cpu[idx, 0].to(device)   # real mask for the first action, shape (B, n_actions)
        with torch.no_grad():
            e0 = model.encoder(o0)
            h = torch.zeros(batch_size, model.d_h, device=device)
            z = torch.zeros(batch_size, model.d_z, device=device)
            post_mu, post_lv = model._split(model.posterior_net(torch.cat([h, e0], -1)))
            z = model.rsample(post_mu, post_lv)

        log_probs, rewards, values, entropies = [], [], [], []
        state = torch.cat([h, z], -1)
        for step in range(imagine_horizon):
            with torch.no_grad():
                mask_logits = mask0.float() * 2 - 1 if step == 0 else model.mask_logits(state.detach())
            logits = actor(state.detach())
            masked_logits = _apply_predicted_mask(logits, mask_logits)
            dist = torch.distributions.Categorical(logits=masked_logits)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            values.append(critic(state.detach()))
            with torch.no_grad():
                h, z = model.imagine_step(h, z, action)
                new_state = torch.cat([h, z], -1)
                r = model.reward(new_state)
            rewards.append(r)
            state = new_state
        bootstrap = critic(state.detach())   # critic's own value estimate of the final imagined state

        rewards = torch.stack(rewards, 1)          # (B,H)
        values = torch.stack(values, 1)            # (B,H)
        log_probs = torch.stack(log_probs, 1)
        entropies = torch.stack(entropies, 1)

        returns = lambda_returns(rewards, values.detach(), bootstrap.detach(), gamma, lam)
        advantage = (returns - values).detach()

        actor_loss = -(log_probs * advantage).mean() - 0.001 * entropies.mean()
        critic_loss = ((values - returns.detach()) ** 2).mean()

        opt_a.zero_grad(); actor_loss.backward(); torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0); opt_a.step()
        opt_c.zero_grad(); critic_loss.backward(); torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0); opt_c.step()

        if (it + 1) % 50 == 0:
            print(f"  actor-critic iter {it+1}/{iterations}  "
                 f"mean_lambda_return={returns[:,0].mean().item():.1f}  "
                 f"mean_reward_per_step={rewards.mean().item():.2f}  "
                 f"actor_loss={actor_loss.item():.3f}  critic_loss={critic_loss.item():.3f}", flush=True)


@torch.no_grad()
def evaluate_policy(actor, model, env: CBSWrapper, device, n_episodes=10, max_len=60, random_policy=False):
    totals = []
    for _ in range(n_episodes):
        obs = env.reset()
        h = torch.zeros(1, model.d_h, device=device)
        z = torch.zeros(1, model.d_z, device=device)
        total_r = 0.0
        for t in range(max_len):
            mask = env.valid_action_mask()
            valid = np.where(mask > 0)[0]
            if random_policy:
                a = int(np.random.choice(valid))
            else:
                o = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                e = model.encoder(o)
                post_mu, post_lv = model._split(model.posterior_net(torch.cat([h, e], -1)))
                z = model.rsample(post_mu, post_lv)
                state = torch.cat([h, z], -1)
                logits = actor(state).squeeze(0).cpu().numpy()
                masked_logits = np.full_like(logits, -1e9)
                masked_logits[valid] = logits[valid]
                probs = np.exp(masked_logits - masked_logits.max())
                probs /= probs.sum()
                a = int(np.random.choice(len(probs), p=probs))
                a_emb = model.action_embed(torch.tensor([a], device=device))
                h = model.gru(torch.cat([z, a_emb], -1), h)
            obs, r, done = env.step(a)
            total_r += r
            if done:
                break
        totals.append(total_r)
    return totals


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=10, help="CyberBattleChain size (nodes = size+2; "
                   "must stay <= MAX_NODE_COUNT-2=10 or obs padding truncates real nodes)")
    p.add_argument("--max_steps", type=int, default=80)
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--wm_epochs", type=int, default=100)
    p.add_argument("--wm_batch", type=int, default=64)
    p.add_argument("--ac_iters", type=int, default=1500)
    p.add_argument("--imagine_horizon", type=int, default=30)
    p.add_argument("--ac_batch", type=int, default=64)
    p.add_argument("--tag", type=str, default="scaled")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    env = CBSWrapper(size=args.size, max_steps=args.max_steps)
    print(f"obs_dim={env.obs_dim}  n_actions={env.n_actions}", flush=True)

    print("collecting random-policy rollouts (now also recording the real per-step "
         "action-validity mask)...", flush=True)
    t0 = time.time()
    obs, act, rew, active, valid_mask = collect_random_episodes(env, n_episodes=args.episodes,
                                                                max_len=args.max_steps)
    print(f"  collected {obs.shape[0]} episodes in {time.time()-t0:.1f}s, "
         f"mean episode reward={rew.sum(1).mean():.1f}, "
         f"mean valid actions/step={valid_mask.sum(-1)[active.astype(bool)].mean():.1f} of {env.n_actions}",
         flush=True)

    model = CBSWorldModel(env.obs_dim, env.n_actions).to(device)
    print(f"world model params: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    print("training world model (+ mask_head, supervised by the real valid_action_mask)...", flush=True)
    train_world_model(model, obs, act, rew, active, valid_mask, device,
                      epochs=args.wm_epochs, batch_size=args.wm_batch)

    actor = Actor(model.d_h + model.d_z, env.n_actions).to(device)
    critic = Critic(model.d_h + model.d_z).to(device)
    print("training actor-critic inside imagined rollouts (masked by the predicted "
         "action-validity mask)...", flush=True)
    train_actor_critic(model, actor, critic, obs, valid_mask, device, imagine_horizon=args.imagine_horizon,
                       iterations=args.ac_iters, batch_size=args.ac_batch)

    print("\nevaluating in the REAL environment (masked to valid actions)...", flush=True)
    random_scores = evaluate_policy(actor, model, env, device, n_episodes=15, max_len=args.max_steps, random_policy=True)
    learned_scores = evaluate_policy(actor, model, env, device, n_episodes=15, max_len=args.max_steps, random_policy=False)
    print(f"random policy   : mean={np.mean(random_scores):.1f}  std={np.std(random_scores):.1f}  "
         f"values={[round(v,1) for v in random_scores]}", flush=True)
    print(f"learned policy  : mean={np.mean(learned_scores):.1f}  std={np.std(learned_scores):.1f}  "
         f"values={[round(v,1) for v in learned_scores]}", flush=True)

    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, f"cbs_worldmodel_policy_{args.tag}.pt")
    torch.save({"model": model.state_dict(), "actor": actor.state_dict(), "critic": critic.state_dict(),
               "obs_dim": env.obs_dim, "n_actions": env.n_actions, "size": args.size},
              ckpt_path)
    print(f"\nsaved: {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
