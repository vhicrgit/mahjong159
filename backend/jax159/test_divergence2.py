"""根治浮点分叉: 在分叉点抓取分叉局面的完整状态对比。

复刻 rollout_jax 循环, 保存每步 (feats, q, acts, phase, hands, last_discard,
wall_pos, turn)。full(B=768) vs split(两个B=384)。找到第一个分叉局面后,
逐字段对比该局面的 State 和当前步的 obs/q, 定位 B 敏感源。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_default_matmul_precision", "float32")

from backend.jax159.test_pmap_rollout import _rand_games
from backend.jax159.rollout import build_world_states, _peng_gang_ok
from backend.jax159.convert import slice_state
from backend.jax159.env159 import step_jit, legal_jit, load_win_tables, P_OVER, State
from backend.jax159.features import encode_obs
from backend.jax159.jax_net import JaxNet
from backend.jax159.shanten import load_front_table


def run_steps(sts, net, n_steps):
    obs_j = jax.jit(encode_obs)
    legal_v = jax.vmap(jax.jit(legal_jit))
    step_v = jax.vmap(jax.jit(step_jit))
    B = sts.hands.shape[0]
    done = jnp.zeros(B, dtype=jnp.bool_)
    hist = []
    for step in range(n_steps):
        if bool(done.all()):
            break
        feats = obs_j(sts, sts.turn.astype(jnp.int8))
        legal = legal_v(sts)[:, :28]
        q = net.q_values(feats)
        q = jnp.where(legal, q, -1e9)
        acts = q.argmax(axis=-1).astype(jnp.int32)
        react_mask = np.asarray(sts.phase) == 1
        if react_mask.any():
            acts_np = np.asarray(acts).copy()
            hands_np = np.asarray(sts.hands)
            pend_p = np.asarray(sts.pend_peng)
            pend_g = np.asarray(sts.pend_gang)
            ld_np = np.asarray(sts.last_discard)
            for i in np.nonzero(react_mask)[0]:
                ps = int(np.argmax((pend_p[i] | pend_g[i]).astype(np.int8)))
                hand = list(hands_np[i, ps])
                t = int(ld_np[i])
                dg = pend_g[i, ps] and _peng_gang_ok(hand, t, 3)
                dp = pend_p[i, ps] and _peng_gang_ok(hand, t, 2)
                acts_np[i] = 2 if dg else (1 if dp else 0)
            acts = jnp.asarray(acts_np, dtype=jnp.int32)
        hist.append(dict(
            acts=np.asarray(acts),
            feats=np.asarray(feats).copy(),
            q=np.asarray(q).copy(),
            **{f: np.asarray(getattr(sts, f)).copy() for f in State._fields},
        ))
        sts = step_v(sts, acts)
        done = done | (np.asarray(sts.phase) == P_OVER)
    return hist


def main():
    load_win_tables(); load_front_table()
    snaps = _rand_games(14, seed=7)
    cand_lists = []
    for g, hero in snaps:
        hc = g.players[hero].hand_counts
        cand_lists.append([t for t in range(28) if hc[t] > 0][:4])
    net = JaxNet("models/dqn_shaped_100k_best_jax.npz")
    sts, meta, _ = build_world_states(snaps, cand_lists, 16, world_seed=99)
    B = sts.hands.shape[0]
    print(f"B={B}", flush=True)

    hf = run_steps(sts, net, 60)
    ha = run_steps(slice_state(sts, 0, B // 2), net, 60)
    hb = run_steps(slice_state(sts, B // 2, B), net, 60)
    print(f"步数 full={len(hf)} a={len(ha)} b={len(hb)}", flush=True)

    n = min(len(hf), len(ha), len(hb))
    for s in range(n):
        a_full = hf[s]["acts"]
        a_split = np.concatenate([ha[s]["acts"], hb[s]["acts"]])
        diff = np.nonzero(a_full != a_split)[0]
        if len(diff) == 0:
            continue
        i = int(diff[0])
        gi = i if i < B // 2 else i - B // 2
        src = ha if i < B // 2 else hb
        print(f"第{s}步分叉 {len(diff)} 局面, 首个局面{i}(块内{gi}): "
              f"full_act={a_full[i]} split_act={a_split[i]}", flush=True)
        # 对比该局面 s 步的输入状态的所有字段
        prev = s
        for f in State._fields:
            vf = hf[prev][f][i]
            vs = src[prev][f][gi]
            if not np.array_equal(vf, vs):
                d = np.abs(vf.astype(np.int64) - vs.astype(np.int64))
                print(f"  !!字段 {f} 不同: 最大差={d.max()}", flush=True)
                if f in ("melds_tile", "melds_kind", "melds_from", "hands",
                         "discards"):
                    print(f"    full={vf.tolist()}", flush=True)
                    print(f"    split={vs.tolist()}", flush=True)
        # 对比分叉局面当前步的 obs(feats) 和 q
        ff = hf[prev]["feats"][i]; fs = src[prev]["feats"][gi]
        fdiff = np.abs(ff - fs)
        print(f"  feats 最大差={fdiff.max():.3e} 不一致维数={(fdiff>0).sum()}", flush=True)
        if fdiff.max() > 0:
            didx = np.nonzero(fdiff > 0)[0]
            print(f"  差异维度前10={didx[:10].tolist()} full={ff[didx[:5]]} split={fs[didx[:5]]}", flush=True)
        qf = hf[prev]["q"][i]; qs = src[prev]["q"][gi]
        qdiff = np.abs(qf - qs)
        print(f"  q 最大差={qdiff.max():.3e}", flush=True)
        fq = np.argsort(qf)[-3:][::-1]; sq = np.argsort(qs)[-3:][::-1]
        print(f"  full q top3: idx={fq.tolist()} val={qf[fq]}", flush=True)
        print(f"  split q top3: idx={sq.tolist()} val={qs[sq]}", flush=True)
        break
    else:
        print(f"前{n}步无分叉", flush=True)


if __name__ == "__main__":
    main()
