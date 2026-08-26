"""基准: v3 特征新增项(任意向听的有效进张, 按未见张数加权)的单步耗时"""
import time

from backend.game.engine import Game
from backend.rules.win import shanten


def eff_draws_all(counts14, unseen):
    """每个可打牌 t: 打后能降向听的进张类型数与加权张数"""
    out = {}
    for t in range(28):
        if counts14[t] <= 0:
            continue
        counts14[t] -= 1
        base = shanten(counts14)
        w = 0
        types = 0
        for d in range(28):
            if unseen[d] <= 0 or counts14[d] >= 4:
                continue
            counts14[d] += 1
            s = shanten(counts14)
            counts14[d] -= 1
            if s < base:
                w += unseen[d]
                types += 1
        out[t] = (base, types, w)
        counts14[t] += 1
    return out


def main():
    hands = [list(Game(seed=i, human_seat=-1).players[0].hand_counts)
             for i in range(30)]
    unseen = [4] * 28

    for h in hands:  # 预热 DFS 缓存
        eff_draws_all(list(h), unseen)

    n = 300
    t0 = time.time()
    for i in range(n):
        eff_draws_all(list(hands[i % 30]), unseen)
    dt = (time.time() - t0) / n * 1000
    print(f"eff_draws_all(热缓存): {dt:.2f} ms/次")

    from backend.rl.features_v2 import encode_state
    games = [Game(seed=i, human_seat=-1) for i in range(30)]
    for g in games:
        encode_state(g, 0)
    t0 = time.time()
    for i in range(n):
        encode_state(games[i % 30], 0)
    dt2 = (time.time() - t0) / n * 1000
    print(f"encode_state v2(热缓存): {dt2:.2f} ms/次")


if __name__ == "__main__":
    main()
