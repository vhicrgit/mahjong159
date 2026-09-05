"""用 mortal-298k(社区权重, AGPL-3.0)在天凤规则下自对弈, 产出 mjai 牌谱。

牌谱用途见 tools/distill_mortal.py: 蒸馏其中与我们规则同构的数牌牌型知识。
libriichi.so 由 cargo build --release --features pymod 产出(构建目录在
/mnt/nebula/.../build/mortal_target, 因 home inode 配额已满)。

用法:
  python -m tools.mortal_selfplay --games 200 --out third_party/logs
"""

import argparse
import os
import secrets
import shutil
import sys

SO_DIR = "/dev/shm/mortal_target/release"
MORTAL_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "third_party", "Mortal", "mortal")
CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "third_party", "mortal-298k", "mortal_298k.pth")


def setup():
    # libriichi.so: cargo 产物名为 liblibriichi.so, python import 需去前缀
    src = os.path.join(SO_DIR, "liblibriichi.so")
    dst = os.path.join(SO_DIR, "libriichi.so")
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copyfile(src, dst)
    sys.path.insert(0, SO_DIR)
    sys.path.insert(0, os.path.abspath(MORTAL_PY))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--out", default="third_party/logs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    setup()

    import torch
    from model import Brain, DQN
    from engine import MortalEngine
    from libriichi.arena import OneVsThree

    state = torch.load(CKPT, weights_only=True, map_location="cpu")
    cfg = state["config"]
    version = cfg["control"].get("version", 1)
    mortal = Brain(version=version,
                   conv_channels=cfg["resnet"]["conv_channels"],
                   num_blocks=cfg["resnet"]["num_blocks"]).eval()
    dqn = DQN(version=version).eval()
    mortal.load_state_dict(state["mortal"])
    dqn.load_state_dict(state["current_dqn"])
    engine = MortalEngine(mortal, dqn, is_oracle=False, version=version,
                          device=torch.device("cpu"), enable_amp=False,
                          name="m298k")
    print(f"模型加载完成 version={version} "
          f"steps={state.get('steps')} best_perf={state.get('best_perf')}")

    os.makedirs(args.out, exist_ok=True)
    key = secrets.randbits(64)
    env = OneVsThree(disable_progress_bar=False, log_dir=args.out)
    # 同一引擎打四个座位(纯自对弈); seed_count = 局数
    rankings = env.py_vs_py(challenger=engine, champion=engine,
                            seed_start=(args.seed or 10000, key),
                            seed_count=args.games)
    print("rankings:", rankings)


if __name__ == "__main__":
    main()
