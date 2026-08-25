"""诊断: BC-only 模型 vs 规则Bot 的真实胜率"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from backend.rl.model import build_model
from backend.rl.selfplay import generate_dataset
from backend.rl.train import get_device, train_bc
from backend.rl.evaluate import play_eval_game

device = get_device()
print("=== 步骤1: 生成 BC 数据 (500局, 并行) ===")
data = generate_dataset(500, workers=8)
print(f"样本数: {len(data['acts'])}")

print("\n=== 步骤2: BC 训练 small 模型 (30 epochs) ===")
model = build_model("small").to(device)
print(f"参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
train_bc(model, data, device, epochs=30)

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "bc_only_small.pt")
torch.save({"model": model.state_dict(), "size": "small"}, out_path)

print("\n=== 步骤3: 评估 BC-only 模型 vs 3规则Bot (200局) ===")
wins, total = 0, 0.0
for i in range(200):
    sd, won = play_eval_game(300000 + i, out_path)
    total += sd
    wins += won
    if (i+1) % 50 == 0:
        print(f"  {i+1}/200: 胜率 {wins/(i+1):.1%}, 场均 {total/(i+1):+.2f}")

print(f"\n最终结果: 胜率 {wins/200:.1%}, 场均 {total/200:+.2f}")
print("(随机基线25%, 完美克隆规则Bot应接近25%)")
