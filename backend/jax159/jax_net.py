"""JAX 版 MLP: 从 torch 导出的 npz 权重构建, 与 backend/rl/model.py 逐位一致。

结构: input_proj(628->H) + N×ResBlock + q_head(H->28) [+value_head 忽略]
前向: h=relu(W0x+b0); block: h=relu(fc2(relu(fc1(h)))+h); q = Wq h + bq
"""

import numpy as np

import jax
import jax.numpy as jnp


class JaxNet:
    def __init__(self, path: str):
        z = np.load(path)
        self.params = {k: jnp.asarray(v) for k, v in z.items()}
        self.blocks = sorted({int(k.split(".")[1])
                              for k in self.params
                              if k.startswith("blocks.")})
        self.hidden = self.params["input_proj.weight"].shape[0]
        self.feat_dim = self.params["input_proj.weight"].shape[1]

    def q_values(self, x: jax.Array) -> jax.Array:
        """x: (B, feat_dim) float32 -> (B, 28) Q值"""
        w0 = self.params["input_proj.weight"]
        b0 = self.params["input_proj.bias"]
        h = jax.nn.relu(x @ w0.T + b0)
        for i in self.blocks:
            w1 = self.params[f"blocks.{i}.fc1.weight"]
            b1 = self.params[f"blocks.{i}.fc1.bias"]
            w2 = self.params[f"blocks.{i}.fc2.weight"]
            b2 = self.params[f"blocks.{i}.fc2.bias"]
            h = jax.nn.relu(jax.nn.relu(h @ w1.T + b1) @ w2.T + b2 + h)
        wq = self.params["q_head.weight"]
        bq = self.params["q_head.bias"]
        return h @ wq.T + bq

    @classmethod
    def from_dict(cls, sd: dict):
        """从 torch state_dict (numpy 值) 构建"""
        obj = cls.__new__(cls)
        obj.params = {k: jnp.asarray(v) for k, v in sd.items()}
        obj.blocks = sorted({int(k.split(".")[1])
                             for k in obj.params
                             if k.startswith("blocks.")})
        obj.hidden = obj.params["input_proj.weight"].shape[0]
        obj.feat_dim = obj.params["input_proj.weight"].shape[1]
        return obj


def q_forward(params: dict, x: jax.Array) -> jax.Array:
    """函数式前向: params 为权重 dict(显式参数, 更新不触发重编译)。"""
    w0 = params["input_proj.weight"]
    b0 = params["input_proj.bias"]
    h = jax.nn.relu(x @ w0.T + b0)
    i = 0
    while True:
        w1 = params.get(f"blocks.{i}.fc1.weight")
        if w1 is None:
            break
        b1 = params[f"blocks.{i}.fc1.bias"]
        w2 = params[f"blocks.{i}.fc2.weight"]
        b2 = params[f"blocks.{i}.fc2.bias"]
        h = jax.nn.relu(jax.nn.relu(h @ w1.T + b1) @ w2.T + b2 + h)
        i += 1
    wq = params["q_head.weight"]
    bq = params["q_head.bias"]
    return h @ wq.T + bq
