"""jax159 包初始化: 修掉 jaxlib 静默退回 CPU 的动态库冲突。

现象: `import jax; jax.devices()` 只报一句 WARNING("a CUDA-enabled jaxlib is not
installed. Falling back to cpu") 就默默用 CPU 跑, 所有 rollout 慢几十倍且不报错。

根因: 系统 ldconfig 里 libnvJitLink.so.12 指向 /usr/local/cuda(12.4), 而 venv 里
装的是 nvidia-cusparse-cu12 12.9 —— 它需要 12.9 才有的符号
__nvJitLinkGetErrorLogSize_12_9, 于是 libcusparse 加载失败, jax 判定"没有 CUDA
jaxlib"。取证: ctypes.CDLL(venv 的 libcusparse.so.12) 报 undefined symbol。

修法: 在 import jax 之前把 venv 自带的 libnvJitLink.so.12 以 RTLD_GLOBAL 预载,
后续 dlopen libcusparse 时符号已在全局命名空间。只碰 nvjitlink 一个库 ——
预载 cudnn 会和 /lib64/libcudnn_graph.so.9 冲突。

用 MJ_NO_CUDA_PRELOAD=1 可跳过。
"""

import ctypes
import glob
import os
import sys


def _preload_nvjitlink() -> None:
    if os.environ.get("MJ_NO_CUDA_PRELOAD") == "1":
        return
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    pats = [
        os.path.join(repo, ".venv-jax", "lib", "python*", "site-packages",
                     "nvidia", "nvjitlink", "lib", "libnvJitLink.so*"),
        os.path.join(os.path.dirname(os.path.dirname(sys.executable)),
                     "lib", "python*", "site-packages", "nvidia",
                     "nvjitlink", "lib", "libnvJitLink.so*"),
    ]
    for pat in pats:
        for so in sorted(glob.glob(pat)):
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                return
            except OSError:
                pass


_preload_nvjitlink()
