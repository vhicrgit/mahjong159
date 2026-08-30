"""mjcore.c(手机 wasm 规则核心) 的原生 ctypes 封装 —— 桌面端 C 版牌型价值 E。

mjcore.c 里已有完整的 HandAnalyzer.E C 移植(为 wasm 所做, 与 Python/JS 逐位一致),
桌面侧直接编译同一源文件为 libmjcore.so 复用, 不另写一份。

用途: tools/hand_value.py 的弃牌表在 kai_max=2 换型下 Python 要数十秒,
C 版单手毫秒级, 整表亚秒。

注意: 与 backend/native/libmj159.so 是两个独立自包含的核(mjcore 无大表、
可编 wasm); 别合并, mj159 的表驱动版仍是 bot 对战的主力。
"""

import ctypes
import os
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SO = os.path.join(_HERE, "libmjcore.so")
_SRC = os.path.join(_ROOT, "mobile", "wasm", "mjcore.c")

_LIB = None


def _compile():
    tmp = f"{_SO}.{os.getpid()}.tmp"
    cmd = ["cc", "-O3", "-fPIC", "-shared", "-o", tmp, _SRC]
    if os.environ.get("MJ_NATIVE_MARCH", "1") == "1":
        cmd.insert(2, "-march=native")
    try:
        subprocess.run(cmd, check=True, cwd=_HERE)
        os.replace(tmp, _SO)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def lib():
    """加载 libmjcore.so; 源文件更新或缺失时自动重编。编译失败返回 None
    (调用方回退 Python 实现)。"""
    global _LIB
    if _LIB is not None:
        return _LIB
    try:
        if (not os.path.exists(_SO)
                or os.path.getmtime(_SO) < os.path.getmtime(_SRC)):
            _compile()
        L = ctypes.CDLL(_SO)
    except Exception:
        return None
    i8p = ctypes.POINTER(ctypes.c_int8)
    L.mjc_hv_set2.argtypes = [i8p, i8p, ctypes.c_double, ctypes.c_int,
                              ctypes.c_int, ctypes.c_int, ctypes.c_int]
    L.mjc_hv_set2.restype = None
    L.mjc_hv_e_after_discard.argtypes = [ctypes.c_int]
    L.mjc_hv_e_after_discard.restype = ctypes.c_double
    L.mjc_hv_choose_discard.restype = ctypes.c_int
    L.mjc_hv_decide_peng.argtypes = [ctypes.c_int]
    L.mjc_hv_decide_peng.restype = ctypes.c_int
    L.mjc_hv_decide_gang.argtypes = [ctypes.c_int, ctypes.c_int]
    L.mjc_hv_decide_gang.restype = ctypes.c_int
    L.mjc_hv_explain_buf.argtypes = [ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_double),
                                     ctypes.POINTER(ctypes.c_int)]
    L.mjc_hv_explain_buf.restype = ctypes.c_int
    _LIB = L
    return L


def _i8(counts28):
    return (ctypes.c_int8 * 28)(*[int(x) for x in counts28])


def set_hand(hand28, visible28, rho=1.0, kaizen=True,
             kai_margin=2, kai_max=1, kai_topk=6) -> bool:
    """设定分析上下文。返回 False 表示 C 库不可用(调用方回退 Python)。"""
    L = lib()
    if L is None:
        return False
    L.mjc_hv_set2(_i8(hand28), _i8(visible28), float(rho), int(kaizen),
                  int(kai_margin), int(kai_max), int(kai_topk))
    return True


def e_after_discard(tile: int) -> float:
    return lib().mjc_hv_e_after_discard(int(tile))


def choose_discard() -> int:
    return lib().mjc_hv_choose_discard()


def decide_peng(tile: int) -> bool:
    return bool(lib().mjc_hv_decide_peng(int(tile)))


def decide_gang(tile: int, kind: str) -> bool:
    k = {"ming": 0, "an": 1, "bu": 2}[kind]
    return bool(lib().mjc_hv_decide_gang(int(tile), k))


def explain(tile: int) -> dict | None:
    """E 的通道分解(与 HandAnalyzer.decompose 同口径)。
    返回 dict(E, wait, c_useful, c_kai, c_peng, useful, kai, peng) 或 None。"""
    L = lib()
    if L is None:
        return None
    outf = (ctypes.c_double * 8)()
    outi = (ctypes.c_int * 192)()
    rc = L.mjc_hv_explain_buf(int(tile), outf, outi)
    if rc != 0:
        return None
    q = 0
    nu = outi[q]; q += 1
    useful = [(outi[q + 2 * i], outi[q + 2 * i + 1]) for i in range(nu)]
    q += 2 * nu
    nk = outi[q]; q += 1
    kai = [(outi[q + 3 * i], outi[q + 3 * i + 1], outi[q + 3 * i + 2])
           for i in range(nk)]
    q += 3 * nk
    np_ = outi[q]; q += 1
    peng = [(outi[q + 2 * i], outi[q + 2 * i + 1] / 1000.0)
            for i in range(np_)]
    return {"E": outf[0], "wait": outf[1], "c_useful": outf[2],
            "c_kai": outf[3], "c_peng": outf[4], "useful": useful,
            "kai": kai, "peng": peng}
