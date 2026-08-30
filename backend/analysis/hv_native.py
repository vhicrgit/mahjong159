"""牌型价值 E 引擎的桌面原生封装 —— 双后端自适应。

后端优先级:
  1. libmj159.so(93MB LUT 查表向听, 最快; 由 backend/native/mj159.c 编译)
  2. libmjcore.so(自包含 DFS 向听, 与手机 wasm 同一份 mjcore.c;
    缺查表的环境回退)
两个后端的 E 引擎共享同一份 mobile/wasm/hv_engine_inc.c, 语义与
backend/analysis/hand_value.py 逐位一致(对拍: tools/perf/test_hv_c_parity.py)。

用途: tools/hand_value.py 的弃牌表在换型层下 Python 要数十秒,
C 引擎整表秒级。
"""

import ctypes
import os
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SO = os.path.join(_HERE, "libmjcore.so")
_SRC = os.path.join(_ROOT, "mobile", "wasm", "mjcore.c")

_STATE = {"lib": None, "kind": None, "tried": False}


def _compile_mjcore():
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


def _load_mjcore():
    if (not os.path.exists(_SO)
            or os.path.getmtime(_SO) < os.path.getmtime(_SRC)):
        _compile_mjcore()
    L = ctypes.CDLL(_SO)
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
    return L


def lib():
    """返回 (lib, kind): kind = "mj159" | "mjcore"。两库都不可用返回 None。"""
    if _STATE["tried"]:
        return _STATE["lib"]
    _STATE["tried"] = True
    # 优先 mj159(查表, 快)。native.py 的 lib() 负责建表 + 初始化。
    try:
        from ..native import native
        L = native.lib()
        i8p = ctypes.POINTER(ctypes.c_int8)
        L.mj_hv_set2.argtypes = [i8p, i8p, ctypes.c_double, ctypes.c_int,
                                 ctypes.c_int, ctypes.c_int, ctypes.c_int]
        L.mj_hv_set2.restype = ctypes.c_int
        L.mj_hv_e_after_discard.argtypes = [ctypes.c_int]
        L.mj_hv_e_after_discard.restype = ctypes.c_double
        L.mj_hv_explain_buf.argtypes = [ctypes.c_int,
                                        ctypes.POINTER(ctypes.c_double),
                                        ctypes.POINTER(ctypes.c_int)]
        L.mj_hv_explain_buf.restype = ctypes.c_int
        _STATE["lib"], _STATE["kind"] = L, "mj159"
        return L
    except Exception:
        pass
    try:
        L = _load_mjcore()
        _STATE["lib"], _STATE["kind"] = L, "mjcore"
        return L
    except Exception:
        return None


def backend_kind() -> str:
    lib()
    return _STATE["kind"] or "none"


def _i8(counts28):
    return (ctypes.c_int8 * 28)(*[int(x) for x in counts28])


def set_hand(hand28, visible28, rho=1.0, kaizen=True,
             kai_margin=2, kai_max=1, kai_topk=6) -> bool:
    """设定分析上下文。返回 False 表示 C 库不可用(调用方回退 Python)。"""
    L = lib()
    if L is None:
        return False
    fn = L.mj_hv_set2 if _STATE["kind"] == "mj159" else L.mjc_hv_set2
    fn(_i8(hand28), _i8(visible28), float(rho), int(kaizen),
       int(kai_margin), int(kai_max), int(kai_topk))
    return True


def e_after_discard(tile: int) -> float:
    L = lib()
    fn = L.mj_hv_e_after_discard if _STATE["kind"] == "mj159" \
        else L.mjc_hv_e_after_discard
    return fn(int(tile))


def choose_discard() -> int:
    L = lib()
    fn = L.mj_hv_choose_discard if _STATE["kind"] == "mj159" \
        else L.mjc_hv_choose_discard
    return fn()


def decide_peng(tile: int) -> bool:
    L = lib()
    fn = L.mj_hv_decide_peng if _STATE["kind"] == "mj159" \
        else L.mjc_hv_decide_peng
    return bool(fn(int(tile)))


def decide_gang(tile: int, kind: str) -> bool:
    k = {"ming": 0, "an": 1, "bu": 2}[kind]
    L = lib()
    fn = L.mj_hv_decide_gang if _STATE["kind"] == "mj159" \
        else L.mjc_hv_decide_gang
    return bool(fn(int(tile), k))


def explain(tile: int) -> dict | None:
    """E 的通道分解(与 HandAnalyzer.decompose 同口径)。
    返回 dict(E, wait, c_useful, c_kai, c_peng, useful, kai, peng) 或 None。"""
    L = lib()
    if L is None:
        return None
    outf = (ctypes.c_double * 8)()
    outi = (ctypes.c_int * 192)()
    fn = L.mj_hv_explain_buf if _STATE["kind"] == "mj159" \
        else L.mjc_hv_explain_buf
    rc = fn(int(tile), outf, outi)
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
