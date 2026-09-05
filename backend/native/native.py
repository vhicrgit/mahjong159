"""ctypes 封装: 加载 libmj159.so, 首次导入时按需编译。

对外提供与 backend.rules / backend.ai.bot_v* 同口径的原生实现:
  shanten(counts28) / is_win(counts28)
  choose_discard_v10(...) / choose_discard_v1(...)
  decide_peng(bot, ...) / decide_gang(bot, ...)

口径说明(为什么 v10 与 v31 的出牌/杠共用一个实现):
  shanten_with_melds(hand, n) 内部直接 return shanten(hand), 而 shanten() 的
  need 由手牌张数推导 —— 所以 v31 的 _sh(hand, n_melds) 与 v10 的 shanten(hand)
  逐位等价, n_melds 只进了 lru_cache 的 key。逐条比对后两者只有 decide_peng
  真正不同(v10 不考虑碰后还要打一张, v31 考虑), 故 peng 用 bot 参数区分。
"""

import ctypes
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SO = os.path.join(_HERE, "libmj159.so")
_SRC = os.path.join(_HERE, "mj159.c")
# mj159.c 尾部 #include 的共享 E 引擎: 只比 mj159.c 会让引擎改动静默不生效
_HV_INC = os.path.join(_ROOT, "mobile", "wasm", "hv_engine_inc.c")

_LIB = None


def _compile():
    """编译到临程文件再原子改名 —— spawn 起的多个 worker 会同时首次导入本模块,
    直接写 libmj159.so 会让别的 worker dlopen 到写了一半的文件。"""
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


def _need_build() -> bool:
    if not os.path.exists(_SO):
        return True
    built = os.path.getmtime(_SO)
    return any(built < os.path.getmtime(p) for p in (_SRC, _HV_INC))


def lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    if _need_build():
        _compile()
    L = ctypes.CDLL(_SO)
    L.mj_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    L.mj_init.restype = ctypes.c_int
    i8p = ctypes.POINTER(ctypes.c_int8)
    u8p = ctypes.POINTER(ctypes.c_uint8)
    L.mj_shanten.argtypes = [i8p]
    L.mj_shanten.restype = ctypes.c_int
    L.mj_is_win.argtypes = [i8p]
    L.mj_is_win.restype = ctypes.c_int
    L.mj_choose_discard_v10.argtypes = [
        i8p, i8p, u8p, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_int]
    L.mj_choose_discard_v10.restype = ctypes.c_int
    L.mj_choose_discard_v1.argtypes = [i8p, i8p, u8p]
    L.mj_choose_discard_v1.restype = ctypes.c_int
    L.mj_decide_peng.argtypes = [ctypes.c_int, i8p, ctypes.c_int]
    L.mj_decide_peng.restype = ctypes.c_int
    L.mj_decide_gang.argtypes = [ctypes.c_int, i8p, ctypes.c_int, ctypes.c_int]
    L.mj_decide_gang.restype = ctypes.c_int
    L.mj_beam_detail.argtypes = [i8p, i8p, ctypes.c_int, ctypes.c_int,
                                 ctypes.POINTER(ctypes.c_int32),
                                 ctypes.POINTER(ctypes.c_int32)]
    L.mj_beam_detail.restype = ctypes.c_int
    L.mj_discard_shanten.argtypes = [i8p, ctypes.POINTER(ctypes.c_int32),
                                     ctypes.POINTER(ctypes.c_int32)]
    L.mj_discard_shanten.restype = ctypes.c_int
    L.mj_waits_ukeire.argtypes = [i8p, i8p]
    L.mj_waits_ukeire.restype = ctypes.c_int
    L.mj_score_discards_v10.argtypes = [
        i8p, i8p, u8p, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double)]
    L.mj_score_discards_v10.restype = ctypes.c_int
    L.mj_shanten_batch.argtypes = [i8p, ctypes.c_int,
                                   ctypes.POINTER(ctypes.c_int32)]
    L.mj_is_win_batch.argtypes = [i8p, ctypes.c_int,
                                  ctypes.POINTER(ctypes.c_int32)]
    # 牌型价值 E 引擎(与 mjcore.c 共享 hv_engine_inc.c)
    L.mj_hv_set2.argtypes = [i8p, i8p, ctypes.c_double, ctypes.c_int,
                             ctypes.c_int, ctypes.c_int, ctypes.c_int]
    L.mj_hv_set2.restype = ctypes.c_int
    L.mj_hv_e_after_discard.argtypes = [ctypes.c_int]
    L.mj_hv_e_after_discard.restype = ctypes.c_double
    L.mj_hv_choose_discard.restype = ctypes.c_int
    L.mj_hv_decide_peng.argtypes = [ctypes.c_int]
    L.mj_hv_decide_peng.restype = ctypes.c_int
    L.mj_hv_decide_gang.argtypes = [ctypes.c_int, ctypes.c_int]
    L.mj_hv_decide_gang.restype = ctypes.c_int
    L.mj_hv_explain_buf.argtypes = [ctypes.c_int,
                                    ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_int)]
    L.mj_hv_explain_buf.restype = ctypes.c_int

    front = os.environ.get("MJ_FRONT_BIN",
                           os.path.join(_ROOT, "models", "mj_front.bin"))
    win = os.environ.get("MJ_WIN_BIN",
                         os.path.join(_ROOT, "models", "mj_win.bin"))
    if not (os.path.exists(front) and os.path.exists(win)):
        # 表是派生产物(不入库); 由 git 里的 suit_front_table.npz 重建(~15s)
        print("首次使用: 由 suit_front_table.npz 生成紧凑表...", flush=True)
        subprocess.run([sys.executable, "-m", "backend.native.build_tables"],
                       check=True, cwd=_ROOT)
    rc = L.mj_init(front.encode(), win.encode())
    if rc != 0:
        raise RuntimeError(
            f"mj_init 失败 rc={rc}; 缺表? 先跑 "
            f"python -m backend.native.build_tables ({front}, {win})")
    _LIB = L
    return _LIB


_Buf28 = ctypes.c_int8 * 28
_UBuf28 = ctypes.c_uint8 * 28


def _i8(seq):
    return _Buf28(*[int(x) for x in seq])


def _u8(seq):
    return _UBuf28(*[1 if x else 0 for x in seq])


def shanten(counts28) -> int:
    return lib().mj_shanten(_i8(counts28))


def is_win(counts28) -> bool:
    return bool(lib().mj_is_win(_i8(counts28)))


def choose_discard_v10(hand28, unseen28, penged28, eg,
                       sw=100.0, uw=1.0, cw=0.5, rw=0.0, cont_max=2) -> int:
    return lib().mj_choose_discard_v10(
        _i8(hand28), _i8(unseen28), _u8(penged28),
        ctypes.c_double(eg), ctypes.c_double(sw), ctypes.c_double(uw),
        ctypes.c_double(cw), ctypes.c_double(rw), ctypes.c_int(cont_max))


def choose_discard_v1(hand28, visible28, penged28) -> int:
    return lib().mj_choose_discard_v1(_i8(hand28), _i8(visible28),
                                      _u8(penged28))


def decide_peng(bot: int, hand28, tile: int) -> bool:
    return bool(lib().mj_decide_peng(bot, _i8(hand28), tile))


def decide_gang(bot: int, hand28, tile: int, kind: str) -> bool:
    k = {"ming": 0, "an": 1, "bu": 2}[kind]
    return bool(lib().mj_decide_gang(bot, _i8(hand28), tile, k))


def beam_detail(counts14, future_draws, beam: int = 12) -> dict:
    """等价于 bot_oracle.search_first_discard_detail(counts14, future_draws, beam)。
    返回 {首出牌: (win_depth 或 None, 视野内最小向听)}。"""
    n = len(future_draws)
    fut = (ctypes.c_int8 * max(n, 1))(*[int(x) for x in future_draws])
    wd = (ctypes.c_int32 * 28)()
    sh = (ctypes.c_int32 * 28)()
    rc = lib().mj_beam_detail(_i8(counts14), fut, n, beam, wd, sh)
    if rc != 0:
        raise RuntimeError(f"mj_beam_detail rc={rc}")
    return {t: (None if wd[t] == -1 else wd[t], sh[t])
            for t in range(28) if wd[t] != -2}


def discard_shanten(hand14):
    """[(tile, shanten), ...] —— discard_options 里除 waits 之外的部分。"""
    tiles = (ctypes.c_int32 * 28)()
    shs = (ctypes.c_int32 * 28)()
    n = lib().mj_discard_shanten(_i8(hand14), tiles, shs)
    return [(tiles[i], shs[i]) for i in range(n)]


def waits_ukeire(hand13, unseen28) -> int:
    """sum(unseen[w] for w in waiting_tiles(hand13))"""
    return lib().mj_waits_ukeire(_i8(hand13), _i8(unseen28))


def score_discards_v10(hand28, unseen28, penged28, eg,
                       sw=100.0, uw=1.0, cw=0.5, rw=0.0, cont_max=2):
    """v10/v31 每个候选弃牌的打分明细(与 chooser 同一套 C 代码)。
    返回 [{'tile','shanten','ukeire','cont','score'}, ...] 按 tile 升序。"""
    buf = (ctypes.c_double * (28 * 5))()
    n = lib().mj_score_discards_v10(
        _i8(hand28), _i8(unseen28), _u8(penged28),
        ctypes.c_double(eg), ctypes.c_double(sw), ctypes.c_double(uw),
        ctypes.c_double(cw), ctypes.c_double(rw), ctypes.c_int(cont_max),
        buf)
    out = []
    for k in range(n):
        t, s, u, c, sc = buf[k * 5:k * 5 + 5]
        out.append({"tile": int(t), "shanten": int(s), "ukeire": int(u),
                    "cont": c, "score": sc})
    return out


def shanten_batch(hands):
    """hands: (n,28) numpy int8 -> (n,) numpy int32"""
    import numpy as np
    h = np.ascontiguousarray(hands, dtype=np.int8)
    out = np.empty(h.shape[0], dtype=np.int32)
    lib().mj_shanten_batch(
        h.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)), h.shape[0],
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
    return out


def is_win_batch(hands):
    import numpy as np
    h = np.ascontiguousarray(hands, dtype=np.int8)
    out = np.empty(h.shape[0], dtype=np.int32)
    lib().mj_is_win_batch(
        h.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)), h.shape[0],
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
    return out


if __name__ == "__main__":
    lib()
    print("libmj159 就绪:", _SO, file=sys.stderr)
