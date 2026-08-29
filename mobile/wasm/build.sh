#!/bin/sh
# 编译 mjcore.c -> wasm, 并生成内嵌 base64 的 JS 载荷文件
#
# 为什么内嵌 base64 而不是单独放 .wasm 文件:
#   Android WebView 从 file:///android_asset/ 加载页面时, fetch() 对 file:// 协议
#   被拦截, 拿不到 .wasm 字节。内嵌成 JS 字符串最稳, 代价是体积涨 33%(44KB -> 59KB)。
#
# 依赖: zig (brew install zig)。也可换成带 wasm32 后端的 clang:
#   /opt/homebrew/opt/llvm@21/bin/clang --target=wasm32 ...
#
# 用法: sh mobile/wasm/build.sh

set -e
HERE=$(cd "$(dirname "$0")" && pwd)
OUT_JS="$HERE/../js/mjcore_wasm.js"

# -O3 且不开 -ffast-math: E 的浮点累加必须与 Python/JS 逐位一致
# stack-size 调大: E_rec 递归深度可达 ~100 层, 每层含若干 28 字节局部数组
zig cc -target wasm32-freestanding -O3 -nostdlib \
    -Wl,--no-entry -Wl,--export-dynamic -Wl,-z,stack-size=4194304 \
    -o "$HERE/mjcore.wasm" "$HERE/mjcore.c"

SIZE=$(wc -c < "$HERE/mjcore.wasm" | tr -d ' ')
B64=$(base64 < "$HERE/mjcore.wasm" | tr -d '\n')

cat > "$OUT_JS" <<EOF
/* 安康159 - mjcore.wasm 的 base64 载荷(由 mobile/wasm/build.sh 自动生成, 请勿手改)
 *
 * 源码: mobile/wasm/mjcore.c
 * 大小: $SIZE 字节
 *
 * 内嵌而非独立 .wasm 文件的原因: Android WebView 下 file:// 协议无法 fetch。
 */
const MJ_WASM_B64 = "$B64";
EOF

echo "mjcore.wasm  $SIZE 字节"
echo "生成 $OUT_JS  ($(wc -c < "$OUT_JS" | tr -d ' ') 字节)"

# 顺带编一份原生动态库, 供 Python 侧对拍测试
cc -O3 -fPIC -shared -o "$HERE/libmjcore.dylib" "$HERE/mjcore.c" 2>/dev/null \
  && echo "libmjcore.dylib (原生对拍用) 已更新"
