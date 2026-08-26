/* 安康159 - 麻将牌面渲染(SVG 程序化绘制, 无外部图片素材)
 * 饼子=圆点阵, 条子=竹节阵, 万子=数字+萬, 红中=中
 * 两版(web/手机)共用
 */

const TileArt = (() => {
  const GREEN = "#1b7a43", RED_C = "#b3242a", BLUE = "#1d5fa8", DARK = "#222";

  // 饼子点阵坐标模板 [x, y, r, 颜色] (viewBox 40x56)
  // 传统配色: 蓝/绿/红三色搭配
  const BING_DOTS = {
    1: [[20, 27, 10, GREEN]],                       // 大圆(绿环红芯, 单独处理)
    2: [[20, 16, 6.5, GREEN], [20, 38, 6.5, BLUE]],  // 上绿下蓝
    3: [[13, 13, 6, BLUE], [20, 27, 6, RED_C], [27, 41, 6, GREEN]],  // 蓝红绿斜线
    4: [[13, 15, 6, BLUE], [27, 15, 6, GREEN], [13, 39, 6, GREEN], [27, 39, 6, BLUE]],  // 两蓝两绿对角
    5: [[13, 14, 5.5, BLUE], [27, 14, 5.5, GREEN], [20, 27, 6.5, RED_C], [13, 40, 5.5, GREEN], [27, 40, 5.5, BLUE]],  // 四角+中心红
    6: [[13, 15, 5, GREEN], [27, 15, 5, GREEN], [13, 30, 5, RED_C], [27, 30, 5, RED_C], [13, 43, 5, RED_C], [27, 43, 5, RED_C]],  // 顶二绿+底四红
    7: [[12, 9, 4.5, GREEN], [20, 17, 4.5, GREEN], [28, 25, 4.5, GREEN], [13, 35, 4.5, RED_C], [27, 35, 4.5, RED_C], [13, 46, 4.5, RED_C], [27, 46, 4.5, RED_C]],  // 顶三绿斜线+底四红
    8: [[13, 11, 4.5, BLUE], [27, 11, 4.5, BLUE], [13, 21.5, 4.5, BLUE], [27, 21.5, 4.5, BLUE], [13, 32, 4.5, BLUE], [27, 32, 4.5, BLUE], [13, 42.5, 4.5, BLUE], [27, 42.5, 4.5, BLUE]],  // 全蓝 2x4
    9: [[13, 13, 4.5, BLUE], [20, 13, 4.5, BLUE], [27, 13, 4.5, BLUE], [13, 26.5, 4.5, RED_C], [20, 26.5, 4.5, RED_C], [27, 26.5, 4.5, RED_C], [13, 40, 4.5, GREEN], [20, 40, 4.5, GREEN], [27, 40, 4.5, GREEN]],  // 顶蓝中红底绿
  };

  function bingSVG(n) {
    const dots = BING_DOTS[n] || [];
    let s = "";
    if (n === 1) {
      // 1饼: 绿外环 + 红芯(太阳饼)
      s += `<circle cx="20" cy="27" r="10" fill="none" stroke="${GREEN}" stroke-width="2.6"/>`;
      s += `<circle cx="20" cy="27" r="5.5" fill="${RED_C}"/>`;
      return s;
    }
    dots.forEach(([x, y, r, color]) => {
      // 双环效果: 外环 + 内点
      s += `<circle cx="${x}" cy="${y}" r="${r}" fill="none" stroke="${color}" stroke-width="2.2"/>`;
      s += `<circle cx="${x}" cy="${y}" r="${r * 0.35}" fill="${color}"/>`;
    });
    return s;
  }

  // 条子(竹节): [x, y, r, len]  len=竹节长度(不给则用 r*2.1)
  // viewBox 40x56, 牌面可用区域约 y=6..50 (44 高), 竖向居中线 y=28
  // 排列依据传统骰牌实物: 6条=3列x2行, 7条=顶一根红+3列x2行, 8条单独绘制
  const TIAO_POS = {
    1: [[20, 27, 9]],
    // 2条: 上下两根居中
    2: [[20, 17, 6, 19], [20, 39, 6, 19]],
    // 3条: 顶部 1 根居中 + 下方 2 根(两行居中对齐 y=17/39)
    3: [[20, 17, 5.6, 19], [13, 39, 5.6, 19], [27, 39, 5.6, 19]],
    // 4条: 2列 x 2行
    4: [[13, 17, 5.6, 19], [27, 17, 5.6, 19],
        [13, 39, 5.6, 19], [27, 39, 5.6, 19]],
    // 5条: 四角 + 中心红(三行错位, 四角与中心留 1 单位间隙不重叠)
    5: [[12, 13, 5, 14], [28, 13, 5, 14],
        [20, 28, 5.2, 14],
        [12, 43, 5, 14], [28, 43, 5, 14]],
    // 6条: 3列 x 2行
    6: [[10, 17, 4.6, 19], [20, 17, 4.6, 19], [30, 17, 4.6, 19],
        [10, 39, 4.6, 19], [20, 39, 4.6, 19], [30, 39, 4.6, 19]],
    // 7条: 顶部 1 根(红) + 下方 3列 x 2行; 三行等长等距, 顶部不能比下面短
    7: [[20, 10.5, 4.4, 15],
        [10, 28, 4.4, 15], [20, 28, 4.4, 15], [30, 28, 4.4, 15],
        [10, 45.5, 4.4, 15], [20, 45.5, 4.4, 15], [30, 45.5, 4.4, 15]],
    // 8条 不用点阵, 见 tiao8SVG()
    // 9条: 3列 x 3行, 中间一行红
    9: [[10, 14, 4.4, 13], [20, 14, 4.4, 13], [30, 14, 4.4, 13],
        [10, 28, 4.4, 13], [20, 28, 4.4, 13], [30, 28, 4.4, 13],
        [10, 42, 4.4, 13], [20, 42, 4.4, 13], [30, 42, 4.4, 13]],
  };

  // 传统条子配色: 哪些位置的竹节用红色(其余绿色)
  const TIAO_RED_IDX = {
    5: [2],           // 五条中心红
    7: [0],           // 七条顶部红
    9: [3, 4, 5],     // 九条中间一行三根红
  };

  // 画一根立体竹节(胶囊形 + 竹节环 + 高光)
  // opts: { len 自定义长度(默认 r*2.1), angle 顺时针旋转角度(默认 0) }
  let _bgSeq = 0;
  function bamboo(x, y, r, color, opts) {
    const o = opts || {};
    const w = r * 1.05;
    const h = o.len !== undefined ? o.len : r * 2.1;
    const gid = `bg${_bgSeq++}`;
    const rot = o.angle ? ` transform="rotate(${o.angle} ${x} ${y})"` : "";
    return `
      <defs><linearGradient id="${gid}" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="${color}" stop-opacity="0.75"/>
        <stop offset="45%" stop-color="${color}"/>
        <stop offset="100%" stop-color="${color}" stop-opacity="0.8"/>
      </linearGradient></defs>
      <g${rot}>
        <rect x="${x - w / 2}" y="${y - h / 2}" width="${w}" height="${h}" rx="${w / 2}" fill="url(#${gid})"/>
        <line x1="${x - w / 2}" y1="${y}" x2="${x + w / 2}" y2="${y}" stroke="rgba(0,0,0,.35)" stroke-width="1"/>
        <line x1="${x - w / 4}" y1="${y - h / 3}" x2="${x - w / 4}" y2="${y + h / 3}" stroke="rgba(255,255,255,.35)" stroke-width="0.7"/>
      </g>`;
  }

  /**
   * 八条: 传统画法 —— 左右各两根竖竹节(上下相接) + 中间四根斜竹节组成菱形,
   * 共 8 根, 整体呈 W/M 叠合的格子状。不是 2列x4行 竖排(那是八饼的排法)。
   */
  function tiao8SVG() {
    const r = 3.5;
    let s = "";
    // 左右各两根竖竹节
    for (const x of [7.5, 32.5]) {
      s += bamboo(x, 17, r, GREEN, { len: 19 });
      s += bamboo(x, 39, r, GREEN, { len: 19 });
    }
    // 中间菱形四边(上峰 20,10 / 左 11,28 / 右 29,28 / 下峰 20,46)
    const A = 26.6, L = 20.1;
    s += bamboo(15.5, 19, r, GREEN, { len: L, angle: A });    // 左上边
    s += bamboo(24.5, 19, r, GREEN, { len: L, angle: -A });   // 右上边
    s += bamboo(15.5, 37, r, GREEN, { len: L, angle: -A });   // 左下边
    s += bamboo(24.5, 37, r, GREEN, { len: L, angle: A });    // 右下边
    return s;
  }

  // 一条: 幺鸡(站立的小鸟)
  function yaojiSVG() {
    return `
      <!-- 尾羽(彩色) -->
      <path d="M20 24 Q10 14 8 6" stroke="#1d5fa8" stroke-width="2.2" fill="none" stroke-linecap="round"/>
      <path d="M21 25 Q14 12 13 4" stroke="#1b7a43" stroke-width="2.2" fill="none" stroke-linecap="round"/>
      <path d="M22 25 Q20 12 19 5" stroke="#b3242a" stroke-width="2.2" fill="none" stroke-linecap="round"/>
      <!-- 身体 -->
      <ellipse cx="21" cy="33" rx="8.5" ry="11" fill="#1b7a43"/>
      <!-- 胸部浅色 -->
      <ellipse cx="22.5" cy="36" rx="4.5" ry="6" fill="#5cb85c" opacity="0.6"/>
      <!-- 头 -->
      <circle cx="19" cy="20" r="6" fill="#1b7a43"/>
      <!-- 眼睛 -->
      <circle cx="17.5" cy="19" r="1.3" fill="#fff"/>
      <circle cx="17.5" cy="19" r="0.6" fill="#000"/>
      <!-- 嘴(红色三角) -->
      <path d="M13.5 20 L10 21.5 L13.5 23 Z" fill="#b3242a"/>
      <!-- 翅膀 -->
      <path d="M24 28 Q30 33 25 40 Q21 36 22 30 Z" fill="#0d5c2e"/>
      <!-- 腿和爪 -->
      <line x1="19" y1="43" x2="19" y2="50" stroke="#b3242a" stroke-width="1.4"/>
      <line x1="23" y1="43" x2="23" y2="50" stroke="#b3242a" stroke-width="1.4"/>
      <path d="M16 50 L22 50 M17 47 L19 50 M21 50 L23 47" stroke="#b3242a" stroke-width="1.2" fill="none"/>`;
  }

  function tiaoSVG(n) {
    if (n === 1) return yaojiSVG();   // 幺鸡
    if (n === 8) return tiao8SVG();  // 八条: 竖+菱形组合
    const pos = TIAO_POS[n] || [];
    const redIdx = TIAO_RED_IDX[n] || [];
    let s = "";
    pos.forEach(([x, y, r, len], i) => {
      const color = redIdx.includes(i) ? RED_C : GREEN;
      s += bamboo(x, y, r, color, len !== undefined ? { len } : undefined);
    });
    return s;
  }

  // 万子: 上数字 + 下萬。传统牌上 5 写作"伍"(大写), 不是"五"
  function wanSVG(n) {
    const nums = ["", "一", "二", "三", "四", "伍", "六", "七", "八", "九"];
    return `
      <text x="20" y="22" text-anchor="middle" font-size="17" font-weight="bold" fill="${DARK}" font-family="serif">${nums[n]}</text>
      <text x="20" y="46" text-anchor="middle" font-size="20" font-weight="bold" fill="${RED_C}" font-family="serif">萬</text>`;
  }

  // 红中
  function hzSVG() {
    return `<text x="20" y="40" text-anchor="middle" font-size="30" font-weight="bold" fill="${RED_C}" font-family="serif">中</text>`;
  }

  // 牌背
  function backSVG() {
    return `<rect x="3" y="3" width="34" height="50" rx="4" fill="none" stroke="rgba(255,255,255,.25)" stroke-width="1.5"/>`;
  }

  /**
   * 生成牌面 SVG 内容(不含外层牌框, 牌框由 CSS 提供)
   * t: 牌id(0-26 条饼万, 27红中), 返回内层 SVG 字符串
   */
  function faceInner(t) {
    if (t === 27) return hzSVG();
    const suit = Math.floor(t / 9), n = (t % 9) + 1;
    if (suit === 0) return tiaoSVG(n);
    if (suit === 1) return bingSVG(n);
    return wanSVG(n);
  }

  /**
   * 完整 SVG 元素字符串。w/h 不指定时省略属性, 由 CSS 控制尺寸
   */
  function svg(t, w, h) {
    const inner = t === -1 ? backSVG() : faceInner(t);
    const wh = (w && h) ? `width="${w}" height="${h}"` : "";
    return `<svg viewBox="0 0 40 56" ${wh} xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet">${inner}</svg>`;
  }

  return { svg, faceInner };
})();
