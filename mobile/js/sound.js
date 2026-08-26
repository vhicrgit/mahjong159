/* 安康159 - 音效模块(Web Audio API 程序化合成, 无外部音频文件)
 * 两版(web/手机)共用
 */

const SoundFX = (() => {
  let ctx = null;
  let enabled = true;

  function ensureCtx() {
    if (!ctx) {
      ctx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (ctx.state === "suspended") ctx.resume();
    return ctx;
  }

  // 基础: 播放一个带包络的振荡器音
  function tone(freq, dur, type = "sine", vol = 0.2, when = 0) {
    if (!enabled) return;
    try {
      const c = ensureCtx();
      const t = c.currentTime + when;
      const o = c.createOscillator();
      const g = c.createGain();
      o.type = type;
      o.frequency.setValueAtTime(freq, t);
      g.gain.setValueAtTime(0.0001, t);
      g.gain.exponentialRampToValueAtTime(vol, t + 0.01);
      g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
      o.connect(g).connect(c.destination);
      o.start(t);
      o.stop(t + dur + 0.02);
    } catch (e) { /* 忽略音频错误 */ }
  }

  // 噪声(用于出牌的"啪"声)
  function noise(dur = 0.08, vol = 0.15, when = 0) {
    if (!enabled) return;
    try {
      const c = ensureCtx();
      const t = c.currentTime + when;
      const buf = c.createBuffer(1, c.sampleRate * dur, c.sampleRate);
      const data = buf.getChannelData(0);
      for (let i = 0; i < data.length; i++) data[i] = (Math.random() * 2 - 1) * (1 - i / data.length);
      const src = c.createBufferSource();
      src.buffer = buf;
      const g = c.createGain();
      g.gain.setValueAtTime(vol, t);
      g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
      const f = c.createBiquadFilter();
      f.type = "bandpass";
      f.frequency.value = 2000;
      src.connect(f).connect(g).connect(c.destination);
      src.start(t);
    } catch (e) { /* 忽略 */ }
  }

  return {
    toggle() { enabled = !enabled; return enabled; },
    setEnabled(v) { enabled = v; },
    isEnabled() { return enabled; },
    draw() { tone(660, 0.06, "triangle", 0.12); },                    // 摸牌: 轻"嗒"
    discard() { noise(0.07, 0.2); tone(180, 0.08, "square", 0.08); }, // 出牌: 啪
    peng() { tone(320, 0.12, "triangle", 0.25); tone(240, 0.14, "triangle", 0.2, 0.05); }, // 碰: 咚
    gang() { tone(220, 0.16, "sawtooth", 0.28); tone(160, 0.2, "sawtooth", 0.22, 0.07); }, // 杠: 重咚
    win() {  // 胡牌: 上扬旋律
      [523, 659, 784, 1047].forEach((f, i) => tone(f, 0.18, "triangle", 0.22, i * 0.1));
    },
    lose() { [400, 320, 240].forEach((f, i) => tone(f, 0.2, "sine", 0.15, i * 0.12)); }, // 别人胡
    huang() { tone(300, 0.3, "sine", 0.15); },                       // 黄庄
    click() { tone(880, 0.04, "square", 0.06); },                    // 点击
  };
})();
