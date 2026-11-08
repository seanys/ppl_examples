// Pictures drawn from recorded episodes.
//   kind "pendulum": a still image of a swing-up, the rod at each turning point of the swing, fading in time.
//   kind "loop":     the swing-up played in a loop, with a short trail of the tip.
(function () {
  function css(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function prepare(canvas) {
    const ratio = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * ratio; canvas.height = h * ratio;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    return [ctx, w, h];
  }

  function settled(ep) {
    const k = ep.angle.findIndex((a, i) => ep.angle.slice(i).every((b) => Math.abs(b) < 0.1));
    return k < 0 ? ep.angle.length - 1 : k;
  }

  function rod(ctx, cx, cy, R, th, width, alpha, color) {
    ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineCap = "round";
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx - R * Math.sin(th), cy - R * Math.cos(th)); ctx.stroke();
    ctx.globalAlpha = 1;
  }

  function frame(ctx, w, h, cx, cy, R) {
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = css("--line"); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(cx, cy, R, 0, 2 * Math.PI); ctx.stroke();
    ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx, cy - R - 10); ctx.stroke();
    ctx.setLineDash([]);
  }

  function pivot(ctx, cx, cy, r) {
    ctx.fillStyle = css("--text");
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, 2 * Math.PI); ctx.fill();
  }

  function pendulum(canvas, ep) {
    const [ctx, w, h] = prepare(canvas);
    const cx = w / 2, cy = h * 0.52, R = h * 0.36;
    frame(ctx, w, h, cx, cy, R);
    const end = settled(ep), color = css("--rod");
    const turns = [0];
    for (let i = 1; i < end; i++) if (ep.velocity[i - 1] * ep.velocity[i] < 0) turns.push(i);
    turns.forEach((i, k) => rod(ctx, cx, cy, R, ep.angle[i], h * 0.022, 0.18 + 0.4 * (k / turns.length), color));
    rod(ctx, cx, cy, R, ep.angle[ep.angle.length - 1], h * 0.032, 1, color);
    pivot(ctx, cx, cy, h * 0.024);
  }

  function loop(canvas, ep, dt) {
    const end = Math.min(settled(ep) + 30, ep.angle.length - 1);
    let t0 = null;
    function step(now) {
      if (t0 === null) t0 = now;
      const [ctx, w, h] = prepare(canvas);
      const cx = w / 2, cy = h / 2, R = h * 0.34;
      frame(ctx, w, h, cx, cy, R);
      const i = Math.floor(((now - t0) / 1000 / dt) % (end + 1));
      for (let k = Math.max(1, i - 12); k <= i; k++) {
        const a = ep.angle[k - 1], b = ep.angle[k];
        if (Math.abs(b - a) > Math.PI) continue;
        ctx.globalAlpha = 0.5 * (1 - (i - k) / 12);
        ctx.strokeStyle = css("--torque"); ctx.lineWidth = 3; ctx.lineCap = "round";
        ctx.beginPath();
        ctx.moveTo(cx - R * Math.sin(a), cy - R * Math.cos(a));
        ctx.lineTo(cx - R * Math.sin(b), cy - R * Math.cos(b));
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
      rod(ctx, cx, cy, R, ep.angle[i], h * 0.035, 1, css("--rod"));
      pivot(ctx, cx, cy, h * 0.026);
      requestAnimationFrame(step);
    }
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) pendulum(canvas, ep);
    else requestAnimationFrame(step);
  }

  function all(first) {
    document.querySelectorAll("canvas[data-thumb]").forEach((c) => {
      const data = (window.PPL_EPISODES || {})[c.dataset.thumb];
      if (!data) return;
      if (c.dataset.kind === "loop") { if (first) loop(c, data.episodes[0], data.dt); }
      else pendulum(c, data.episodes[0]);
    });
  }
  all(true);
  window.addEventListener("resize", () => all(false));
})();
