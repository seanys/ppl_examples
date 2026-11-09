// Player for pendulum, acrobot (data.kind "acrobot": two links, the elbow driven), cart-pole and lander episodes (data.kind
// "lander": positions, angle and fired engine in world coordinates): draws the system from recorded states and actions,
// with play, pause, a time slider,
// speed and episode choice. The active principle is marked in the principle table (rows with data-rule="0" or "1").
(function () {
  function css(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function setup(root) {
    const canvas = root.querySelector("canvas");
    const ctx = canvas.getContext("2d");
    const play = root.querySelector("[data-play]");
    const slider = root.querySelector("[data-slider]");
    const clock = root.querySelector("[data-clock]");
    const speedSel = root.querySelector("[data-speed]");
    const tabs = root.querySelector("[data-episodes]");
    const readAngle = root.querySelector("[data-angle]");
    const readVel = root.querySelector("[data-velocity]");
    const readTorque = root.querySelector("[data-torque]");
    const rows = document.querySelectorAll(root.dataset.rules + " [data-rule]");
    let data, ep, t = 0, playing = false, last = null;

    function size() {
      const w = canvas.clientWidth;
      const ratio = window.devicePixelRatio || 1;
      canvas.width = w * ratio;
      canvas.height = w * ratio;
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      draw();
    }

    function sample(arr) {
      const i = Math.min(Math.floor(t), arr.length - 1);
      const j = Math.min(i + 1, arr.length - 1);
      return [arr[i], arr[j], t - i];
    }

    function angleAt(arr) {
      let [a, b, f] = sample(arr || ep.angle);
      if (b - a > Math.PI) b -= 2 * Math.PI;
      if (a - b > Math.PI) b += 2 * Math.PI;
      return a + (b - a) * f;
    }

    function arrow(cx, cy, r, u, color) {
      // torque: an arc around (cx, cy), counterclockwise for a positive torque
      const span = (Math.abs(u) / data.max_torque) * 1.6;
      if (span <= 0.02) return;
      const start = -Math.PI / 2, ccw = u > 0;
      const end = ccw ? start - span : start + span;
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.beginPath(); ctx.arc(cx, cy, r, start, end, ccw); ctx.stroke();
      const ex = cx + r * Math.cos(end), ey = cy + r * Math.sin(end);
      const dir = end + (ccw ? -Math.PI / 2 : Math.PI / 2);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(ex + 7 * Math.cos(dir), ey + 7 * Math.sin(dir));
      ctx.lineTo(ex + 5 * Math.cos(dir + 2.3), ey + 5 * Math.sin(dir + 2.3));
      ctx.lineTo(ex + 5 * Math.cos(dir - 2.3), ey + 5 * Math.sin(dir - 2.3));
      ctx.fill();
    }

    function drawAcrobot() {
      // angles as in Acrobot-v1: the upper link from hanging down, the elbow relative to the upper link
      const w = canvas.clientWidth, cx = w / 2, cy = w / 2, R = w * 0.21;
      const a1 = angleAt(ep.angle), a2 = angleAt(ep.angle2);
      const i = Math.min(Math.round(t), ep.angle.length - 1);
      const u = ep.torque[Math.min(i, ep.torque.length - 1)];
      ctx.clearRect(0, 0, w, w);
      // the line the tip has to clear, one link length above the free joint
      ctx.strokeStyle = css("--line");
      ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 5]);
      ctx.beginPath(); ctx.moveTo(w * 0.08, cy - R); ctx.lineTo(w * 0.92, cy - R); ctx.stroke();
      ctx.setLineDash([]);
      const ex = cx + R * Math.sin(a1), ey = cy + R * Math.cos(a1);
      const tx = ex + R * Math.sin(a1 + a2), ty = ey + R * Math.cos(a1 + a2);
      ctx.strokeStyle = css("--rod");
      ctx.lineWidth = w * 0.03;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(ex, ey); ctx.lineTo(tx, ty); ctx.stroke();
      arrow(ex, ey, w * 0.06, u, css("--torque"));
      ctx.fillStyle = "#fff";
      ctx.strokeStyle = css("--rod");
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(ex, ey, w * 0.014, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      ctx.fillStyle = css("--text");
      ctx.beginPath(); ctx.arc(cx, cy, w * 0.02, 0, 2 * Math.PI); ctx.fill();
      clock.textContent = (t * data.dt).toFixed(2) + " s";
      slider.value = t;
      const active = ep.rule[i];
      rows.forEach((row) => row.classList.toggle("active", Number(row.dataset.rule) === active));
    }

    function drawCartpole() {
      // the cart on its track (data.track: half length, m) and the pole; a positive angle leans the pole right
      const w = canvas.clientWidth, h = w, ground = h * 0.66;
      const i = Math.min(Math.round(t), ep.angle.length - 1);
      const [pa, pb, f] = sample(ep.position);
      const x = pa + (pb - pa) * f, th = angleAt(ep.angle);
      const u = ep.torque[Math.min(i, ep.torque.length - 1)];
      const scale = (w * 0.42) / data.track, cx = w / 2 + x * scale;
      const cw = w * 0.16, ch = w * 0.08, pole = w * 0.36;
      ctx.clearRect(0, 0, w, h);
      ctx.strokeStyle = css("--line");
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(w / 2 - data.track * scale, ground); ctx.lineTo(w / 2 + data.track * scale, ground); ctx.stroke();
      [-1, 1].forEach((sd) => {
        ctx.beginPath(); ctx.moveTo(w / 2 + sd * data.track * scale, ground - 8); ctx.lineTo(w / 2 + sd * data.track * scale, ground + 8); ctx.stroke();
      });
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(cx, ground - ch); ctx.lineTo(cx, ground - ch - pole - 12); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#fff";
      ctx.strokeStyle = css("--text");
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.rect(cx - cw / 2, ground - ch, cw, ch); ctx.fill(); ctx.stroke();
      // push: an arrow beside the cart
      ctx.strokeStyle = ctx.fillStyle = css("--torque");
      ctx.lineWidth = 3;
      const ax = cx - u * (cw / 2 + 6), bx = ax - u * w * 0.07, ay = ground - ch / 2;
      ctx.beginPath(); ctx.moveTo(bx, ay); ctx.lineTo(ax, ay); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(ax + u * 2, ay); ctx.lineTo(ax - u * 7, ay - 5); ctx.lineTo(ax - u * 7, ay + 5); ctx.fill();
      const px = cx + pole * Math.sin(th), py = ground - ch - pole * Math.cos(th);
      ctx.strokeStyle = css("--rod");
      ctx.lineWidth = w * 0.025;
      ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(cx, ground - ch); ctx.lineTo(px, py); ctx.stroke();
      ctx.fillStyle = css("--text");
      ctx.beginPath(); ctx.arc(cx, ground - ch, w * 0.014, 0, 2 * Math.PI); ctx.fill();
      clock.textContent = (t * data.dt).toFixed(2) + " s";
      slider.value = t;
      const active = ep.rule[i];
      rows.forEach((row) => row.classList.toggle("active", Number(row.dataset.rule) === active));
    }

    function drawLander() {
      // world coordinates (m) of Lunar Lander: a 14 m square view around the pad, the terrain, the two flags, the lander
      // with its fired engine (action 1 left, 2 main, 3 right) and the path it has flown
      const w = canvas.clientWidth, view = 14, x0 = data.width / 2 - view / 2, y0 = 1.0, s = w / view;
      const X = (x) => (x - x0) * s, Y = (y) => w - (y - y0) * s;
      const i = Math.min(Math.round(t), ep.angle.length - 1);
      const [xa, xb, f] = sample(ep.x), [ya, yb] = sample(ep.y);
      const x = xa + (xb - xa) * f, y = ya + (yb - ya) * f, th = angleAt(ep.angle);
      const a = ep.action[Math.min(i, ep.action.length - 1)];
      ctx.clearRect(0, 0, w, w);
      ctx.fillStyle = css("--line");
      ctx.beginPath(); ctx.moveTo(X(ep.terrain[0][0]), w);
      ep.terrain.forEach((p) => ctx.lineTo(X(p[0]), Y(p[1])));
      ctx.lineTo(X(ep.terrain[ep.terrain.length - 1][0]), w); ctx.closePath(); ctx.fill();
      [ep.pad[0], ep.pad[1]].forEach((fx) => {
        const fy = ep.pad[2];
        ctx.strokeStyle = css("--muted"); ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(X(fx), Y(fy)); ctx.lineTo(X(fx), Y(fy + 1.67)); ctx.stroke();
        ctx.fillStyle = "#e3b505";
        ctx.beginPath(); ctx.moveTo(X(fx), Y(fy + 1.67)); ctx.lineTo(X(fx), Y(fy + 1.33)); ctx.lineTo(X(fx + 0.83), Y(fy + 1.5)); ctx.fill();
      });
      ctx.strokeStyle = css("--rod"); ctx.globalAlpha = 0.35; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(X(ep.x[0]), Y(ep.y[0]));
      for (let k = 1; k <= i; k++) ctx.lineTo(X(ep.x[k]), Y(ep.y[k]));
      ctx.stroke(); ctx.globalAlpha = 1;
      // body frame: x to the right of the lander, y up, rotated by the lander angle (counterclockwise positive)
      ctx.save();
      ctx.translate(X(x), Y(y)); ctx.rotate(-th); ctx.scale(s, -s);
      ctx.fillStyle = css("--torque");
      if (a === 2) { ctx.beginPath(); ctx.moveTo(-0.2, -0.33); ctx.lineTo(0.2, -0.33); ctx.lineTo(0, -1.1); ctx.fill(); }
      if (a === 1 || a === 3) {
        const sd = a === 1 ? -1 : 1;
        ctx.beginPath(); ctx.moveTo(sd * 0.6, 0.15); ctx.lineTo(sd * 0.6, -0.1); ctx.lineTo(sd * 1.1, 0.02); ctx.fill();
      }
      ctx.strokeStyle = css("--rod"); ctx.lineWidth = 0.12; ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(-0.4, -0.33); ctx.lineTo(-0.7, -0.8); ctx.moveTo(0.4, -0.33); ctx.lineTo(0.7, -0.8); ctx.stroke();
      ctx.fillStyle = css("--rod");
      ctx.beginPath(); ctx.moveTo(-0.47, -0.33); ctx.lineTo(0.47, -0.33); ctx.lineTo(0.6, 0.1); ctx.lineTo(0.33, 0.43);
      ctx.lineTo(-0.33, 0.43); ctx.lineTo(-0.6, 0.1); ctx.closePath(); ctx.fill();
      ctx.restore();
      clock.textContent = (t * data.dt).toFixed(2) + " s";
      slider.value = t;
      const active = ep.rule[i];
      rows.forEach((row) => row.classList.toggle("active", Number(row.dataset.rule) === active));
    }

    function draw() {
      if (!ep) return;
      if (data.kind === "acrobot") return drawAcrobot();
      if (data.kind === "cartpole") return drawCartpole();
      if (data.kind === "lander") return drawLander();
      const w = canvas.clientWidth, cx = w / 2, cy = w / 2, R = w * 0.34;
      const th = angleAt();
      const i = Math.min(Math.round(t), ep.angle.length - 1);
      const u = ep.torque[Math.min(i, ep.torque.length - 1)];
      ctx.clearRect(0, 0, w, w);
      // path of the tip and the upright direction
      ctx.strokeStyle = css("--line");
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(cx, cy, R, 0, 2 * Math.PI); ctx.stroke();
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx, cy - R - 14); ctx.stroke();
      ctx.setLineDash([]);
      // torque: an arc around the pivot, counterclockwise for a positive torque
      const span = (Math.abs(u) / data.max_torque) * 1.6;
      if (span > 0.02) {
        const r = w * 0.09, start = -Math.PI / 2, ccw = u > 0;
        const end = ccw ? start - span : start + span;
        ctx.strokeStyle = css("--torque");
        ctx.lineWidth = 3;
        ctx.beginPath(); ctx.arc(cx, cy, r, start, end, ccw); ctx.stroke();
        const ex = cx + r * Math.cos(end), ey = cy + r * Math.sin(end);
        const dir = end + (ccw ? -Math.PI / 2 : Math.PI / 2);
        ctx.fillStyle = css("--torque");
        ctx.beginPath();
        ctx.moveTo(ex + 7 * Math.cos(dir), ey + 7 * Math.sin(dir));
        ctx.lineTo(ex + 5 * Math.cos(dir + 2.3), ey + 5 * Math.sin(dir + 2.3));
        ctx.lineTo(ex + 5 * Math.cos(dir - 2.3), ey + 5 * Math.sin(dir - 2.3));
        ctx.fill();
      }
      // rod and bob; theta = 0 is upright and a positive angle turns counterclockwise
      const x = cx - R * Math.sin(th), y = cy - R * Math.cos(th);
      ctx.strokeStyle = css("--rod");
      ctx.lineWidth = w * 0.035;
      ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(x, y); ctx.stroke();
      ctx.fillStyle = css("--text");
      ctx.beginPath(); ctx.arc(cx, cy, w * 0.022, 0, 2 * Math.PI); ctx.fill();
      // readouts and the active principle
      const deg = Math.abs(th * 180 / Math.PI) % 360;
      if (readAngle) readAngle.textContent = (deg > 180 ? 360 - deg : deg).toFixed(1) + "°";
      if (readVel) readVel.textContent = ep.velocity[i].toFixed(2) + " rad/s";
      if (readTorque) readTorque.textContent = u.toFixed(2);
      clock.textContent = (t * data.dt).toFixed(2) + " s";
      slider.value = t;
      const active = ep.rule[i];
      rows.forEach((row) => row.classList.toggle("active", Number(row.dataset.rule) === active));
    }

    function frame(now) {
      if (!playing) return;
      if (last !== null) t += ((now - last) / 1000) * Number(speedSel.value) / data.dt;
      last = now;
      if (t >= ep.angle.length - 1) {
        t = ep.angle.length - 1;
        stop();
      }
      draw();
      if (playing) requestAnimationFrame(frame);
    }

    function start() {
      if (t >= ep.angle.length - 1) t = 0;
      playing = true; last = null;
      play.textContent = "Pause";
      requestAnimationFrame(frame);
    }

    function stop() {
      playing = false;
      play.textContent = "Play";
    }

    function choose(k) {
      ep = data.episodes[k];
      t = 0;
      slider.max = ep.angle.length - 1;
      tabs.querySelectorAll("button").forEach((b, j) => b.setAttribute("aria-pressed", String(j === k)));
      draw();
    }

    play.addEventListener("click", () => (playing ? stop() : start()));
    slider.addEventListener("input", () => { stop(); t = Number(slider.value); draw(); });
    window.addEventListener("resize", size);

    const loaded = (window.PPL_EPISODES || {})[root.dataset.run];
    (loaded ? Promise.resolve(loaded) : Promise.reject())
      .then((d) => {
        data = d;
        d.episodes.forEach((e, k) => {
          const b = document.createElement("button");
          b.type = "button";
          b.textContent = e.label;
          b.addEventListener("click", () => { stop(); choose(k); });
          tabs.appendChild(b);
        });
        choose(0);
        size();
        if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) start();
      })
      .catch(() => {
        root.querySelector(".player-error").hidden = false;
      });
  }

  document.querySelectorAll("[data-player]").forEach(setup);
})();
