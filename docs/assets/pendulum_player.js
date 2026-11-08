// Player for pendulum episodes: draws the pendulum from recorded angles and torques, with play, pause, a time slider,
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

    function angleAt() {
      let [a, b, f] = sample(ep.angle);
      if (b - a > Math.PI) b -= 2 * Math.PI;
      if (a - b > Math.PI) b += 2 * Math.PI;
      return a + (b - a) * f;
    }

    function draw() {
      if (!ep) return;
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
