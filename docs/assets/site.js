// Downloads that save the file instead of opening it, and copy buttons for command blocks.
(function () {
  document.querySelectorAll("[data-download]").forEach((a) => {
    a.setAttribute("download", a.dataset.download);
    a.addEventListener("click", (e) => {
      if (location.protocol === "file:") return;            // a local file: the browser handles the link itself
      e.preventDefault();
      fetch(a.href)
        .then((r) => r.blob())
        .then((blob) => {
          const url = URL.createObjectURL(new Blob([blob], { type: "application/octet-stream" }));
          const tmp = document.createElement("a");
          tmp.href = url; tmp.download = a.dataset.download;
          document.body.appendChild(tmp); tmp.click(); tmp.remove();
          setTimeout(() => URL.revokeObjectURL(url), 1000);
        })
        .catch(() => { location.href = a.href; });
    });
  });
  document.querySelectorAll("[data-copy]").forEach((b) => {
    b.addEventListener("click", () => {
      const text = b.parentElement.querySelector("code").innerText.replace(/^\$ /gm, "");
      navigator.clipboard.writeText(text).then(() => {
        b.textContent = "Copied";
        setTimeout(() => (b.textContent = "Copy"), 1500);
      });
    });
  });
})();
