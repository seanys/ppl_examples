// Header and footer shared by every page: each page holds <div data-site-header></div> and <div data-site-footer></div>.
(function () {
  const mark = '<svg class="mark" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-opacity="0.3" stroke-width="1.5"/><line x1="12" y1="12" x2="12" y2="3" stroke="currentColor" stroke-width="2.6" stroke-linecap="round"/><circle cx="12" cy="12" r="2.2" fill="currentColor"/></svg>';
  const icons = {
    mail: '<svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M3 5h18a1 1 0 0 1 1 1v12a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1zm9 7.2L4 7.3V17h16V7.3l-8 4.9zM5.2 7l6.8 4.2L18.8 7H5.2z"/></svg>',
    github: '<svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M12 2a10 10 0 0 0-3.16 19.49c.5.09.68-.22.68-.48v-1.7c-2.78.6-3.37-1.34-3.37-1.34-.45-1.15-1.11-1.46-1.11-1.46-.91-.62.07-.61.07-.61 1 .07 1.53 1.03 1.53 1.03.9 1.52 2.34 1.08 2.91.83.09-.65.35-1.08.63-1.33-2.22-.25-4.55-1.11-4.55-4.94 0-1.09.39-1.98 1.03-2.68-.1-.25-.45-1.27.1-2.65 0 0 .84-.27 2.75 1.02a9.5 9.5 0 0 1 5 0c1.91-1.29 2.75-1.02 2.75-1.02.55 1.38.2 2.4.1 2.65.64.7 1.03 1.59 1.03 2.68 0 3.84-2.34 4.69-4.57 4.93.36.31.68.92.68 1.85v2.75c0 .27.18.58.69.48A10 10 0 0 0 12 2z"/></svg>',
    scholar: '<svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M12 3 1 9l11 6 9-4.9V17h2V9L12 3zm-6 9.3v3.2c0 1.9 2.7 3.5 6 3.5s6-1.6 6-3.5v-3.2l-6 3.3-6-3.3z"/></svg>',
    linkedin: '<svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M4 3h16a1 1 0 0 1 1 1v16a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1zm3.3 6.8H5v8.4h2.3V9.8zM6.15 5.9a1.3 1.3 0 1 0 0 2.6 1.3 1.3 0 0 0 0-2.6zM19 13.4c0-2.3-1.2-3.8-3.3-3.8-1.1 0-1.9.6-2.2 1.2V9.8h-2.2v8.4h2.3v-4.3c0-1.1.4-1.9 1.4-1.9s1.4.8 1.4 1.9v4.3H19v-4.8z"/></svg>',
  };
  const ext = 'target="_blank" rel="noopener"';

  const header = `
  <div class="topbar">
    <div class="wrap">
      <a class="brand" href="index.html">${mark}<span>PPL</span></a>
      <nav><a href="index.html#examples">Examples</a></nav>
    </div>
  </div>`;

  const footer = `
  <footer class="site-foot">
    <div class="wrap">
      <div class="foot-grid">
        <div class="foot-about">
          <a class="brand" href="index.html">${mark}<span>Prescriptive Policy Learning</span></a>
          <p>Principles of behavior handed directly to agents, with the training code of every example.</p>
        </div>
        <div>
          <h4>Resources</h4>
          <ul>
            <li><a href="https://seanys.github.io/" ${ext}>Homepage</a></li>
            <li><a href="https://ai4systems.org/" ${ext}>Decision for Complex Systems</a></li>
          </ul>
        </div>
        <div>
          <h4>Contact</h4>
          <a class="mail" href="mailto:yang-shan@nus.edu.sg">${icons.mail}yang-shan@nus.edu.sg</a>
          <div class="socials">
            <a href="https://github.com/seanys" ${ext} aria-label="GitHub">${icons.github}</a>
            <a href="https://scholar.google.com/citations?user=Z__RCosAAAAJ" ${ext} aria-label="Google Scholar">${icons.scholar}</a>
            <a href="https://www.linkedin.com/in/shan-yang-712135141" ${ext} aria-label="LinkedIn">${icons.linkedin}</a>
          </div>
        </div>
      </div>
      <div class="foot-bottom">
        <span>© Copyright 2026 YANG Shan.</span>
        <a href="#top">Back to top ↑</a>
      </div>
    </div>
  </footer>`;

  document.querySelectorAll("[data-site-header]").forEach((el) => (el.outerHTML = header));
  document.querySelectorAll("[data-site-footer]").forEach((el) => (el.outerHTML = footer));
})();
