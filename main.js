/* ML Pipeline Dashboard — main.js */

document.addEventListener("DOMContentLoaded", () => {
  // Auto-dismiss alerts after 5 seconds
  document.querySelectorAll(".alert").forEach(alert => {
    setTimeout(() => {
      const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
      if (bsAlert) bsAlert.close();
    }, 5000);
  });

  // Highlight active nav link
  const path = window.location.pathname;
  document.querySelectorAll(".nav-link").forEach(link => {
    if (link.getAttribute("href") === path) {
      link.classList.add("active");
    }
  });

  // Table row hover highlight for best row
  document.querySelectorAll(".best-row td").forEach(td => {
    td.style.color = "inherit";
  });

  // Animate stat values (count up)
  document.querySelectorAll(".stat-value").forEach(el => {
    const target = parseFloat(el.textContent);
    if (!isNaN(target)) {
      let start = 0;
      const step = target / 30;
      const timer = setInterval(() => {
        start += step;
        if (start >= target) {
          el.textContent = Number.isInteger(target) ? target : target.toFixed(4);
          clearInterval(timer);
        } else {
          el.textContent = Number.isInteger(target) ? Math.floor(start) : start.toFixed(4);
        }
      }, 25);
    }
  });
});