(() => {
  "use strict";

  const API_URL = "/analyze";

  const $post      = document.getElementById("post-input");
  const $btn       = document.getElementById("analyze-btn");
  const $error     = document.getElementById("error-box");
  const $results   = document.getElementById("results");
  const $pills     = document.getElementById("intent-pills");
  const $indicator  = document.getElementById("concern-indicator");
  const $why       = document.getElementById("why-text");
  const $sugBox    = document.getElementById("suggestion-box");
  const $sugText   = document.getElementById("suggestion-text");

  const CONCERN_POS = { low: "12%", medium: "50%", high: "88%" };

  $btn.addEventListener("click", run);
  $post.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") run();
  });

  async function run() {
    const text = $post.value.trim();
    if (!text) {
      showError("Please paste a post before analyzing.");
      return;
    }

    hideError();
    hideResults();
    setLoading(true);

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ post: text }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Server error (${res.status})`);
      }

      const data = await res.json();
      render(data);
    } catch (err) {
      showError(err.message || "Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  function render(data) {
    renderIntents(data.intents);
    renderConcern(data.concern);
    renderWhy(data.post_excerpt);
    renderSuggestion(data.reply, data.crisis_level, data.crisis_detected);
    $results.hidden = false;
  }

  function renderIntents(intents) {
    $pills.innerHTML = "";
    if (!intents || intents.length === 0) {
      $pills.innerHTML = '<span class="pill pill-0">None detected</span>';
      return;
    }
    intents.forEach((tag, i) => {
      const el = document.createElement("span");
      el.className = `pill pill-${i % 9}`;
      el.textContent = tag;
      $pills.appendChild(el);
    });
  }

  function renderConcern(concern) {
    const level = (concern || "medium").toLowerCase();
    $indicator.style.left = CONCERN_POS[level] || CONCERN_POS.medium;
  }

  function renderWhy(excerpt) {
    $why.textContent = excerpt || "";
  }

  function renderSuggestion(reply, crisisLevel, crisisDetected) {
    const isCrisis = crisisDetected ||
      ["immediate", "high"].includes((crisisLevel || "").toLowerCase());

    $sugBox.classList.toggle("crisis", isCrisis);
    $sugText.textContent = reply || "";
  }

  /* ── UI helpers ──────────────────────────────────────────────────────── */

  function setLoading(on) {
    $btn.disabled = on;
    $btn.innerHTML = on
      ? '<span class="spinner"></span>Analyzing…'
      : "Analyze";
  }

  function showError(msg) {
    $error.textContent = msg;
    $error.hidden = false;
  }

  function hideError() {
    $error.hidden = true;
  }

  function hideResults() {
    $results.hidden = true;
  }
})();
