const API_BASE = "http://localhost:5000";

document.addEventListener("DOMContentLoaded", async () => {
  const urlInput = document.getElementById("urlInput");
  const scanBtn = document.getElementById("scanBtn");
  const apiStatus = document.getElementById("apiStatus");
  const loading = document.getElementById("loading");
  const results = document.getElementById("results");
  const errorMsg = document.getElementById("errorMsg");
  const verdictBanner = document.getElementById("verdictBanner");
  const confidenceVal = document.getElementById("confidenceVal");
  const progressBar = document.getElementById("progressBar");
  const modelsList = document.getElementById("modelsList");

  // Check API status
  try {
    const res = await fetch(`${API_BASE}/`, { method: "GET" });
    if (!res.ok) throw new Error();
    apiStatus.textContent = "API Online";
    apiStatus.classList.remove("offline");
  } catch (e) {
    apiStatus.textContent = "API Offline";
    apiStatus.classList.add("offline");
  }

  // Get current active tab URL
  if (typeof chrome !== "undefined" && chrome.tabs && chrome.tabs.query) {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs && tabs[0] && tabs[0].url) {
        const currentUrl = tabs[0].url;
        // Don't auto-fill chrome:// or extension pages
        if (currentUrl.startsWith("http://") || currentUrl.startsWith("https://")) {
          urlInput.value = currentUrl;
          // Auto-trigger scan on tab open
          runScan(currentUrl);
        }
      }
    });
  }

  scanBtn.addEventListener("click", () => {
    const targetUrl = urlInput.value.trim();
    if (targetUrl) {
      runScan(targetUrl);
    }
  });

  urlInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
      const targetUrl = urlInput.value.trim();
      if (targetUrl) {
        runScan(targetUrl);
      }
    }
  });

  async function runScan(targetUrl) {
    // UI Reset
    loading.classList.remove("hidden");
    results.classList.add("hidden");
    errorMsg.classList.add("hidden");

    try {
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: targetUrl })
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || `Server returned ${res.status}`);
      }

      const data = await res.json();
      renderResults(data);
    } catch (err) {
      loading.classList.add("hidden");
      errorMsg.textContent = `❌ ${err.message || "Failed to connect to local API server. Make sure run_api.bat is running."}`;
      errorMsg.classList.remove("hidden");
    }
  }

  function renderResults(data) {
    loading.classList.add("hidden");
    results.classList.remove("hidden");

    const isPhish = data.final_label === 1;
    verdictBanner.textContent = isPhish ? "⚠️ PHISHING DETECTED" : "✅ LEGITIMATE SITE";
    verdictBanner.className = `verdict-banner ${isPhish ? "phishing" : "legit"}`;

    const confPct = Math.round((data.confidence || 0) * 100);
    confidenceVal.textContent = `${confPct}%`;
    progressBar.style.width = `${Math.min(Math.max(confPct, 0), 100)}%`;

    // Render per-model details
    modelsList.innerHTML = "";
    if (data.per_model) {
      Object.entries(data.per_model).forEach(([mName, mRes]) => {
        const row = document.createElement("div");
        row.className = "model-row";
        const labelText = mRes.label === 1 ? "PHISH" : "LEGIT";
        const badgeClass = mRes.label === 1 ? "phish" : "legit";
        const probText = (mRes.prob * 100).toFixed(1);

        row.innerHTML = `
          <span class="model-name">${mName.replace("_", " ").toUpperCase()}</span>
          <span>
            <span class="badge-vote ${badgeClass}">${labelText}</span>
            <code style="margin-left: 6px; font-size: 0.75rem; color: #a0a0c0;">${probText}%</code>
          </span>
        `;
        modelsList.appendChild(row);
      });
    }
  }
});
