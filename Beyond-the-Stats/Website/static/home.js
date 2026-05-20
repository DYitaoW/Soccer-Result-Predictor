// functions used for the home page
const upcomingCache = { global: [], mls: [], extra: [], cups: [] };
const winnerCache = { global: [], mls: [], extra: [] };

function renderTopPicks() {
    const allRows = dedupeFixtures(
    [
        ...(upcomingCache.global || []),
        ...(upcomingCache.mls || []),
        ...(upcomingCache.extra || []),
        ...(upcomingCache.cups || []),
    ]
        .filter(isValidProbabilityRow)
    );
    const futureRows = allRows.filter(isLikelyFutureFixture);
    const sourceRows = futureRows.length ? futureRows : allRows;
    const picks = pickRandomRows(sourceRows, 12);
    if (!picks.length) {
    topPicksList.innerHTML = "<p class=\"muted-placeholder\">Run daily predictions to populate future games.</p>";
    return;
    }
    topPicksList.innerHTML = picks.map((r) => `
    <button
        class="pick-card match-toggle"
        type="button"
        data-home-team="${escapeHtml(r.home_team)}"
        data-away-team="${escapeHtml(r.away_team)}"
        aria-label="Open ${escapeHtml(r.home_team)} vs ${escapeHtml(r.away_team)} head to head"
    >
        <p class="pick-league">${escapeHtml(r.competition)}</p>
        <p class="pick-match">${escapeHtml(r.home_team)} vs ${escapeHtml(r.away_team)}</p>
        <p class="match-meta">${escapeHtml(`${r.weekday || ""} ${r.date_label || ""}`.trim())}${r.time_label ? ` - ${escapeHtml(r.time_label)}` : ""}</p>
        <p class="pick-prediction">Prediction: ${escapeHtml(r.winner_label)}</p>
        <div class="probability-track">
        <div style="width: ${Number(r.prob_home) || 0}%; background-color: #55d37a;" title="${escapeHtml(r.home_team)}"></div>
        <div style="width: ${Number(r.prob_draw) || 0}%; background-color: #93a4b3;" title="Draw"></div>
        <div style="width: ${Number(r.prob_away) || 0}%; background-color: #7297ff;" title="${escapeHtml(r.away_team)}"></div>
        </div>
        <div class="probability-labels">
        <span>H: ${pctLabel(r.prob_home)}%</span>
        <span>D: ${pctLabel(r.prob_draw)}%</span>
        <span>A: ${pctLabel(r.prob_away)}%</span>
        </div>
        <p class="pick-confidence">Confidence: ${pctLabel(toConfidence(r))}%</p>
    </button>
    `).join("");
} 

async function preloadHomeData() {
    if (!upcomingCache.global.length) {
    try {
        const respGlobal = await fetch("/api/upcoming/global");
        const dataGlobal = await respGlobal.json();
        if (respGlobal.ok && dataGlobal.ok) {
        upcomingCache.global = dataGlobal.rows || [];
        }
    } catch (err) {
        console.error("Failed to preload global upcoming rows", err);
    }
    }
    if (!upcomingCache.mls.length) {
    try {
        const respMls = await fetch("/api/upcoming/mls");
        const dataMls = await respMls.json();
        if (respMls.ok && dataMls.ok) {
        upcomingCache.mls = dataMls.rows || [];
        }
    } catch (err) {
        console.error("Failed to preload MLS upcoming rows", err);
    }
    }
    if (!upcomingCache.extra.length) {
    try {
        const respExtra = await fetch("/api/upcoming/extra");
        const dataExtra = await respExtra.json();
        if (respExtra.ok && dataExtra.ok) {
        upcomingCache.extra = dataExtra.rows || [];
        }
    } catch (err) {
        console.error("Failed to preload extra upcoming rows", err);
    }
    }
    if (!upcomingCache.cups.length) {
    try {
        const respCups = await fetch("/api/upcoming/cups");
        const dataCups = await respCups.json();
        if (respCups.ok && dataCups.ok) {
        upcomingCache.cups = dataCups.rows || [];
        }
    } catch (err) {
        console.error("Failed to preload cup upcoming rows", err);
    }
    }
    renderTopPicks();
}

function renderWinners() {
  const dataset = winnerDataset.value || "global";
  const rows = winnerCache[dataset] || [];

  if (!rows.length) {
    winnerView.innerHTML = "<p class='muted-placeholder'>No predictions available.</p>";
    return;
  }

  winnerView.innerHTML = `
    <table class="winner-table">
      <thead>
        <tr>
          <th>League</th>
          <th>Predicted Winner</th>
          <th>Confidence</th>
        </tr>
      </thead>
      <tbody>
        ${rows.map(r => `
          <tr>
            <td>${escapeHtml(r.league)}</td>
            <td>${escapeHtml(r.team)}</td>
            <td>${pctLabel(r.confidence)}%</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function renderWinnerView() {
    const mode = winnerDataset.value;
    const payload = leagueTablesCache[mode];
    if (!payload || !payload.tables) {
        winnerView.innerHTML = "<p>No winner data available.</p>";
        return;
    }
    const leagues = Object.keys(payload.tables || {}).filter((name) => name !== "__mls_bracket__");
    if (!leagues.length) {
        winnerView.innerHTML = "<p>No winner data available.</p>";
        return;
    }
    const ordered = [...leagues].sort((a, b) => a.localeCompare(b));
    let html = `
        <table class="league-table">
        <thead>
            <tr>
            <th>League</th><th>Predicted Winner</th><th>Win Chance</th>
            </tr>
        </thead>
        <tbody>
    `;
    for (const league of ordered) {
        const rows = payload.tables[league] || [];
        if (!rows.length) continue;
        const winner = [...rows].sort((a, b) => {
        const aw = Number(a.win_league_pct) || 0;
        const bw = Number(b.win_league_pct) || 0;
        if (bw !== aw) return bw - aw;
        return (Number(a.position) || 999) - (Number(b.position) || 999);
        })[0];
        html += `
        <tr>
            <td>${league}</td>
            <td>${winner ? winner.team : "N/A"}</td>
            <td>${winner ? asPct(winner.win_league_pct) : "0%"}</td>
        </tr>
        `;
    }
    html += "</tbody></table>";
    winnerView.innerHTML = html;
}

async function preloadWinners() {
  const datasets = ["global", "mls", "extra"];

  for (const key of datasets) {
    try {
      const resp = await fetch(`/api/winners/${key}`);
      const data = await resp.json();

      if (resp.ok && data.ok) {
        winnerCache[key] = data.rows || [];
      }
    } catch (err) {
      console.error(`Failed to load winners ${key}`, err);
    }
  }

  renderWinners();
}

preloadHomeData();
preloadWinners();





//winnerDataset.addEventListener("change", renderWinners);
//preloadWinners();