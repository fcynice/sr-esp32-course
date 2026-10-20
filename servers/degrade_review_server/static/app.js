"use strict";

const state = {
  sessions: [],
  session: null,
  records: [],
  filtered: [],
  profiles: new Set(),
  sources: new Set(),
  query: "",
  view: "review",
  reference: [],
  currentIndex: -1,
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const data = await res.json();
      if (data.detail) detail = data.detail;
    } catch (_) { /* ignore */ }
    throw new Error(detail);
  }
  return res.json();
}

function fmt(value, digits = 2) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : "—";
}

function recordName(record) {
  const source = String(record.source || "");
  const base = source.split("/").pop() || record.id || "?";
  return base;
}

function sourceKey(record) {
  return `${record.source_group || "?"}/${record.split || "?"}`;
}

function init() {
  $("#sessionSelect").addEventListener("change", (event) => loadSession(event.target.value));
  $("#searchInput").addEventListener("input", (event) => {
    state.query = event.target.value.trim().toLowerCase();
    applyFilters();
  });
  $$(".tab").forEach((button) => {
    button.addEventListener("click", () => setView(button.dataset.view));
  });
  document.addEventListener("keydown", onKeyDown);
  $$("[data-close]").forEach((el) => el.addEventListener("click", closeModal));
  $("#prevBtn").addEventListener("click", () => navigate(-1));
  $("#nextBtn").addEventListener("click", () => navigate(1));
  bootstrap();
}

async function bootstrap() {
  try {
    const [health, sessions, reference] = await Promise.all([
      fetchJSON("/api/health"),
      fetchJSON("/api/sessions"),
      fetchJSON("/api/reference"),
    ]);
    $("#health").textContent = `${health.sessions} sessions · ${health.reference_dir.split("/").pop()}`;
    state.reference = reference.images || [];
    state.sessions = sessions.sessions || [];
    $("#versionBadge").textContent = state.sessions[0]?.version || "v1";
    const select = $("#sessionSelect");
    select.innerHTML = state.sessions
      .map((s) => {
        const label = `${s.id} · ${s.version || "?"} · ${s.record_count ?? "?"} 张`;
        return `<option value="${escapeHtml(s.id)}">${escapeHtml(label)}</option>`;
      })
      .join("");
    if (state.sessions.length) {
      await loadSession(state.sessions[0].id);
    } else {
      $("#sessionMeta").textContent = "审查目录中还没有 session。请先运行 python -m superresolution.degrade";
      renderEmpty("没有可审查的 session");
    }
  } catch (error) {
    $("#sessionMeta").textContent = `加载失败：${error.message}`;
    console.error(error);
  }
}

async function loadSession(sessionId) {
  try {
    const data = await fetchJSON(`/api/session/${encodeURIComponent(sessionId)}`);
    state.session = data;
    state.records = data.records || [];
    state.profiles.clear();
    state.sources.clear();
    state.query = "";
    $("#searchInput").value = "";
    renderSessionMeta();
    renderFilterChips();
    applyFilters();
  } catch (error) {
    $("#sessionMeta").textContent = `加载 session 失败：${error.message}`;
  }
}

function renderSessionMeta() {
  const { header, summary } = state.session;
  const counts = state.session.source_counts || {};
  const sourceText = Object.entries(counts)
    .map(([key, value]) => `${escapeHtml(key)}：<b>${value}</b>`)
    .join("<br>");
  const fp = header.fingerprint ? String(header.fingerprint).slice(0, 12) : "—";
  $("#sessionMeta").innerHTML = `
    <div>版本 <b>${escapeHtml(header.version)}</b> · scale <b>${escapeHtml(header.scale)}</b> ·
      severity <b>${escapeHtml(header.severity)}</b> · seed <b>${escapeHtml(header.seed)}</b></div>
    <div style="margin-top:4px">代码指纹 <b title="${escapeHtml(header.fingerprint || '')}">${escapeHtml(fp)}</b></div>
    <div style="margin-top:6px">${sourceText}</div>
    <div style="margin-top:6px">共 <b>${state.records.length}</b> 张审查图</div>
  `;
  $("#versionBadge").textContent = header.version || "v1";

  const contact = state.session.preview_urls?.contact_sheet;
  const refSheet = state.session.preview_urls?.reference_sheet;
  const link = $("#contactLink");
  if (contact) {
    link.hidden = false;
    link.href = contact;
  } else {
    link.hidden = true;
  }
  const refLink = $("#referenceLink");
  if (refSheet) {
    refLink.hidden = false;
    refLink.href = refSheet;
  } else {
    refLink.hidden = true;
  }

  const overall = summary?.overall || {};
  const reference = summary?.reference?.stats || {};
  const rows = [
    ["green_cast", "绿偏 G-(R+B)/2", 2],
    ["saturation", "饱和度", 2],
    ["noise_sigma", "噪声 σ", 3],
    ["laplacian_var", "清晰度 LapVar", 1],
    ["y_mean", "亮度 Y", 1],
  ];
  $("#statsTable").innerHTML = rows
    .map(([key, label, digits]) => {
      const sim = overall[key]?.median;
      const ref = reference[key]?.median;
      return `<tr><td>${label}</td><td>${fmt(sim, digits)} / <span class="muted">${fmt(ref, digits)}</span></td></tr>`;
    })
    .join("") + `<tr><td colspan="2" class="muted">格式：模拟 / 真实参考（中位数）</td></tr>`;
}

function makeChip(label, count, active, onClick) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `chip${active ? " active" : ""}`;
  button.innerHTML = `${escapeHtml(label)}${count !== null ? ` <span class="count">${count}</span>` : ""}`;
  button.addEventListener("click", onClick);
  return button;
}

function renderFilterChips() {
  const profileBox = $("#profileFilters");
  const sourceBox = $("#sourceFilters");
  profileBox.innerHTML = "";
  sourceBox.innerHTML = "";

  const profiles = state.session.profile_counts || {};
  profileBox.appendChild(
    makeChip("全部", null, state.profiles.size === 0, () => {
      state.profiles.clear();
      renderFilterChips();
      applyFilters();
    }),
  );
  Object.entries(profiles)
    .sort((a, b) => a[0].localeCompare(b[0]))
    .forEach(([name, count]) => {
      const active = state.profiles.has(name);
      profileBox.appendChild(
        makeChip(name, count, active, () => {
          if (state.profiles.has(name)) state.profiles.delete(name);
          else state.profiles.add(name);
          renderFilterChips();
          applyFilters();
        }),
      );
    });

  const sources = state.session.source_counts || {};
  sourceBox.appendChild(
    makeChip("全部", null, state.sources.size === 0, () => {
      state.sources.clear();
      renderFilterChips();
      applyFilters();
    }),
  );
  Object.entries(sources)
    .sort((a, b) => a[0].localeCompare(b[0]))
    .forEach(([key, count]) => {
      const active = state.sources.has(key);
      sourceBox.appendChild(
        makeChip(key, count, active, () => {
          if (state.sources.has(key)) state.sources.delete(key);
          else state.sources.add(key);
          renderFilterChips();
          applyFilters();
        }),
      );
    });
}

function applyFilters() {
  if (!state.session) return;
  state.filtered = state.records.filter((record) => {
    if (state.profiles.size && !state.profiles.has(record.profile)) return false;
    if (state.sources.size && !state.sources.has(sourceKey(record))) return false;
    if (state.query) {
      const haystack = [
        record.source,
        record.id,
        record.profile,
        record.seed,
        sourceKey(record),
      ]
        .join(" ")
        .toLowerCase();
      if (!haystack.includes(state.query)) return false;
    }
    return true;
  });
  $("#visibleCount").textContent = `显示 ${state.filtered.length} / ${state.records.length} 张`;
  renderReviewGrid();
}

function renderReviewGrid() {
  const grid = $("#reviewGrid");
  grid.hidden = state.view !== "review";
  if (!state.filtered.length) {
    renderEmpty("当前筛选没有结果");
    return;
  }
  grid.innerHTML = state.filtered
    .map((record, index) => {
      const stats = record.stats || {};
      return `
        <article class="card" data-index="${index}">
          <img class="card-image" loading="lazy" src="${escapeHtml(record.lr_thumb_url)}"
               alt="${escapeHtml(record.profile)}" />
          <div class="card-body">
            <div class="card-title">
              <span class="card-name" title="${escapeHtml(record.source)}">${escapeHtml(recordName(record))}</span>
              <span class="badge profile">${escapeHtml(record.profile)}</span>
            </div>
            <div class="card-meta">
              <span class="badge muted">${escapeHtml(sourceKey(record))}</span>
              <span class="badge muted">G ${fmt(stats.green_cast, 1)}</span>
              <span class="badge muted">σ ${fmt(stats.noise_sigma, 3)}</span>
            </div>
          </div>
        </article>`;
    })
    .join("");
  grid.querySelectorAll(".card").forEach((card) => {
    card.addEventListener("click", () => openModal(Number(card.dataset.index)));
  });
}

function renderEmpty(text) {
  const grid = $("#reviewGrid");
  grid.innerHTML = `<div class="empty">${escapeHtml(text)}</div>`;
}

function setView(view) {
  state.view = view;
  $$(".tab").forEach((button) => button.classList.toggle("active", button.dataset.view === view));
  const isReview = view === "review";
  $("#reviewGrid").hidden = !isReview;
  $("#referenceGrid").hidden = isReview;
  $("#viewHint").textContent = isReview
    ? "点击卡片查看 HR / 模拟 LR 并排对比"
    : "data01 中的真实 ESP32 摄像头图片，用于画质对照";
  if (!isReview) renderReferenceGrid();
}

function referenceSample(limit = 12) {
  if (!state.reference.length) return [];
  const step = Math.max(1, Math.floor(state.reference.length / limit));
  const out = [];
  for (let i = 0; i < state.reference.length && out.length < limit; i += step) {
    out.push(state.reference[i]);
  }
  return out;
}

function renderReferenceGrid() {
  const grid = $("#referenceGrid");
  if (!state.reference.length) {
    grid.innerHTML = `<div class="empty">没有找到真实 ESP32 参考图</div>`;
    return;
  }
  grid.innerHTML = state.reference
    .map((item) => `
      <article class="card" data-url="${escapeHtml(item.url)}">
        <img class="card-image" loading="lazy" src="${escapeHtml(item.thumb_url)}" alt="${escapeHtml(item.name)}" />
        <div class="card-body">
          <div class="card-title"><span class="card-name">${escapeHtml(item.name)}</span></div>
        </div>
      </article>`)
    .join("");
  grid.querySelectorAll(".card").forEach((card) => {
    card.addEventListener("click", () => window.open(card.dataset.url, "_blank", "noreferrer"));
  });
}

function openModal(index) {
  if (index < 0 || index >= state.filtered.length) return;
  state.currentIndex = index;
  fillModal(state.filtered[index]);
  $("#modal").hidden = false;
  document.body.style.overflow = "hidden";
}

function fillModal(record) {
  const stats = record.stats || {};
  $("#modalTitle").textContent = `${recordName(record)} · ${record.profile}`;
  $("#modalSub").textContent =
    `${sourceKey(record)} · HR ${record.hr_size?.join("×")} → LR ${record.lr_size?.join("×")} · ` +
    `seed ${record.seed} · ${record.version}`;

  const hr = $("#modalHr");
  const lr = $("#modalLr");
  if (record.hr_available) {
    hr.src = record.hr_url;
    hr.style.opacity = "1";
    hr.alt = "HR source";
  } else {
    hr.removeAttribute("src");
    hr.alt = "HR source unavailable";
    hr.style.opacity = "0.35";
  }
  lr.src = record.lr_url;

  const chips = [
    ["profile", record.profile, "profile"],
    ["version", record.version, "muted"],
    ["scale", `x${record.scale}`, "muted"],
    ["seed", record.seed, "muted"],
    ["green", fmt(stats.green_cast, 2), "cast"],
    ["sat", fmt(stats.saturation, 2), "muted"],
    ["noise", fmt(stats.noise_sigma, 3), "muted"],
    ["lapvar", fmt(stats.laplacian_var, 1), "muted"],
    ["jpeg", record.spec?.jpeg_quality ?? "—", "muted"],
  ];
  $("#modalChips").innerHTML = chips
    .map(([label, value, cls]) => `<span class="badge ${cls}">${escapeHtml(label)} ${escapeHtml(value)}</span>`)
    .join("");

  $("#modalSpec").textContent = JSON.stringify(record.spec || {}, null, 2);

  const strip = $("#modalReference");
  strip.innerHTML = referenceSample(12)
    .map((item) => `<a href="${escapeHtml(item.url)}" target="_blank" rel="noreferrer" title="${escapeHtml(item.name)}">
        <img loading="lazy" src="${escapeHtml(item.thumb_url)}" alt="ESP32 reference" />
      </a>`)
    .join("");

  $("#modalIndex").textContent = `${state.currentIndex + 1} / ${state.filtered.length}`;
  $("#prevBtn").disabled = state.currentIndex <= 0;
  $("#nextBtn").disabled = state.currentIndex >= state.filtered.length - 1;
}

function closeModal() {
  $("#modal").hidden = true;
  document.body.style.overflow = "";
  $("#modalHr").removeAttribute("src");
  $("#modalLr").removeAttribute("src");
}

function navigate(delta) {
  const next = state.currentIndex + delta;
  if (next < 0 || next >= state.filtered.length) return;
  openModal(next);
}

function onKeyDown(event) {
  if ($("#modal").hidden) return;
  if (event.key === "Escape") closeModal();
  if (event.key === "ArrowLeft") navigate(-1);
  if (event.key === "ArrowRight") navigate(1);
}

init();
