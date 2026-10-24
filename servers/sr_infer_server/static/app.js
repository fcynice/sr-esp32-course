"use strict";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
  runs: [],
  run: null,
  checkpoint: null,
  browse: null,
  browsePath: "data",
  selected: null,
  result: null,
  jobId: null,
  busy: false,
  mode: "side", // side | line
  pair: "lr_sr",
  swapped: false,
  view: { z: 1, tx: 0, ty: 0 },
  line: { angle: 90, cx: null, cy: null },
  images: { lr: null, sr: null, hr: null },
  pollTimer: null,
  pan: null,
  lineDrag: null,
  layoutRaf: 0,
};

const el = {};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function imageUrl(path, variant = "full", download = false) {
  const params = new URLSearchParams({ path, variant });
  if (download) params.set("download", "1");
  return `/api/image?${params.toString()}`;
}

function lrTypeText(value) {
  if (value === "bicubic") return "bicubic（干净下采样）";
  if (value === "unknown") return "unknown（真实退化）";
  return value || "—";
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const data = await response.json();
      if (data && data.detail) {
        detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
      }
    } catch {
      // ignore non-JSON error bodies
    }
    throw new Error(detail);
  }
  return response.json();
}

// ---------------------------------------------------------------------------
// Small UI helpers
// ---------------------------------------------------------------------------

let toastTimer = null;

function toast(message, isError = false) {
  el.toast.textContent = message;
  el.toast.classList.toggle("error", Boolean(isError));
  el.toast.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.toast.classList.add("hidden"), 5200);
}

function showError(message) {
  el.errorText.textContent = message;
  el.errorText.classList.remove("hidden");
}

function hideError() {
  el.errorText.textContent = "";
  el.errorText.classList.add("hidden");
}

function showProgress(fraction, message) {
  el.progressWrap.classList.remove("hidden");
  const percent = Math.round(clamp(fraction, 0, 1) * 100);
  el.progressBar.style.width = `${percent}%`;
  el.progressText.textContent = message ? `${percent}% · ${message}` : `${percent}%`;
}

function hideProgress() {
  el.progressWrap.classList.add("hidden");
  el.progressBar.style.width = "0%";
  el.progressText.textContent = "";
}

function setBusy(busy) {
  state.busy = busy;
  el.runSelect.disabled = busy;
  el.ckptSelect.disabled = busy || !state.checkpoint;
  el.inferBtn.textContent = busy ? "推理中…" : "开始推理";
  updateInferButton();
}

function updateInferButton() {
  el.inferBtn.disabled = state.busy || !state.selected || !state.checkpoint;
}

// ---------------------------------------------------------------------------
// Health / runs / model info
// ---------------------------------------------------------------------------

async function loadHealth() {
  try {
    const health = await api("/api/health");
    state.browsePath = health.default_browse_path || "data";
    el.healthText.textContent = `本地服务 · ${health.runs} 个 run · 图片直接读取本地磁盘`;
    el.healthText.title = `项目根：${health.project_root}\n结果目录：${health.results_dir}`;
    el.devicePill.textContent = health.cuda ? `CUDA · ${health.gpu_name}` : "CPU";
    el.devicePill.title = health.cuda ? `GPU：${health.gpu_name}` : "CUDA 不可用，使用 CPU";
    el.devicePill.classList.toggle("cpu", !health.cuda);
  } catch (error) {
    el.healthText.textContent = "服务连接失败";
    el.devicePill.textContent = "离线";
    el.devicePill.classList.add("cpu");
    toast(error.message, true);
  }
}

async function loadRuns() {
  try {
    const data = await api("/api/runs");
    state.runs = data.runs || [];
    el.runCount.textContent = `${state.runs.length} 个 run`;
    el.runSelect.innerHTML = "";

    if (state.runs.length === 0) {
      const option = new Option("无可用 run", "");
      option.disabled = true;
      el.runSelect.appendChild(option);
      el.ckptSelect.innerHTML = "";
      const ckptOption = new Option("无可用权重", "");
      ckptOption.disabled = true;
      el.ckptSelect.appendChild(ckptOption);
      state.run = null;
      state.checkpoint = null;
      renderModelInfo();
      updateInferButton();
      return;
    }

    for (const run of state.runs) {
      const info = run.info || {};
      const option = new Option(
        `${run.name} · ${String(info.model || "?").toUpperCase()} · x${info.scale} · ${info.lr_type || "?"}`,
        run.name
      );
      el.runSelect.appendChild(option);
    }
    el.runSelect.value = state.runs[0].name;
    onRunChange();
  } catch (error) {
    el.runCount.textContent = "加载失败";
    toast(error.message, true);
  }
}

function currentRun() {
  const name = el.runSelect.value;
  return state.runs.find((run) => run.name === name) || null;
}

function onRunChange() {
  state.run = currentRun();
  state.checkpoint = null;
  el.ckptSelect.innerHTML = "";

  if (!state.run) {
    const option = new Option("无可用权重", "");
    option.disabled = true;
    el.ckptSelect.appendChild(option);
    renderModelInfo();
    setBusy(state.busy);
    clearResult();
    return;
  }

  const checkpoints = state.run.checkpoints || [];
  for (const checkpoint of checkpoints) {
    const parts = [checkpoint.label];
    if (checkpoint.epoch !== null && checkpoint.epoch !== undefined) {
      parts.push(`epoch ${checkpoint.epoch}`);
    }
    if (checkpoint.psnr !== null && checkpoint.psnr !== undefined) {
      parts.push(`${Number(checkpoint.psnr).toFixed(2)} dB`);
    }
    el.ckptSelect.appendChild(new Option(parts.join(" · "), checkpoint.name));
  }

  if (checkpoints.length === 0) {
    const option = new Option("无可用权重", "");
    option.disabled = true;
    el.ckptSelect.appendChild(option);
  } else {
    const preferred =
      checkpoints.find((checkpoint) => checkpoint.name === "best") || checkpoints[0];
    el.ckptSelect.value = preferred.name;
    state.checkpoint = preferred;
  }

  renderModelInfo();
  setBusy(state.busy);
  clearResult();
}

function onCheckpointChange() {
  if (!state.run) return;
  state.checkpoint =
    (state.run.checkpoints || []).find(
      (checkpoint) => checkpoint.name === el.ckptSelect.value
    ) || null;
  renderModelInfo();
  setBusy(state.busy);
  clearResult();
}

function addInfoItem(label, value) {
  if (value === null || value === undefined || value === "") return;
  const item = document.createElement("div");
  item.className = "info-item";

  const labelEl = document.createElement("span");
  labelEl.className = "info-label";
  labelEl.textContent = label;

  const valueEl = document.createElement("span");
  valueEl.className = "info-value";
  valueEl.textContent = String(value);
  valueEl.title = String(value);

  item.append(labelEl, valueEl);
  el.modelInfo.appendChild(item);
}

function renderModelInfo() {
  el.modelInfo.innerHTML = "";
  if (!state.run) {
    addInfoItem("状态", "未找到 runs/resNN");
    return;
  }

  const info = state.run.info || {};
  const log = state.run.log || {};

  addInfoItem("运行", state.run.name);  addInfoItem("架构", String(info.model || "?").toUpperCase());
  addInfoItem("训练倍数", info.scale ? `x${info.scale}` : null);
  addInfoItem("LR 类型", lrTypeText(info.lr_type));
  addInfoItem("结构", info.arch_text || null);
  addInfoItem(
    "训练块",
    info.patch_size ? `${info.patch_size}×${info.patch_size}` : null
  );
  addInfoItem("Batch", info.batch_size);
  addInfoItem("Epochs", info.epochs);
  addInfoItem(
    "学习率",
    info.lr !== null && info.lr !== undefined
      ? Number(info.lr).toExponential(1)
      : null
  );
  addInfoItem("随机种子", info.seed);
  addInfoItem(
    "最优验证 PSNR",
    log.best_psnr !== null && log.best_psnr !== undefined
      ? `${Number(log.best_psnr).toFixed(2)} dB${log.best_epoch ? ` @ epoch ${log.best_epoch}` : ""}`
      : null
  );
  addInfoItem(
    "已训练",
    log.last_epoch !== null && log.last_epoch !== undefined
      ? `${log.last_epoch} epoch`
      : null
  );
  addInfoItem(
    "数据增强",
    info.no_augment === null || info.no_augment === undefined
      ? null
      : info.no_augment
        ? "关闭"
        : "开启"
  );
  addInfoItem(
    "混合精度",
    info.no_amp === null || info.no_amp === undefined
      ? null
      : info.no_amp
        ? "关闭"
        : "开启"
  );
  addInfoItem("训练设备", info.train_device);

  if (state.checkpoint) {
    const checkpoint = state.checkpoint;
    const bits = [checkpoint.label];
    if (checkpoint.epoch !== null && checkpoint.epoch !== undefined) {
      bits.push(`epoch ${checkpoint.epoch}`);
    }
    if (checkpoint.psnr !== null && checkpoint.psnr !== undefined) {
      bits.push(`${Number(checkpoint.psnr).toFixed(2)} dB`);
    }
    addInfoItem("当前权重", bits.join(" · "));
  }
}

// ---------------------------------------------------------------------------
// Directory browser
// ---------------------------------------------------------------------------

async function browse(path, fallback = true) {
  el.browseLoading.classList.remove("hidden");
  try {
    const data = await api(`/api/browse?path=${encodeURIComponent(path ?? "")}`);
    state.browse = data;
    state.browsePath = data.path;
    renderBrowse();
  } catch (error) {
    if (fallback && path) {
      toast(`无法打开 ${path}：${error.message}`, true);
      el.browseLoading.classList.add("hidden");
      return browse("", false);
    }
    toast(error.message, true);
  } finally {
    el.browseLoading.classList.add("hidden");
  }
}

function renderBrowse() {
  const data = state.browse;
  if (!data) return;

  el.quickRoots.innerHTML = "";
  for (const root of data.quick_roots || []) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "quick-root" + (root.active ? " active" : "");
    button.textContent = root.name;
    button.title = root.path || "项目根";
    button.addEventListener("click", () => browse(root.path));
    el.quickRoots.appendChild(button);
  }

  el.breadcrumbs.innerHTML = "";
  (data.breadcrumbs || []).forEach((crumb, index) => {
    if (index > 0) {
      const separator = document.createElement("span");
      separator.className = "sep";
      separator.textContent = "›";
      el.breadcrumbs.appendChild(separator);
    }
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = crumb.name;
    button.title = crumb.path || "项目根";
    button.addEventListener("click", () => browse(crumb.path));
    el.breadcrumbs.appendChild(button);
  });

  el.dirList.innerHTML = "";
  for (const dir of data.dirs || []) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "dir-row";
    button.title = dir.path;

    const icon = document.createElement("span");
    icon.className = "folder";
    icon.textContent = "📁";

    const name = document.createElement("span");
    name.textContent = dir.name;

    const arrow = document.createElement("span");
    arrow.className = "arrow";
    arrow.textContent = "›";

    button.append(icon, name, arrow);
    button.addEventListener("click", () => browse(dir.path));
    el.dirList.appendChild(button);
  }

  el.imageGrid.innerHTML = "";
  for (const image of data.images || []) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "image-card";
    card.dataset.path = image.path;
    card.title = image.path;

    const img = document.createElement("img");
    img.src = image.thumb_url || imageUrl(image.path, "thumb");
    img.loading = "lazy";
    img.alt = image.name;

    const name = document.createElement("span");
    name.textContent = image.name;

    card.append(img, name);
    card.addEventListener("click", () => selectImage(image.path));
    el.imageGrid.appendChild(card);
  }

  el.browseEmpty.classList.toggle(
    "hidden",
    (data.images || []).length > 0 || (data.dirs || []).length > 0
  );
  highlightSelected();
}

function highlightSelected() {
  const selectedPath = state.selected ? state.selected.path : null;
  $$(".image-card", el.imageGrid).forEach((card) => {
    card.classList.toggle("selected", card.dataset.path === selectedPath);
  });
}

async function selectImage(path) {
  if (state.busy) {
    toast("推理进行中，请稍候再切换图片", true);
    return;
  }
  try {
    const info = await api(`/api/image-info?path=${encodeURIComponent(path)}`);
    state.selected = info;
    clearResult();
    renderSelected();
    highlightSelected();
  } catch (error) {
    toast(error.message, true);
  }
}

function renderSelected() {
  const info = state.selected;
  if (!info) {
    el.selectedPreview.innerHTML = "<span>从左侧目录选择一张图片</span>";
    el.selectedMeta.textContent = "尚未选择图片";
    updateInferButton();
    return;
  }

  el.selectedPreview.innerHTML = "";
  const img = document.createElement("img");
  img.src = imageUrl(info.path, "thumb");
  img.alt = info.name;
  img.title = info.path;
  el.selectedPreview.appendChild(img);

  const bits = [info.name, `${info.width}×${info.height}`];
  if (info.dataset) {
    const source = info.dataset.lr_type
      ? `${info.dataset.lr_type} · x${info.dataset.lr_scale}`
      : `x${info.dataset.lr_scale}`;
    bits.push(`已匹配 ${info.dataset.dataset} HR（${source}）`);
  }
  el.selectedMeta.textContent = bits.join(" · ");
  updateInferButton();
}

// ---------------------------------------------------------------------------
// Inference job
// ---------------------------------------------------------------------------

async function startInfer() {
  if (state.busy || !state.selected || !state.checkpoint) return;

  setBusy(true);
  clearResult();
  hideError();
  showProgress(0.01, "提交任务…");

  try {
    const response = await api("/api/infer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_path: state.selected.path,
        checkpoint_path: state.checkpoint.path,
      }),
    });
    state.jobId = response.job_id;
    pollJob();
  } catch (error) {
    finishError(error.message);
  }
}

async function pollJob() {
  if (!state.jobId) return;
  try {
    const job = await api(`/api/jobs/${encodeURIComponent(state.jobId)}`);
    showProgress(job.progress || 0, job.message || job.status);
    if (job.status === "success" && job.result) {
      finishSuccess(job.result);
    } else if (job.status === "success") {
      finishError("任务完成但缺少结果");
    } else if (job.status === "error") {
      finishError(job.error || "推理失败");
    } else {
      state.pollTimer = setTimeout(pollJob, 350);
    }
  } catch (error) {
    finishError(error.message);
  }
}

function finishSuccess(result) {
  clearTimeout(state.pollTimer);
  state.jobId = null;
  state.result = result;
  state.busy = false;
  setBusy(false);
  hideError();
  showProgress(1, "完成");
  setTimeout(() => {
    if (!state.busy) hideProgress();
  }, 900);

  const lrSource =
    result.dataset && result.dataset.lr_type ? result.dataset.lr_type : "";
  state.images.lr = {
    role: "lr",
    label: lrSource ? `LR 推理输入（${lrSource}）` : "LR 推理输入",
    path: result.lr_path,
    url: imageUrl(result.lr_path),
    thumb_url: imageUrl(result.lr_path, "thumb"),
    width: result.lr_width,
    height: result.lr_height,
  };
  state.images.sr = {
    role: "sr",
    label: "SR 推理输出",
    path: result.sr_path,
    url: imageUrl(result.sr_path),
    thumb_url: imageUrl(result.sr_path, "thumb"),
    width: result.sr_width,
    height: result.sr_height,
  };
  state.images.hr = result.hr
    ? {
        role: "hr",
        label: "HR 高清真值",
        path: result.hr.path,
        url: imageUrl(result.hr.path),
        thumb_url: imageUrl(result.hr.path, "thumb"),
        width: result.hr.width,
        height: result.hr.height,
      }
    : null;

  state.pair = "lr_sr";
  state.swapped = false;
  resetViewState();
  updateResultInfo();
  renderComparison({ resetView: true });

  el.downloadBtn.href = imageUrl(result.sr_path, "full", true);
  el.downloadBtn.download = result.sr_name || "sr.png";
  el.downloadBtn.classList.remove("hidden");

  toast("推理完成");
}

function finishError(message) {
  clearTimeout(state.pollTimer);
  state.jobId = null;
  state.busy = false;
  setBusy(false);
  hideProgress();
  showError(message || "推理失败");
  toast(message || "推理失败", true);
}

// ---------------------------------------------------------------------------
// Comparison state / rendering
// ---------------------------------------------------------------------------

function resetViewState(resetAngle = true) {
  state.view = { z: 1, tx: 0, ty: 0 };
  if (resetAngle) {
    state.line.angle = 90;
    el.angleSlider.value = "90";
    el.angleValue.textContent = "90°";
  }
  state.line.cx = null;
  state.line.cy = null;
}

function clearResult() {
  state.result = null;
  state.images = { lr: null, sr: null, hr: null };
  state.pair = "lr_sr";
  state.swapped = false;
  el.resultInfo.innerHTML = "";
  el.resultInfo.classList.add("hidden");
  el.metricText.textContent = "";
  el.refStrip.innerHTML = "";
  el.refStrip.classList.add("hidden");
  el.downloadBtn.classList.add("hidden");
  el.downloadBtn.removeAttribute("href");
  el.downloadBtn.removeAttribute("download");
  hideError();
  hideProgress();
  resetViewState();
  renderComparison();
}

function hasHr() {
  return Boolean(state.images.hr);
}

function updatePairAvailability() {
  const haveResult = Boolean(state.images.sr);
  for (const button of $$("#pairSwitch button")) {
    const pair = button.dataset.pair;
    let enabled = haveResult;
    if (pair === "sr_hr" || pair === "lr_hr") {
      enabled = haveResult && hasHr();
    }
    button.disabled = !enabled;
  }

  if (
    (state.pair === "sr_hr" || state.pair === "lr_hr") &&
    !hasHr()
  ) {
    state.pair = "lr_sr";
  }
  if (!haveResult) state.pair = "lr_sr";

  for (const button of $$("#pairSwitch button")) {
    button.classList.toggle("active", button.dataset.pair === state.pair);
  }
  el.swapPairBtn.disabled = !haveResult;
}

function getPairData() {
  let pair;
  if (state.pair === "sr_hr") {
    pair = [state.images.sr, state.images.hr];
  } else if (state.pair === "lr_hr") {
    pair = [state.images.lr, state.images.hr];
  } else {
    pair = [state.images.lr, state.images.sr];
  }
  if (!pair[0] || !pair[1]) return null;
  return state.swapped ? [pair[1], pair[0]] : pair;
}

function renderComparison({ resetView = false } = {}) {
  updatePairAvailability();
  const pair = getPairData();

  if (resetView) resetViewState(false);

  const havePair = Boolean(pair);
  el.emptyStage.classList.toggle("hidden", havePair);
  const sideVisible = havePair && state.mode === "side";
  const lineVisible = havePair && state.mode === "line";
  el.sideMode.classList.toggle("hidden", !sideVisible);
  el.lineMode.classList.toggle("hidden", !lineVisible);
  el.angleRow.classList.toggle("hidden", !lineVisible);

  if (!havePair) {
    el.refStrip.classList.add("hidden");
    return;
  }

  if (sideVisible) {
    renderSide(pair);
  } else {
    renderLine(pair);
  }
  renderRefStrip();
}

function setImage(img, url) {
  if (img.dataset.url === url) return;
  img.dataset.url = url;
  img.src = url;
}

function renderSide(pair) {
  const [a, b] = pair;
  el.sideLabelA.textContent = a.label;
  el.sideLabelB.textContent = b.label;
  el.sideDimA.textContent = `${a.width}×${a.height}`;
  el.sideDimB.textContent = `${b.width}×${b.height}`;

  setImage($("img", el.sideCanvasA), a.url);
  setImage($("img", el.sideCanvasB), b.url);
  layoutStage();
}

function renderLine(pair) {
  const [a, b] = pair;
  el.lineLabelA.textContent = a.label;
  el.lineLabelB.textContent = b.label;
  el.lineTagA.textContent = a.label;
  el.lineTagB.textContent = b.label;

  setImage(el.lineImgA, a.url);
  setImage(el.lineImgB, b.url);
  layoutStage();
}

function renderRefStrip() {
  const items = [state.images.lr, state.images.hr, state.images.sr].filter(Boolean);
  if (items.length < 2) {
    el.refStrip.classList.add("hidden");
    return;
  }

  const pair = getPairData() || [];
  const activeRoles = new Set(pair.map((item) => item.role));

  el.refStrip.innerHTML = "";
  for (const item of items) {
    const card = document.createElement("div");
    card.className = "ref-card" + (activeRoles.has(item.role) ? " active" : "");

    const img = document.createElement("img");
    img.src = item.thumb_url;
    img.loading = "lazy";
    img.alt = item.label;

    const box = document.createElement("div");
    const label = document.createElement("strong");
    label.textContent = item.label;
    const dim = document.createElement("span");
    dim.textContent = `${item.width}×${item.height}`;
    box.append(label, dim);

    card.append(img, box);
    el.refStrip.appendChild(card);
  }
  el.refStrip.classList.remove("hidden");
}

function updateResultInfo() {
  const result = state.result;
  el.resultInfo.innerHTML = "";
  if (!result) {
    el.resultInfo.classList.add("hidden");
    el.metricText.textContent = "";
    return;
  }

  el.resultInfo.classList.remove("hidden");

  const appendBit = (label, value) => {
    const span = document.createElement("span");
    const strong = document.createElement("strong");
    strong.textContent = label;
    span.append(strong, document.createTextNode(` ${value}`));
    el.resultInfo.appendChild(span);
  };

  appendBit("SR", `${result.sr_width}×${result.sr_height}`);
  appendBit(
    "模型",
    `${String(result.model || "").toUpperCase()} · x${result.scale} · ${result.run}/${result.checkpoint}.pth`
  );
  if (result.dataset) {
    appendBit(
      "LR 来源",
      `${result.dataset.dataset} · ${result.dataset.lr_type || "?"} · x${result.dataset.lr_scale}`
    );
  }
  appendBit("用时", `${Number(result.elapsed_seconds || 0).toFixed(2)} s`);

  if (result.metrics) {
    appendBit("PSNR", `${Number(result.metrics.psnr).toFixed(2)} dB`);
    appendBit("SSIM", Number(result.metrics.ssim).toFixed(4));
    el.metricText.textContent = `PSNR ${Number(result.metrics.psnr).toFixed(2)} dB · SSIM ${Number(result.metrics.ssim).toFixed(4)}`;
  } else {
    el.metricText.textContent = "";
    if (result.metrics_note) {
      appendBit("提示", result.metrics_note);
    }
  }
}

// ---------------------------------------------------------------------------
// Canvas fit / zoom / pan
// ---------------------------------------------------------------------------

function fitSize(viewport, naturalWidth, naturalHeight) {
  const padding = 20;
  const availableWidth = Math.max(1, viewport.clientWidth - padding * 2);
  const availableHeight = Math.max(1, viewport.clientHeight - padding * 2);
  if (!naturalWidth || !naturalHeight) {
    return { w: availableWidth, h: availableHeight };
  }
  const scale = Math.min(
    availableWidth / naturalWidth,
    availableHeight / naturalHeight
  );
  return {
    w: Math.max(1, naturalWidth * scale),
    h: Math.max(1, naturalHeight * scale),
  };
}

function setCanvasSize(canvas, width, height) {
  canvas.dataset.w = String(width);
  canvas.dataset.h = String(height);
}

function applyCanvasTransform(canvas, viewport) {
  const width = Number(canvas.dataset.w);
  const height = Number(canvas.dataset.h);
  if (!width || !height) return;

  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  canvas.style.left = `${(viewport.clientWidth - width) / 2}px`;
  canvas.style.top = `${(viewport.clientHeight - height) / 2}px`;
  canvas.style.transform = `translate(${state.view.tx}px, ${state.view.ty}px) scale(${state.view.z})`;
}

function applyAllTransforms() {
  if (state.mode === "side") {
    applyCanvasTransform(el.sideCanvasA, el.sideViewportA);
    applyCanvasTransform(el.sideCanvasB, el.sideViewportB);
  } else {
    applyCanvasTransform(el.lineCanvasA, el.lineViewport);
    applyCanvasTransform(el.lineCanvasB, el.lineViewport);
  }
}

function layoutStage() {
  const pair = getPairData();
  if (!pair) return;

  if (state.mode === "side") {
    const [a, b] = pair;
    const fitA = fitSize(el.sideViewportA, a.width, a.height);
    const fitB = fitSize(el.sideViewportB, b.width, b.height);
    setCanvasSize(el.sideCanvasA, fitA.w, fitA.h);
    setCanvasSize(el.sideCanvasB, fitB.w, fitB.h);
    applyCanvasTransform(el.sideCanvasA, el.sideViewportA);
    applyCanvasTransform(el.sideCanvasB, el.sideViewportB);
  } else {
    const [a, b] = pair;
    const reference =
      a.width * a.height >= b.width * b.height ? a : b;
    const fit = fitSize(el.lineViewport, reference.width, reference.height);
    setCanvasSize(el.lineCanvasA, fit.w, fit.h);
    setCanvasSize(el.lineCanvasB, fit.w, fit.h);
    applyCanvasTransform(el.lineCanvasA, el.lineViewport);
    applyCanvasTransform(el.lineCanvasB, el.lineViewport);
    applyLineVisuals();
  }
}

function scheduleLayout() {
  if (state.layoutRaf) return;
  state.layoutRaf = requestAnimationFrame(() => {
    state.layoutRaf = 0;
    if (getPairData()) layoutStage();
  });
}

// ---------------------------------------------------------------------------
// Draggable line
// ---------------------------------------------------------------------------

function clipHalfPlane(width, height, cx, cy, nx, ny) {
  const corners = [
    [0, 0],
    [width, 0],
    [width, height],
    [0, height],
  ];
  const output = [];
  for (let index = 0; index < corners.length; index += 1) {
    const a = corners[index];
    const b = corners[(index + 1) % corners.length];
    const sa = (a[0] - cx) * nx + (a[1] - cy) * ny;
    const sb = (b[0] - cx) * nx + (b[1] - cy) * ny;
    if (sa >= 0) output.push(a);
    if ((sa >= 0) !== (sb >= 0)) {
      const t = sa / (sa - sb);
      output.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
    }
  }
  return output;
}

function positionTag(tag, x, y, width, height) {
  tag.style.left = `${clamp(x, 54, Math.max(54, width - 54))}px`;
  tag.style.top = `${clamp(y, 22, Math.max(22, height - 22))}px`;
}

function applyLineVisuals() {
  const viewport = el.lineViewport;
  const width = viewport.clientWidth;
  const height = viewport.clientHeight;
  if (!width || !height) return;

  if (state.line.cx === null || state.line.cy === null) {
    state.line.cx = width / 2;
    state.line.cy = height / 2;
  }
  state.line.cx = clamp(state.line.cx, 0, width);
  state.line.cy = clamp(state.line.cy, 0, height);

  const radians = (state.line.angle * Math.PI) / 180;
  // The line direction is (cos, sin); the normal points to the "after" side.
  const nx = Math.sin(radians);
  const ny = -Math.cos(radians);

  const polygon = clipHalfPlane(
    width,
    height,
    state.line.cx,
    state.line.cy,
    nx,
    ny
  );
  el.clipLayer.style.clipPath =
    polygon.length >= 3
      ? `polygon(${polygon
          .map((point) => `${point[0].toFixed(2)}px ${point[1].toFixed(2)}px`)
          .join(", ")})`
      : "none";

  const dx = Math.cos(radians);
  const dy = Math.sin(radians);
  const extent = Math.hypot(width, height) * 1.5;
  const x1 = (state.line.cx - dx * extent).toFixed(2);
  const y1 = (state.line.cy - dy * extent).toFixed(2);
  const x2 = (state.line.cx + dx * extent).toFixed(2);
  const y2 = (state.line.cy + dy * extent).toFixed(2);
  el.dividerHit.setAttribute("x1", x1);
  el.dividerHit.setAttribute("y1", y1);
  el.dividerHit.setAttribute("x2", x2);
  el.dividerHit.setAttribute("y2", y2);
  el.dividerLine.setAttribute(
    "x1",
    (state.line.cx - dx * extent).toFixed(2)
  );
  el.dividerLine.setAttribute(
    "y1",
    (state.line.cy - dy * extent).toFixed(2)
  );
  el.dividerLine.setAttribute(
    "x2",
    (state.line.cx + dx * extent).toFixed(2)
  );
  el.dividerLine.setAttribute(
    "y2",
    (state.line.cy + dy * extent).toFixed(2)
  );
  el.dividerHandle.setAttribute("cx", state.line.cx.toFixed(2));
  el.dividerHandle.setAttribute("cy", state.line.cy.toFixed(2));

  const offset = clamp(Math.min(width, height) * 0.18, 54, 120);
  positionTag(
    el.lineTagB,
    state.line.cx + nx * offset,
    state.line.cy + ny * offset,
    width,
    height
  );
  positionTag(
    el.lineTagA,
    state.line.cx - nx * offset,
    state.line.cy - ny * offset,
    width,
    height
  );
}

// ---------------------------------------------------------------------------
// Pointer / wheel interactions
// ---------------------------------------------------------------------------

function onWheel(event) {
  if (!getPairData()) return;
  event.preventDefault();

  const rect = event.currentTarget.getBoundingClientRect();
  const cursorX = event.clientX - rect.left - rect.width / 2;
  const cursorY = event.clientY - rect.top - rect.height / 2;
  const oldZoom = state.view.z;
  const factor = Math.exp(-event.deltaY * 0.0015);
  const newZoom = clamp(oldZoom * factor, 0.05, 20);
  if (newZoom === oldZoom) return;

  state.view.tx = cursorX - (cursorX - state.view.tx) * (newZoom / oldZoom);
  state.view.ty = cursorY - (cursorY - state.view.ty) * (newZoom / oldZoom);
  state.view.z = newZoom;
  applyAllTransforms();
}

function onPointerDown(event) {
  if (!getPairData() || event.button !== 0) return;
  if (event.target.closest("#dividerHandle, #dividerHit")) return;

  const viewport = event.currentTarget;
  viewport.setPointerCapture(event.pointerId);
  state.pan = {
    id: event.pointerId,
    x: event.clientX,
    y: event.clientY,
  };
  viewport.classList.add("dragging");
}

function onPointerMove(event) {
  if (!state.pan || state.pan.id !== event.pointerId) return;
  const dx = event.clientX - state.pan.x;
  const dy = event.clientY - state.pan.y;
  state.pan.x = event.clientX;
  state.pan.y = event.clientY;
  state.view.tx += dx;
  state.view.ty += dy;
  applyAllTransforms();
}

function onPointerUp(event) {
  if (!state.pan || state.pan.id !== event.pointerId) return;
  const viewport = event.currentTarget;
  try {
    if (viewport.hasPointerCapture(event.pointerId)) {
      viewport.releasePointerCapture(event.pointerId);
    }
  } catch {
    // ignore
  }
  viewport.classList.remove("dragging");
  state.pan = null;
}

function bindViewport(viewport) {
  viewport.addEventListener("wheel", onWheel, { passive: false });
  viewport.addEventListener("pointerdown", onPointerDown);
  viewport.addEventListener("pointermove", onPointerMove);
  viewport.addEventListener("pointerup", onPointerUp);
  viewport.addEventListener("pointercancel", onPointerUp);
}

function bindLineDragTarget(target) {
  target.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    event.stopPropagation();
    target.setPointerCapture(event.pointerId);
    state.lineDrag = { id: event.pointerId, target };
  });

  target.addEventListener("pointermove", (event) => {
    if (!state.lineDrag || state.lineDrag.id !== event.pointerId) return;
    const rect = el.lineViewport.getBoundingClientRect();
    state.line.cx = clamp(event.clientX - rect.left, 0, rect.width);
    state.line.cy = clamp(event.clientY - rect.top, 0, rect.height);
    applyLineVisuals();
  });

  const endDrag = (event) => {
    if (!state.lineDrag || state.lineDrag.id !== event.pointerId) return;
    try {
      if (target.hasPointerCapture(event.pointerId)) {
        target.releasePointerCapture(event.pointerId);
      }
    } catch {
      // ignore
    }
    state.lineDrag = null;
  };

  target.addEventListener("pointerup", endDrag);
  target.addEventListener("pointercancel", endDrag);
}

function bindLineHandle() {
  bindLineDragTarget(el.dividerHandle);
  bindLineDragTarget(el.dividerHit);
}

// ---------------------------------------------------------------------------
// Wire-up
// ---------------------------------------------------------------------------

function cacheElements() {
  const ids = [
    "healthText",
    "devicePill",
    "runCount",
    "runSelect",
    "ckptSelect",
    "modelInfo",
    "selectedMeta",
    "selectedPreview",
    "inferBtn",
    "downloadBtn",
    "progressWrap",
    "progressBar",
    "progressText",
    "errorText",
    "metricText",
    "resultInfo",
    "refStrip",
    "angleRow",
    "angleSlider",
    "angleValue",
    "stageWrap",
    "emptyStage",
    "sideMode",
    "sideLabelA",
    "sideDimA",
    "sideLabelB",
    "sideDimB",
    "sideViewportA",
    "sideCanvasA",
    "sideViewportB",
    "sideCanvasB",
    "lineMode",
    "lineLabelA",
    "lineLabelB",
    "lineViewport",
    "lineCanvasA",
    "lineImgA",
    "lineCanvasB",
    "lineImgB",
    "clipLayer",
    "dividerHit",
    "dividerLine",
    "dividerHandle",
    "lineTagA",
    "lineTagB",
    "quickRoots",
    "breadcrumbs",
    "browseLoading",
    "dirList",
    "imageGrid",
    "browseEmpty",
    "refreshBrowse",
    "swapPairBtn",
    "resetViewBtn",
    "pairSwitch",
    "modeSwitch",
    "toast",
  ];
  for (const id of ids) {
    el[id] = document.getElementById(id);
  }
}

function bindUI() {
  el.runSelect.addEventListener("change", onRunChange);
  el.ckptSelect.addEventListener("change", onCheckpointChange);
  el.inferBtn.addEventListener("click", startInfer);
  el.refreshBrowse.addEventListener("click", () => browse(state.browsePath));

  for (const button of $$("#modeSwitch button")) {
    button.addEventListener("click", () => {
      const mode = button.dataset.mode;
      if (state.mode === mode) return;
      state.mode = mode;
      for (const candidate of $$("#modeSwitch button")) {
        candidate.classList.toggle("active", candidate === button);
      }
      state.line.cx = null;
      state.line.cy = null;
      renderComparison({ resetView: true });
    });
  }

  for (const button of $$("#pairSwitch button")) {
    button.addEventListener("click", () => {
      if (button.disabled) return;
      state.pair = button.dataset.pair;
      state.swapped = false;
      renderComparison({ resetView: true });
    });
  }

  el.swapPairBtn.addEventListener("click", () => {
    if (!getPairData()) return;
    state.swapped = !state.swapped;
    renderComparison({ resetView: true });
  });

  el.resetViewBtn.addEventListener("click", () => {
    resetViewState(false);
    renderComparison();
  });

  el.angleSlider.addEventListener("input", () => {
    state.line.angle = Number(el.angleSlider.value);
    el.angleValue.textContent = `${state.line.angle}°`;
    applyLineVisuals();
  });

  bindViewport(el.sideViewportA);
  bindViewport(el.sideViewportB);
  bindViewport(el.lineViewport);
  bindLineHandle();

  window.addEventListener("resize", scheduleLayout);
  if (typeof ResizeObserver !== "undefined") {
    new ResizeObserver(scheduleLayout).observe(el.stageWrap);
  }
}

async function init() {
  cacheElements();
  bindUI();
  await loadHealth();
  await loadRuns();
  await browse(state.browsePath || "data");
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
