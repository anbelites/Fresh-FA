const API = "";

/** Человекочитаемые подписи SER; uncertain — после калибровки «плоского» softmax. */
const SER_LABEL_DISPLAY = {
  uncertain: "неопределённо",
  neutral: "нейтральный",
  angry: "раздражение",
  positive: "позитив",
  sad: "грусть",
  other: "прочее",
  Angry: "раздражение",
  Disgusted: "отвращение",
  Happy: "радость",
  Neutral: "нейтральный",
  Sad: "грусть",
  Scared: "страх",
  Surprised: "удивление",
};

const STAGE_LABELS = {
  queued: "В очереди",
  starting: "Запуск пайплайна",
  extract_audio: "1/7 — извлечение WAV из видео",
  diarization: "2/7 — диаризация (pyannote: кто когда говорит)",
  diarization_skip: "2/7 — без HF: диаризация будет грубее (MFCC)",
  whisper_load: "3/7 — загрузка модели Whisper",
  asr_whisper: "4/7 — распознавание речи (Whisper, обычно самый долгий шаг)",
  segments_build: "5/7 — спикеры, слова, скорость речи, тон по кускам",
  transcribing: "Транскрибация (устар.)",
  tone: "6/7 — эмоции по аудио (SER)",
  evaluating: "7/7 — оценка по чеклисту (ИИ)",
  done: "Готово",
  error: "Ошибка",
};

/** Подписи для задачи «только пересчёт оценки» (не шаг 7/7 полного пайплайна). */
const EVAL_ONLY_STAGE_LABELS = {
  queued: "В очереди",
  eval_prep: "Подготовка к оценке по чеклисту…",
  eval_waiting: "Ожидание: выполняется другая обработка…",
  evaluating: "Запрос к модели оценки (ИИ)…",
  done: "Готово",
  error: "Ошибка",
};

function jobStageLabel(job) {
  if (!job) return "…";
  const stage = job.stage;
  if (job.kind === "eval_only") {
    return EVAL_ONLY_STAGE_LABELS[stage] || STAGE_LABELS[stage] || stage || "…";
  }
  return STAGE_LABELS[stage] || stage || "…";
}

let _transcriptMedia = { el: null, onTime: null };
let _evalPlaybackSync = null;
let _lastEvalScrollKey = null;
let selectedStem = null;
let _allLibraryItems = [];
/** Запись, для которой открыт диалог метаданных (менеджер / локация / теги). */
let metaEditStem = null;
/** Последний stem, для которого загружали workspace (для ?criteria= при смене чеклиста). */
let lastWorkspaceStem = null;
/** Имя чеклиста из последнего успешного ответа workspace — если <select> кратковременно пуст (гонка с updateEvalToolbar). */
let lastWorkspaceCriteriaRequested = null;
let libraryPollTimer = null;
let workspaceJobPollTimer = null;
let criteriaPopulate = false;
let _currentEvalMode = "ai"; // "ai" | "human" | "compare"
let _lastWorkspaceData = null;

/** Какой чеклист запросить у API: явный override, иначе последний успешный ответ для этого stem (не DOM — избегаем гонки с <select>). */
function resolveWorkspaceCriteriaQuery(stem, criteriaOverride) {
  if (criteriaOverride != null && String(criteriaOverride).trim() !== "") {
    return String(criteriaOverride).trim();
  }
  if (lastWorkspaceStem === stem && lastWorkspaceCriteriaRequested) {
    return lastWorkspaceCriteriaRequested;
  }
  return "";
}

function detachEvalPlaybackSync() {
  if (_evalPlaybackSync && _evalPlaybackSync.el && _evalPlaybackSync.fn) {
    _evalPlaybackSync.el.removeEventListener("timeupdate", _evalPlaybackSync.fn);
  }
  _evalPlaybackSync = null;
  _lastEvalScrollKey = null;
}

function stopWorkspaceJobPoll() {
  if (workspaceJobPollTimer) {
    clearInterval(workspaceJobPollTimer);
    workspaceJobPollTimer = null;
  }
}

function clearWorkspaceView() {
  stopWorkspaceJobPoll();
  detachTranscriptMedia();
  document.getElementById("workspace-empty").style.display = "block";
  document.getElementById("workspace").style.display = "none";
  lastWorkspaceStem = null;
  lastWorkspaceCriteriaRequested = null;
}

function detachTranscriptMedia() {
  detachEvalPlaybackSync();
  if (_transcriptMedia.el && _transcriptMedia.onTime) {
    _transcriptMedia.el.removeEventListener("timeupdate", _transcriptMedia.onTime);
  }
  if (_transcriptMedia.el) {
    _transcriptMedia.el.pause();
    _transcriptMedia.el.removeAttribute("src");
    _transcriptMedia.el.load();
    _transcriptMedia.el.onerror = null;
  }
  _transcriptMedia = { el: null, onTime: null };

  const v = document.getElementById("transcript-video");
  const a = document.getElementById("transcript-audio");
  for (const el of [v, a]) {
    if (!el) continue;
    el.onerror = null;
    el.removeAttribute("src");
    el.load();
  }
}

function scoreClass(score) {
  if (score === null || score === undefined) return "score-null";
  const n = Number(score);
  if (Number.isNaN(n)) return "score-null";
  if (n < 40) return "score-low";
  if (n < 70) return "score-mid";
  return "score-high";
}

function formatScore(score) {
  if (score === null || score === undefined) return "—";
  return String(score);
}

/** Человекочитаемая дата/время оценки (ISO из API). */
function formatEvaluatedAt(iso) {
  if (!iso || typeof iso !== "string") return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString("ru-RU", {
    day: "numeric",
    month: "long",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s == null ? "" : String(s);
  return d.innerHTML;
}

/** Переход к таймкоду в плеере (та же запись, что и транскрипт). */
function getActiveMediaElement() {
  const v = document.getElementById("transcript-video");
  const a = document.getElementById("transcript-audio");
  if (v && v.style.display !== "none") return v;
  if (a && a.style.display !== "none") return a;
  return v || a;
}

function seekMediaToSeconds(t) {
  const el = getActiveMediaElement();
  if (!el || !el.getAttribute("src")) return;
  el.currentTime = Math.max(0, Number(t));
  el.play().catch(() => {});
}

/** Подсветка строк чеклиста и таймкодов по текущему времени видео. */
function syncEvalHighlight(t, evaluation) {
  const tbody = document.querySelector("#eval-criteria-table tbody");
  const segList = document.getElementById("transcript-segments");
  if (!evaluation) return;

  if (tbody) {
    const rows = tbody.querySelectorAll("tr[data-criterion-id]");
    rows.forEach((tr) => {
      tr.classList.remove("eval-row--evidence-active");
      tr.querySelectorAll(".evidence-chip").forEach((b) => b.classList.remove("evidence-chip--active"));
    });
    for (const c of evaluation.criteria || []) {
      const tr = tbody.querySelector(`tr[data-criterion-id="${CSS.escape(c.id)}"]`);
      if (!tr) continue;
      let rowHit = false;
      tr.querySelectorAll(".evidence-chip").forEach((btn) => {
        const a = Number(btn.dataset.evStart);
        const b = Number(btn.dataset.evEnd);
        if (Number.isFinite(a) && Number.isFinite(b) && t >= a && t < b) {
          btn.classList.add("evidence-chip--active");
          rowHit = true;
        }
      });
      if (rowHit) tr.classList.add("eval-row--evidence-active");
    }
  }

  if (segList) {
    segList.querySelectorAll(".seg-eval-item").forEach((item) => {
      item.classList.remove("seg-eval-item--active");
      for (const w of item.querySelectorAll(".seg-ev-window")) {
        const a = Number(w.dataset.evStart);
        const b = Number(w.dataset.evEnd);
        if (Number.isFinite(a) && Number.isFinite(b) && t >= a && t < b) {
          item.classList.add("seg-eval-item--active");
          break;
        }
      }
    });
  }

  const activeInTranscript =
    segList && segList.querySelector(".seg-eval-item.seg-eval-item--active");
  const activeChip = tbody && tbody.querySelector(".evidence-chip--active");
  const activeRow = tbody && tbody.querySelector("tr.eval-row--evidence-active");
  /* Сначала прокрутка чеклиста справа; иначе — встроенный блок в транскрипте */
  const scrollTarget = activeChip || activeRow || activeInTranscript;
  const scrollKey = scrollTarget
    ? activeChip
      ? `chip:${activeChip.dataset.evStart}:${activeChip.dataset.evEnd}`
      : activeRow
        ? `row:${activeRow.dataset.criterionId}`
        : `tr:${activeInTranscript && activeInTranscript.dataset.criterionId}`
    : null;
  if (scrollKey !== _lastEvalScrollKey) {
    _lastEvalScrollKey = scrollKey;
    if (scrollTarget) {
      scrollTarget.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }
}

function attachEvalPlaybackSync(evaluation) {
  detachEvalPlaybackSync();
  if (!evaluation || !evaluation.criteria) return;
  const media = getActiveMediaElement();
  if (!media || !media.getAttribute("src")) return;
  const fn = () => syncEvalHighlight(media.currentTime, evaluation);
  media.addEventListener("timeupdate", fn);
  _evalPlaybackSync = { el: media, fn };
  fn();
}

function jobTooltip(item) {
  const j = item.job;
  if (j && (j.status === "queued" || j.status === "running")) {
    const label = jobStageLabel(j);
    return `${label} (нажмите для просмотра)`;
  }
  if (j && j.status === "error") {
    return `Ошибка: ${j.error || "неизвестно"}`;
  }
  if (item.has_transcript && item.has_tone && item.has_evaluation) {
    return "Готово: транскрипт, SER, оценка ИИ";
  }
  if (item.has_transcript) {
    return "Есть транскрипт (ожидаются тон и/или оценка)";
  }
  return "Файл загружен, ожидает обработки";
}

function statusDotClass(item) {
  const j = item.job;
  if (j && (j.status === "queued" || j.status === "running")) return "status-dot--processing";
  if (j && j.status === "error") return "status-dot--error";
  if (item.has_transcript && item.has_tone && item.has_evaluation) return "status-dot--ready";
  if (item.has_transcript) return "status-dot--partial";
  return "status-dot--waiting";
}

async function fetchLibrary() {
  const r = await fetch(`${API}/api/library`);
  if (!r.ok) return [];
  return r.json();
}

function filterAndSortLibrary(items) {
  const searchEl = document.getElementById("library-search");
  const sortEl = document.getElementById("library-sort");
  const q = (searchEl?.value || "").trim().toLowerCase();
  const sort = sortEl?.value || "date_desc";

  let filtered = items;
  if (q) {
    filtered = items.filter((item) => {
      const haystack = [
        item.video_file,
        item.stem,
        item.manager_name,
        item.location_name,
        ...(item.tags || []),
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      return haystack.includes(q);
    });
  }

  const sorted = [...filtered];
  switch (sort) {
    case "date_asc":
      sorted.sort((a, b) => (a.mtime || 0) - (b.mtime || 0));
      break;
    case "name_asc":
      sorted.sort((a, b) => (a.video_file || "").localeCompare(b.video_file || "", "ru"));
      break;
    case "name_desc":
      sorted.sort((a, b) => (b.video_file || "").localeCompare(a.video_file || "", "ru"));
      break;
    default:
      sorted.sort((a, b) => (b.mtime || 0) - (a.mtime || 0));
  }
  return sorted;
}

const LIBRARY_GROUP_NONE = "__none__";

function libraryGroupKey(name) {
  const t = name != null ? String(name).trim() : "";
  return t || LIBRARY_GROUP_NONE;
}

function libraryGroupLabel(key) {
  return key === LIBRARY_GROUP_NONE ? "Не указано" : key;
}

function sortLibraryGroupKeys(keys) {
  return [...keys].sort((a, b) => {
    const aNone = a === LIBRARY_GROUP_NONE;
    const bNone = b === LIBRARY_GROUP_NONE;
    if (aNone !== bNone) return aNone ? 1 : -1;
    return libraryGroupLabel(a).localeCompare(libraryGroupLabel(b), "ru", { sensitivity: "base" });
  });
}

/** Локация → менеджер → список записей (порядок внутри группы = порядок в filterAndSortLibrary). */
function buildLibraryGroupMap(visible) {
  const byLoc = new Map();
  for (const item of visible) {
    const lk = libraryGroupKey(item.location_name);
    const mk = libraryGroupKey(item.manager_name);
    if (!byLoc.has(lk)) byLoc.set(lk, new Map());
    const byMan = byLoc.get(lk);
    if (!byMan.has(mk)) byMan.set(mk, []);
    byMan.get(mk).push(item);
  }
  return byLoc;
}

function createLibraryRow(item) {
  const wrap = document.createElement("div");
  wrap.className = "library-row" + (item.stem === selectedStem ? " selected" : "");
  wrap.dataset.stem = item.stem;

  const row = document.createElement("button");
  row.type = "button";
  row.className = "library-item";

  const dotWrap = document.createElement("span");
  dotWrap.className = "status-dot-wrap";
  const dot = document.createElement("span");
  dot.className = "status-dot " + statusDotClass(item);
  dotWrap.appendChild(dot);
  dotWrap.title = jobTooltip(item);

  const text = document.createElement("div");
  text.style.minWidth = "0";
  const name = document.createElement("div");
  name.className = "library-item-name";
  name.textContent = item.video_file || item.stem;
  const meta = document.createElement("div");
  meta.className = "library-item-meta";
  const parts = [];
  if (item.has_transcript) parts.push("текст");
  if (item.has_tone) parts.push("тон");
  if (item.has_evaluation) parts.push("оценка");
  if (item.has_video_file === false) parts.push("нет файла в 01.Video");
  meta.textContent = parts.length ? parts.join(" · ") : "—";

  text.appendChild(name);
  text.appendChild(meta);

  row.appendChild(dotWrap);
  row.appendChild(text);
  row.addEventListener("click", () => {
    selectedStem = item.stem;
    document.querySelectorAll(".library-row").forEach((el) => {
      el.classList.toggle("selected", el.dataset.stem === selectedStem);
    });
    loadWorkspace(item.stem);
  });

  const delBtn = document.createElement("button");
  delBtn.type = "button";
  delBtn.className = "library-item-delete";
  delBtn.setAttribute("aria-label", "Удалить");
  delBtn.title = "Удалить";
  delBtn.innerHTML =
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>';
  delBtn.addEventListener("click", async (e) => {
    e.stopPropagation();
    e.preventDefault();
    const label = item.video_file || item.stem;
    if (
      !confirm(
        `Удалить «${label}»? Будут удалены файл в 01.Video (если есть), транскрипт, тон и оценки.`
      )
    ) {
      return;
    }
    const r = await fetch(`${API}/api/library/${encodeURIComponent(item.stem)}`, {
      method: "DELETE",
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      alert(data.detail || data.message || "Не удалось удалить");
      return;
    }
    if (selectedStem === item.stem) {
      selectedStem = null;
      clearWorkspaceView();
    }
    await loadLibrary();
  });

  const settingsBtn = document.createElement("button");
  settingsBtn.type = "button";
  settingsBtn.className = "library-item-settings";
  settingsBtn.setAttribute("aria-label", "Настройки записи");
  settingsBtn.title = "Менеджер, локация, теги";
  settingsBtn.innerHTML =
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>';
  settingsBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    e.preventDefault();
    openMetaDialog(item.stem);
  });

  wrap.appendChild(row);
  wrap.appendChild(settingsBtn);
  wrap.appendChild(delBtn);
  return wrap;
}

function renderLibrary(items) {
  const list = document.getElementById("library-list");
  const empty = document.getElementById("library-empty");
  const prevScroll = list ? list.scrollTop : 0;
  list.innerHTML = "";
  const visible = filterAndSortLibrary(items);
  if (!visible.length) {
    empty.style.display = "block";
    empty.textContent = items.length
      ? "Нет записей по фильтру"
      : "В папке 01.Video пока пусто — загрузите файл выше.";
    return;
  }
  empty.style.display = "none";

  const byLoc = buildLibraryGroupMap(visible);
  for (const locKey of sortLibraryGroupKeys([...byLoc.keys()])) {
    const byMan = byLoc.get(locKey);
    const locDetails = document.createElement("details");
    locDetails.className = "library-folder";
    locDetails.open = true;
    const locSum = document.createElement("summary");
    locSum.className = "library-folder-summary";
    locSum.textContent = libraryGroupLabel(locKey);
    locDetails.appendChild(locSum);

    for (const manKey of sortLibraryGroupKeys([...byMan.keys()])) {
      const groupItems = byMan.get(manKey);
      const manDetails = document.createElement("details");
      manDetails.className = "library-subfolder";
      manDetails.open = true;
      const manSum = document.createElement("summary");
      manSum.className = "library-subfolder-summary";
      manSum.textContent = libraryGroupLabel(manKey);
      manDetails.appendChild(manSum);

      const groupRows = document.createElement("div");
      groupRows.className = "library-group-rows";
      for (const it of groupItems) {
        groupRows.appendChild(createLibraryRow(it));
      }
      manDetails.appendChild(groupRows);
      locDetails.appendChild(manDetails);
    }
    list.appendChild(locDetails);
  }
  if (list) list.scrollTop = prevScroll;
}

/** Интервал опроса библиотеки (пока идёт задача — чаще, чтобы видеть этапы). */
const LIBRARY_POLL_MS = 2000;

function startLibraryPoll() {
  if (libraryPollTimer) clearInterval(libraryPollTimer);
  let wasBusy = true;

  libraryPollTimer = setInterval(async () => {
    const items = await fetchLibrary();
    _allLibraryItems = items;
    const busy = items.some(
      (x) => x.job && (x.job.status === "queued" || x.job.status === "running")
    );

    renderLibrary(items);

    if (wasBusy && !busy && selectedStem) {
      await loadWorkspace(selectedStem, true);
    }
    wasBusy = busy;

    if (!busy) {
      clearInterval(libraryPollTimer);
      libraryPollTimer = null;
    }
  }, LIBRARY_POLL_MS);
}

async function loadLibrary() {
  const items = await fetchLibrary();
  _allLibraryItems = items;
  renderLibrary(items);
  const busy = items.some(
    (x) => x.job && (x.job.status === "queued" || x.job.status === "running")
  );
  if (busy) startLibraryPoll();
}

function setupLibraryControls() {
  const searchEl = document.getElementById("library-search");
  const sortEl = document.getElementById("library-sort");
  if (searchEl) {
    searchEl.addEventListener("input", () => renderLibrary(_allLibraryItems));
  }
  if (sortEl) {
    sortEl.addEventListener("change", () => renderLibrary(_allLibraryItems));
  }
}

async function loadWorkspace(stem, silent, criteriaOverride) {
  const crit = resolveWorkspaceCriteriaQuery(stem, criteriaOverride);
  const q = crit ? `?criteria=${encodeURIComponent(crit)}` : "";
  const r = await fetch(`${API}/api/workspace/${encodeURIComponent(stem)}${q}`);
  if (!r.ok) return;
  const ws = await r.json();
  lastWorkspaceStem = stem;
  lastWorkspaceCriteriaRequested = ws.evaluation_criteria || ws.criteria?.active || null;

  document.getElementById("workspace-empty").style.display = "none";
  document.getElementById("workspace").style.display = "block";

  const titleEl = document.getElementById("workspace-title");
  titleEl.textContent = ws.video_file || ws.stem;

  const jobBanner = document.getElementById("workspace-job-banner");
  const j = ws.job;
  stopWorkspaceJobPoll();
  if (j && (j.status === "queued" || j.status === "running")) {
    jobBanner.style.display = "block";
    const label = jobStageLabel(j);
    jobBanner.textContent = `Сейчас: ${label}`;
    workspaceJobPollTimer = setInterval(async () => {
      const cPoll = resolveWorkspaceCriteriaQuery(stem, undefined);
      const qPoll = cPoll ? `?criteria=${encodeURIComponent(cPoll)}` : "";
      const r2 = await fetch(`${API}/api/workspace/${encodeURIComponent(stem)}${qPoll}`);
      if (!r2.ok) return;
      const w2 = await r2.json();
      const j2 = w2.job;
      if (!j2 || (j2.status !== "queued" && j2.status !== "running")) {
        stopWorkspaceJobPoll();
        await loadWorkspace(stem, true, lastWorkspaceCriteriaRequested || undefined);
        return;
      }
      const b = document.getElementById("workspace-job-banner");
      if (b) {
        b.style.display = "block";
        b.textContent = `Сейчас: ${jobStageLabel(j2)}`;
      }
    }, 500);
  } else if (j && j.status === "error") {
    jobBanner.style.display = "block";
    jobBanner.textContent = `Ошибка пайплайна: ${j.error || "неизвестно"}`;
  } else {
    jobBanner.style.display = "none";
  }

  _lastWorkspaceData = ws;

  renderMediaAndTranscript(ws);
  renderToneHint(ws.tone);
  renderEvaluation(ws.evaluation, {
    hasTranscript: !!ws.transcript,
    criteriaLabel: ws.evaluation_criteria || ws.criteria?.active || "",
  });
  renderHumanEvalForm(ws);
  updateEvalToggleState(ws);
  updateEvalToolbar(ws);

  if (!silent) {
    document.getElementById("workspace").scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function renderMediaAndTranscript(ws) {
  detachTranscriptMedia();

  const segs = document.getElementById("transcript-segments");
  const videoEl = document.getElementById("transcript-video");
  const audioEl = document.getElementById("transcript-audio");
  const mediaErr = document.getElementById("transcript-media-error");
  const miss = document.getElementById("transcript-missing");

  const data = ws.transcript;
  mediaErr.style.display = "none";
  mediaErr.textContent = "";

  const vfName = ((data && data.video_file) || ws.video_file || "").toLowerCase();
  const isAudio = /\.(m4a|mp3|aac|wav|flac|ogg)$/.test(vfName);

  videoEl.removeAttribute("src");
  audioEl.removeAttribute("src");
  videoEl.load();
  audioEl.load();

  let mediaEl;
  if (isAudio) {
    videoEl.style.display = "none";
    audioEl.style.display = "block";
    mediaEl = audioEl;
  } else {
    audioEl.style.display = "none";
    videoEl.style.display = "block";
    mediaEl = videoEl;
  }

  if (ws.video_url) {
    mediaEl.src = `${API}${ws.video_url}`;
    mediaEl.onerror = () => {
      mediaErr.style.display = "block";
      mediaErr.textContent =
        "Не удалось воспроизвести файл из 01.Video (проверьте формат или путь).";
    };
  } else {
    mediaErr.style.display = "block";
    mediaErr.textContent = "Исходный файл не найден в 01.Video.";
  }

  if (!data) {
    miss.style.display = "block";
    miss.textContent = "Транскрипт появится после этапа распознавания речи.";
    segs.innerHTML = "";
    const eh = document.getElementById("transcript-eval-hint");
    const ar = document.getElementById("transcript-asr-hint");
    if (eh) eh.style.display = "none";
    if (ar) {
      ar.style.display = "none";
      ar.textContent = "";
    }
    return;
  }
  miss.style.display = "none";

  const asrHint = document.getElementById("transcript-asr-hint");
  if (asrHint) {
    const text = formatAsrHintText(data);
    if (text) {
      asrHint.textContent = text;
      asrHint.style.display = "block";
    } else {
      asrHint.textContent = "";
      asrHint.style.display = "none";
    }
  }

  const evalHint = document.getElementById("transcript-eval-hint");
  if (evalHint) {
    const ev = ws.evaluation;
    const hasEv =
      ev &&
      Array.isArray(ev.criteria) &&
      ev.criteria.some(
        (c) => Array.isArray(c.evidence_segments) && c.evidence_segments.length > 0
      );
    if (hasEv) {
      evalHint.style.display = "block";
      evalHint.textContent =
        "Справа от транскрипции — таблица оценки: критерии чеклиста, пересекающиеся с репликой по времени (балл и комментарий ИИ). Таймкоды — в этой таблице.";
    } else {
      evalHint.style.display = "none";
    }
  }

  syncTranscriptHintsPopover();

  const segments = data.segments || [];
  const toneSegs =
    ws.tone && Array.isArray(ws.tone.segments) ? ws.tone.segments : [];

  let lastActiveSegIdx = -1;
  const onTime = () => {
    const t = mediaEl.currentTime;
    const segDivs = segs.querySelectorAll(".seg");
    segDivs.forEach((div) => div.classList.remove("seg--active"));
    let activeIdx = -1;
    for (let i = 0; i < segments.length; i++) {
      const s = segments[i];
      const start = Number(s.start) || 0;
      const end = Number(s.end);
      const endOk = Number.isFinite(end) ? end : start + 1e9;
      if (t >= start && t < endOk) {
        activeIdx = i;
        if (segDivs[i]) segDivs[i].classList.add("seg--active");
        break;
      }
    }
    if (activeIdx === -1) {
      lastActiveSegIdx = -1;
    } else if (activeIdx !== lastActiveSegIdx && segDivs[activeIdx]) {
      lastActiveSegIdx = activeIdx;
      segDivs[activeIdx].scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  };
  mediaEl.addEventListener("timeupdate", onTime);
  _transcriptMedia = { el: mediaEl, onTime };

  function seekToSegment(s) {
    if (!ws.video_url) return;
    const start = Number(s.start) || 0;
    mediaEl.currentTime = Math.max(0, start);
    mediaEl.play().catch(() => {});
  }

  segs.innerHTML = "";
  segments.forEach((s, i) => {
    const div = document.createElement("div");
    div.className = "seg";
    div.tabIndex = 0;
    const t0 = s.start != null ? s.start : "";
    const t1 = s.end != null ? s.end : "";
    const ts = toneSegmentForIndex(i, s, toneSegs);
    const metaParts = [];
    if (ts) {
      const rawLab = ts.top_label || "—";
      const lab =
        SER_LABEL_DISPLAY[rawLab] != null ? SER_LABEL_DISPLAY[rawLab] : rawLab;
      const pct =
        ts.top_score != null ? (Number(ts.top_score) * 100).toFixed(1) + "%" : "—";
      const hint =
        rawLab === "uncertain" && ts.model_top_label != null
          ? ` title="модель: ${String(ts.model_top_label)} (${ts.model_top_score != null ? (Number(ts.model_top_score) * 100).toFixed(1) : "—"}%), отрыв ${ts.label_margin != null ? (Number(ts.label_margin) * 100).toFixed(1) : "—"} п.п."`
          : "";
      metaParts.push(
        `аудио: <strong${hint}>${escapeHtml(lab)}</strong> · ${escapeHtml(pct)}`
      );
    }
    const delPart = formatDeliveryPart(s.delivery);
    if (delPart) metaParts.push(delPart);
    const metaBlock =
      metaParts.length > 0
        ? `<div class="seg-meta">${metaParts.join(" · ")}</div>`
        : "";

    const roleTag = s.speaker_role
      ? ` <span class="spk-role">${escapeHtml(s.speaker_role)}</span>`
      : "";
    div.innerHTML = `
      <div class="seg-main">
        <span class="spk">${escapeHtml(s.speaker || "")}${roleTag}</span>
        <span class="time">[${t0} – ${t1}]</span>
        <div class="txt">${escapeHtml(s.text || "")}</div>
        ${metaBlock}
      </div>
    `;
    const segHits = criteriaOverlappingTranscriptSegment(
      Number(s.start),
      Number(s.end),
      ws.evaluation
    );
    appendSegEvalInline(div, segHits);
    div.addEventListener("click", () => seekToSegment(s));
    div.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        seekToSegment(s);
      }
    });
    segs.appendChild(div);
  });
}

/** Скорость речи из delivery (Whisper-эвристики). */
function formatDeliveryPart(d) {
  if (!d || typeof d !== "object") return "";
  const wps = d.words_per_sec;
  if (wps == null || !Number.isFinite(Number(wps))) return "";
  let line = `скорость: ${Number(wps).toFixed(1)} сл/с`;
  const flags = d.flags;
  if (Array.isArray(flags) && flags.includes("fast_pace")) {
    line += " · быстрый темп";
  }
  return line;
}

/** Критерии, у которых evidence пересекается с репликой [lo, hi] по времени. */
function criteriaOverlappingTranscriptSegment(segStart, segEnd, evaluation) {
  if (!evaluation || !Array.isArray(evaluation.criteria)) return [];
  const s0 = Number(segStart);
  const s1 = Number(segEnd);
  if (!Number.isFinite(s0) || !Number.isFinite(s1)) return [];
  const lo = Math.min(s0, s1);
  const hi = Math.max(s0, s1);
  const out = [];
  for (const c of evaluation.criteria) {
    const segs = c.evidence_segments;
    if (!Array.isArray(segs)) continue;
    const overlapping = [];
    const seen = new Set();
    for (const ev of segs) {
      if (!ev || ev.start == null || ev.end == null) continue;
      const a = Number(ev.start);
      const b = Number(ev.end);
      if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
      const x = Math.max(a, lo);
      const y = Math.min(b, hi);
      if (y > x) {
        const key = `${a}:${b}`;
        if (seen.has(key)) continue;
        seen.add(key);
        overlapping.push({ start: a, end: b });
      }
    }
    if (overlapping.length) {
      out.push({
        id: c.id,
        name: c.name || c.id,
        score: c.score,
        comment: (c.comment && String(c.comment).trim()) || "",
        overlapping,
      });
    }
  }
  return out;
}

function appendSegEvalInline(segDiv, criteriaHits) {
  if (!criteriaHits.length) return;
  const wrap = document.createElement("div");
  wrap.className = "seg-eval-inline";

  for (const c of criteriaHits) {
    const item = document.createElement("div");
    item.className = "seg-eval-item " + scoreClass(c.score);
    item.dataset.criterionId = c.id;

    const head = document.createElement("div");
    head.className = "seg-eval-head";
    const nameEl = document.createElement("span");
    nameEl.className = "seg-eval-name";
    nameEl.textContent = c.name;
    const scoreEl = document.createElement("span");
    scoreEl.className = "seg-eval-score";
    scoreEl.textContent = formatScore(c.score);
    head.appendChild(nameEl);
    head.appendChild(scoreEl);
    item.appendChild(head);

    if (c.comment) {
      const com = document.createElement("div");
      com.className = "seg-eval-comment";
      com.textContent = c.comment;
      item.appendChild(com);
    }

    for (const iv of c.overlapping) {
      const w = document.createElement("span");
      w.className = "seg-ev-window";
      w.setAttribute("aria-hidden", "true");
      w.dataset.evStart = String(iv.start);
      w.dataset.evEnd = String(iv.end);
      item.appendChild(w);
    }

    wrap.appendChild(item);
  }
  segDiv.appendChild(wrap);
}

/** SER-сегмент по индексу или по близкому start, если длины разошлись. */
function toneSegmentForIndex(i, trSeg, toneSegments) {
  if (!toneSegments.length) return null;
  const cand = toneSegments[i];
  const t0 = Number(trSeg.start);
  if (cand != null && Number.isFinite(t0) && Number.isFinite(Number(cand.start))) {
    if (Math.abs(Number(cand.start) - t0) < 0.6) return cand;
  }
  for (const ts of toneSegments) {
    if (Number.isFinite(t0) && Math.abs(Number(ts.start) - t0) < 0.12) return ts;
  }
  return cand || null;
}

/** Текст подсказки про Whisper и разделение ролей (диаризация) — из полей JSON транскрипта. */
function formatAsrHintText(data) {
  if (!data) return "";
  const wm = data.whisper_model || data.asr_model;
  if (!wm) return "";
  const lang = data.language ? `язык: ${data.language}` : "язык не указан";
  const lines = [];
  lines.push(
    `Распознавание речи (OpenAI Whisper): модель ${wm}, ${lang}. Текст реплик — результат ASR по аудио.`
  );
  const dm = data.diarization_method;
  const hasDiar = data.diarization === true;
  if (hasDiar && dm === "pyannote") {
    lines.push("");
    lines.push(
      "Кто как говорит: на этапе транскрипции вызывается pyannote (speaker diarization) по аудио — по времени решается, какой фрагмент кому принадлежит. В интерфейсе и JSON — метки SPEAKER_01, SPEAKER_02… (не имена людей, а слоты говорящих)."
    );
  } else if (hasDiar && (dm === "mfcc_kmeans" || dm === "mfcc_word_kmeans")) {
    lines.push("");
    lines.push(
      "Кто как говорит: при отсутствии HF_TOKEN или сбое pyannote включается локальный режим — MFCC-признаки голоса и KMeans по сегментам/словам Whisper; метки SPEAKER_01, SPEAKER_02… грубее, чем у pyannote."
    );
  } else if (dm === "mfcc_kmeans_failed") {
    lines.push("");
    lines.push(
      "Кто как говорит: разделить говорящих не удалось; реплики с одной меткой (обычно SPEAKER_01)."
    );
  } else if (hasDiar && dm) {
    lines.push("");
    lines.push(
      `Кто как говорит: использован метод ${dm}; в списке — метки SPEAKER_01, SPEAKER_02…`
    );
  } else if (hasDiar) {
    lines.push("");
    lines.push(
      "Кто как говорит: реплики разнесены по спикерам; метки — SPEAKER_01, SPEAKER_02…"
    );
  } else {
    lines.push("");
    lines.push(
      "Разделение по говорящим не выполнялось или отключено (один поток речи или один спикер)."
    );
  }
  const spk = Array.isArray(data.speakers) ? data.speakers.filter(Boolean) : [];
  if (spk.length) {
    lines.push(`В этом файле: ${spk.join(", ")}.`);
  }
  if (data.diarization_note && String(data.diarization_note).trim()) {
    lines.push("");
    lines.push(String(data.diarization_note).trim());
  }
  return lines.join("\n");
}

/** Показать «?» только если есть хотя бы одна подсказка (ASR / оценка / SER). */
function syncTranscriptHintsPopover() {
  const det = document.getElementById("transcript-hints-details");
  const ar = document.getElementById("transcript-asr-hint");
  const eh = document.getElementById("transcript-eval-hint");
  const sh = document.getElementById("transcript-ser-hint");
  if (!det || !ar || !eh || !sh) return;
  const show =
    ar.style.display !== "none" || eh.style.display !== "none" || sh.style.display !== "none";
  det.hidden = !show;
  if (!show) det.removeAttribute("open");
}

/** Текст подсказки про эмоции по аудио (SER) в транскрипте — сначала «что это», потом детали из JSON. */
function formatSerHintText(tone) {
  if (!tone) return "";
  const intro =
    "Эмоции по аудио (SER): для каждой реплики в списке показан тон по записи голоса — нейтральный, раздражение, позитив, грусть и т.п. Это не анализ смысла текста, а отдельная модель по звуку; подпись «аудио: …» у строки относится именно к этому.";
  const parts = [intro];
  if (tone.note && String(tone.note).trim()) {
    parts.push("");
    parts.push(String(tone.note).trim());
  }
  if (tone.model && String(tone.model).trim()) {
    parts.push("");
    parts.push(`Модель (Hugging Face): ${tone.model.trim()}`);
  }
  return parts.join("\n");
}

function renderToneHint(tone) {
  const hint = document.getElementById("transcript-ser-hint");
  if (!hint) return;
  if (!tone) {
    hint.style.display = "none";
    hint.textContent = "";
    syncTranscriptHintsPopover();
    return;
  }
  const text = formatSerHintText(tone);
  const hasContent = Boolean(text);
  hint.textContent = text;
  hint.style.display = hasContent ? "block" : "none";
  syncTranscriptHintsPopover();
}

/** id для API: латиница/кириллица/цифры; если после очистки пусто — запасной ключ */
function slugifyMetaId(rawName) {
  const name = (rawName || "").trim();
  if (!name) return "";
  let id = name
    .toLowerCase()
    .replace(/\s+/g, "_")
    .replace(/[^\p{L}\p{N}_-]+/gu, "");
  if (!id) id = "id_" + Date.now().toString(36);
  return id;
}

/**
 * Модальное поле вместо window.prompt (во встроенных браузерах prompt часто отключён).
 * @returns {Promise<string|null>}
 */
function openMetaNameDialog({ title, label, placeholder }) {
  const dlg = document.getElementById("meta-name-dialog");
  const titleEl = document.getElementById("meta-name-title");
  const labelEl = document.getElementById("meta-name-label");
  const inp = document.getElementById("meta-name-input");
  const cancelBtn = document.getElementById("meta-name-cancel");
  const okBtn = document.getElementById("meta-name-ok");
  if (!dlg || !inp || !cancelBtn || !okBtn) return Promise.resolve(null);
  if (titleEl) titleEl.textContent = title;
  if (labelEl) labelEl.textContent = label;
  inp.placeholder = placeholder || "";
  inp.value = "";

  return new Promise((resolve) => {
    let done = false;
    function finish(val) {
      if (done) return;
      done = true;
      cleanup();
      try {
        dlg.close();
      } catch (_) {
        /* ignore */
      }
      resolve(val);
    }
    function cleanup() {
      inp.removeEventListener("keydown", onKey);
      cancelBtn.removeEventListener("click", onCancel);
      okBtn.removeEventListener("click", onOk);
      dlg.removeEventListener("cancel", onEsc);
    }
    function onCancel() {
      finish(null);
    }
    function onEsc() {
      finish(null);
    }
    function onOk(e) {
      e.preventDefault();
      const v = inp.value.trim();
      if (!v) {
        inp.focus();
        return;
      }
      finish(v);
    }
    function onKey(ev) {
      if (ev.key === "Enter") {
        ev.preventDefault();
        onOk(ev);
      }
    }
    cancelBtn.addEventListener("click", onCancel);
    okBtn.addEventListener("click", onOk);
    inp.addEventListener("keydown", onKey);
    dlg.addEventListener("cancel", onEsc);
    if (typeof dlg.showModal === "function") dlg.showModal();
    else {
      finish(null);
      return;
    }
    requestAnimationFrame(() => {
      inp.focus();
      inp.select();
    });
  });
}

function fillMetaForm({ managers = [], locations = [], meta = {} }) {
  const managerSel = document.getElementById("meta-manager");
  const locationSel = document.getElementById("meta-location");
  const dateIn = document.getElementById("meta-date");
  const tagsIn = document.getElementById("meta-tags");
  if (!managerSel || !locationSel) return;

  managerSel.innerHTML = '<option value="">—</option>';
  for (const m of managers) {
    const o = document.createElement("option");
    o.value = m.id;
    o.textContent = m.name || m.id;
    if (m.id === meta.manager_id) o.selected = true;
    managerSel.appendChild(o);
  }

  locationSel.innerHTML = '<option value="">—</option>';
  for (const l of locations) {
    const o = document.createElement("option");
    o.value = l.id;
    o.textContent = l.name || l.id;
    if (l.id === meta.location_id) o.selected = true;
    locationSel.appendChild(o);
  }

  if (dateIn) dateIn.value = meta.interaction_date || "";
  if (tagsIn) tagsIn.value = (meta.tags || []).join(", ");
}

async function openMetaDialog(stem) {
  const dlg = document.getElementById("meta-dialog");
  const stemLabel = document.getElementById("meta-dialog-stem-label");
  if (!dlg || !stem) return;
  metaEditStem = stem;
  if (stemLabel) stemLabel.textContent = stem;
  try {
    const [metaR, mR, lR] = await Promise.all([
      fetch(`${API}/api/workspace/${encodeURIComponent(stem)}/meta`),
      fetch(`${API}/api/managers`),
      fetch(`${API}/api/locations`),
    ]);
    if (!metaR.ok) {
      const d = await metaR.json().catch(() => ({}));
      throw new Error(d.detail || "Не удалось загрузить метаданные");
    }
    const meta = await metaR.json();
    const managers = await mR.json();
    const locations = await lR.json();
    fillMetaForm({ managers, locations, meta });
    if (typeof dlg.showModal === "function") dlg.showModal();
  } catch (e) {
    metaEditStem = null;
    alert(String(e));
  }
}

function setupMetaPanel() {
  const saveBtn = document.getElementById("meta-save");
  const dlg = document.getElementById("meta-dialog");
  const closeBtn = document.getElementById("meta-dialog-close");

  if (dlg) {
    dlg.addEventListener("close", () => {
      metaEditStem = null;
    });
    dlg.addEventListener("click", async (e) => {
      const t = e.target;
      const mgrBtn = t && t.closest && t.closest("#meta-manager-add");
      const locBtn = t && t.closest && t.closest("#meta-location-add");
      if (!mgrBtn && !locBtn) return;
      e.preventDefault();
      e.stopPropagation();

      const isManager = Boolean(mgrBtn);
      const name = await openMetaNameDialog({
        title: isManager ? "Новый менеджер" : "Новая локация",
        label: isManager ? "Имя" : "Название",
        placeholder: isManager ? "Например: Иван Петров" : "Например: Офис Центр",
      });
      if (!name) return;

      const id = slugifyMetaId(name);
      const url = isManager ? `${API}/api/managers` : `${API}/api/locations`;
      try {
        const r = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id, name }),
        });
        if (!r.ok) {
          const d = await r.json().catch(() => ({}));
          alert(d.detail || "Не удалось добавить");
          return;
        }
        if (metaEditStem) await openMetaDialog(metaEditStem);
        else if (selectedStem) await loadWorkspace(selectedStem, true);
      } catch (err) {
        alert(String(err));
      }
    });
  }
  if (closeBtn && dlg) {
    closeBtn.addEventListener("click", () => dlg.close());
  }

  if (saveBtn) {
    saveBtn.addEventListener("click", async () => {
      const stem = metaEditStem;
      if (!stem) return;
      const managerSel = document.getElementById("meta-manager");
      const locationSel = document.getElementById("meta-location");
      const dateIn = document.getElementById("meta-date");
      const tagsIn = document.getElementById("meta-tags");
      const managerId = managerSel?.value || null;
      const managerName = managerId ? managerSel.selectedOptions[0]?.textContent || null : null;
      const locationId = locationSel?.value || null;
      const locationName = locationId ? locationSel.selectedOptions[0]?.textContent || null : null;
      const tagsRaw = (tagsIn?.value || "").split(",").map((s) => s.trim()).filter(Boolean);
      try {
        const r = await fetch(`${API}/api/workspace/${encodeURIComponent(stem)}/meta`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            manager_id: managerId,
            manager_name: managerName,
            location_id: locationId,
            location_name: locationName,
            interaction_date: dateIn?.value || null,
            tags: tagsRaw,
          }),
        });
        if (!r.ok) {
          const d = await r.json().catch(() => ({}));
          alert(d.detail || "Не удалось сохранить");
          return;
        }
        if (dlg) dlg.close();
        await loadLibrary();
        if (selectedStem === stem) await loadWorkspace(stem, true);
      } catch (e) {
        alert(String(e));
      }
    });
  }
}

function updateEvalToolbar(ws) {
  const sel = document.getElementById("criteria-select");
  const btn = document.getElementById("eval-refresh");
  if (!sel || !btn) return;

  const crit = ws.criteria || { active: "criteria.yaml", files: [] };
  const files = crit.files || [];
  const active = crit.active || "criteria.yaml";

  criteriaPopulate = true;
  sel.innerHTML = "";
  const names = files.length ? files.map((f) => (f && f.name) || f) : [active];
  for (const name of names) {
    if (!name) continue;
    const o = document.createElement("option");
    o.value = name;
    o.textContent = name;
    if (name === active) o.selected = true;
    sel.appendChild(o);
  }
  criteriaPopulate = false;

  const j = ws.job;
  const busy = j && (j.status === "queued" || j.status === "running");
  const hasTr = !!ws.transcript;
  btn.disabled = !hasTr || busy;
  sel.disabled = busy;

  btn.title = busy
    ? "Дождитесь окончания задачи"
    : !hasTr
      ? "Сначала нужен транскрипт"
      : "Сгенерировать или пересчитать оценку по выбранному в списке чеклисту";

  const editBtn = document.getElementById("criteria-edit-btn");
  const newBtn = document.getElementById("criteria-new-btn");
  if (editBtn) editBtn.disabled = busy;
  if (newBtn) newBtn.disabled = busy;
}

function setupEvalToolbar() {
  const sel = document.getElementById("criteria-select");
  const btn = document.getElementById("eval-refresh");
  if (!sel || !btn) return;

  sel.addEventListener("change", async () => {
    if (criteriaPopulate) return;
    const file = sel.value;
    try {
      const r = await fetch(`${API}/api/criteria/active`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file }),
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        const d = data.detail;
        const msg =
          typeof d === "string"
            ? d
            : Array.isArray(d)
              ? d.map((x) => x.msg || x).join("; ")
              : "Не удалось сохранить чеклист";
        alert(msg);
        if (selectedStem) await loadWorkspace(selectedStem, true);
        return;
      }
      if (selectedStem) await loadWorkspace(selectedStem, true, file);
    } catch (e) {
      alert(String(e));
    }
  });

  btn.addEventListener("click", async () => {
    if (!selectedStem || btn.disabled) return;
    const crit = sel.value || "";
    try {
      const r = await fetch(
        `${API}/api/workspace/${encodeURIComponent(selectedStem)}/re-evaluate`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(crit ? { criteria: crit } : {}),
        }
      );
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        const d = data.detail;
        const msg =
          typeof d === "string"
            ? d
            : Array.isArray(d)
              ? d.map((x) => x.msg || x).join("; ")
              : data.message || "Ошибка запуска";
        alert(msg);
        return;
      }
      loadLibrary();
      await loadWorkspace(selectedStem, true, crit || undefined);
    } catch (e) {
      alert(String(e));
    }
  });
}

let criteriaEditingFilename = null;

function flattenApiDetail(d) {
  if (typeof d === "string") return d;
  if (Array.isArray(d)) return d.map((x) => (x.msg != null ? x.msg : String(x))).join("; ");
  return "";
}

function appendCriteriaEditorRow(container, data = {}) {
  const row = document.createElement("div");
  row.className = "criteria-editor-row";
  const head = document.createElement("div");
  head.className = "criteria-editor-row-head";
  const idIn = document.createElement("input");
  idIn.type = "text";
  idIn.className = "criteria-row-id";
  idIn.placeholder = "id_snake_case";
  idIn.value = data.id || "";
  const nameIn = document.createElement("input");
  nameIn.type = "text";
  nameIn.className = "criteria-row-name";
  nameIn.placeholder = "Название";
  nameIn.value = data.name || "";
  const rm = document.createElement("button");
  rm.type = "button";
  rm.className = "criteria-row-remove";
  rm.setAttribute("aria-label", "Удалить критерий");
  rm.textContent = "×";
  rm.addEventListener("click", () => row.remove());
  head.appendChild(idIn);
  head.appendChild(nameIn);
  head.appendChild(rm);
  const ta = document.createElement("textarea");
  ta.className = "criteria-row-desc";
  ta.placeholder = "Описание для ИИ (что оценивать по этому пункту)";
  ta.value = data.description || "";
  row.appendChild(head);
  row.appendChild(ta);
  container.appendChild(row);
}

function setupCriteriaDialogs() {
  const newDlg = document.getElementById("criteria-new-dialog");
  const editDlg = document.getElementById("criteria-edit-dialog");
  const newForm = document.getElementById("criteria-new-form");
  const newName = document.getElementById("criteria-new-name");
  const newCopy = document.getElementById("criteria-new-copy-from");
  const newCancel = document.getElementById("criteria-new-cancel");
  const editClose = document.getElementById("criteria-edit-close");
  const editCancel = document.getElementById("criteria-edit-cancel");
  const editSave = document.getElementById("criteria-edit-save");
  const editDelete = document.getElementById("criteria-edit-delete");
  const editRows = document.getElementById("criteria-editor-rows");
  const editVer = document.getElementById("criteria-editor-version");
  const editTitle = document.getElementById("criteria-edit-title");
  const addRowBtn = document.getElementById("criteria-edit-add-row");
  const editBtn = document.getElementById("criteria-edit-btn");
  const newBtn = document.getElementById("criteria-new-btn");

  if (!newDlg || !editDlg || !newForm || !editRows) return;

  function fillNewCopySelect() {
    const sel = document.getElementById("criteria-select");
    newCopy.innerHTML = "";
    const empty = document.createElement("option");
    empty.value = "";
    empty.textContent = "Пустой шаблон (один критерий)";
    newCopy.appendChild(empty);
    if (sel) {
      for (const opt of sel.querySelectorAll("option")) {
        const o = document.createElement("option");
        o.value = opt.value;
        o.textContent = opt.textContent;
        newCopy.appendChild(o);
      }
      newCopy.value = sel.value || "";
    }
  }

  newCancel.addEventListener("click", () => newDlg.close());
  newDlg.addEventListener("click", (e) => {
    if (e.target === newDlg) newDlg.close();
  });

  newForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const fn = (newName.value || "").trim();
    const copyRaw = (newCopy.value || "").trim();
    const copyFrom = copyRaw ? copyRaw : null;
    try {
      const r = await fetch(`${API}/api/criteria`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: fn, copy_from: copyFrom }),
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        alert(flattenApiDetail(data.detail) || "Не удалось создать чеклист");
        return;
      }
      const created = data.filename || fn;
      await fetch(`${API}/api/criteria/active`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file: created }),
      });
      newDlg.close();
      newName.value = "";
      if (selectedStem) await loadWorkspace(selectedStem, true);
    } catch (err) {
      alert(String(err));
    }
  });

  if (editBtn) {
    editBtn.addEventListener("click", async () => {
      const sel = document.getElementById("criteria-select");
      if (!sel || !sel.value || editBtn.disabled) return;
      const filename = sel.value;
      try {
        const r = await fetch(`${API}/api/criteria/content/${encodeURIComponent(filename)}`);
        const data = await r.json().catch(() => ({}));
        if (!r.ok) {
          alert(flattenApiDetail(data.detail) || "Не удалось загрузить чеклист");
          return;
        }
        criteriaEditingFilename = data.filename;
        editTitle.textContent = `Редактирование: ${data.filename}`;
        editVer.value = data.version || "1";
        editRows.innerHTML = "";
        (data.criteria || []).forEach((c) => appendCriteriaEditorRow(editRows, c));
        if (!editRows.children.length) appendCriteriaEditorRow(editRows, {});
        editDelete.hidden = !data.can_delete;
        editDlg.showModal();
      } catch (err) {
        alert(String(err));
      }
    });
  }

  if (newBtn) {
    newBtn.addEventListener("click", () => {
      if (newBtn.disabled) return;
      newName.value = "";
      fillNewCopySelect();
      newDlg.showModal();
    });
  }

  editDlg.addEventListener("click", (e) => {
    if (e.target === editDlg) editDlg.close();
  });

  if (editClose) editClose.addEventListener("click", () => editDlg.close());
  if (editCancel) editCancel.addEventListener("click", () => editDlg.close());

  if (addRowBtn) {
    addRowBtn.addEventListener("click", () => appendCriteriaEditorRow(editRows, {}));
  }

  if (editSave) {
    editSave.addEventListener("click", async () => {
      if (!criteriaEditingFilename) return;
      const rows = editRows.querySelectorAll(".criteria-editor-row");
      const criteria = [];
      let i = 0;
      rows.forEach((row) => {
        const id = row.querySelector(".criteria-row-id")?.value?.trim() || "";
        const name = row.querySelector(".criteria-row-name")?.value?.trim() || "";
        const description = row.querySelector(".criteria-row-desc")?.value?.trim() || "";
        if (!id && !name && !description) return;
        i += 1;
        const idFinal = id || `criterion_${i}`;
        criteria.push({
          id: idFinal,
          name: name || idFinal,
          description,
        });
      });
      if (!criteria.length) {
        alert("Добавьте хотя бы один критерий.");
        return;
      }
      try {
        const r = await fetch(
          `${API}/api/criteria/content/${encodeURIComponent(criteriaEditingFilename)}`,
          {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              version: (editVer.value || "").trim() || "1",
              criteria,
            }),
          }
        );
        const data = await r.json().catch(() => ({}));
        if (!r.ok) {
          alert(flattenApiDetail(data.detail) || "Не удалось сохранить");
          return;
        }
        editDlg.close();
        if (selectedStem) await loadWorkspace(selectedStem, true);
      } catch (err) {
        alert(String(err));
      }
    });
  }

  if (editDelete) {
    editDelete.addEventListener("click", async () => {
      if (!criteriaEditingFilename || editDelete.hidden) return;
      if (!confirm(`Удалить файл «${criteriaEditingFilename}» с диска?`)) return;
      try {
        const r = await fetch(
          `${API}/api/criteria/content/${encodeURIComponent(criteriaEditingFilename)}`,
          { method: "DELETE" }
        );
        const data = await r.json().catch(() => ({}));
        if (!r.ok) {
          alert(flattenApiDetail(data.detail) || "Не удалось удалить");
          return;
        }
        editDlg.close();
        if (selectedStem) await loadWorkspace(selectedStem, true);
      } catch (err) {
        alert(String(err));
      }
    });
  }
}

function renderEvaluation(evaluation, options = {}) {
  const { hasTranscript = false, criteriaLabel = "" } = options;
  const summary = document.getElementById("eval-summary");
  const miss = document.getElementById("eval-missing");
  const table = document.getElementById("eval-criteria-table");
  const tbody = table.querySelector("tbody");

  if (!evaluation) {
    detachEvalPlaybackSync();
    summary.innerHTML = "";
    miss.style.display = "block";
    table.style.display = "none";
    if (hasTranscript) {
      const label = criteriaLabel || "чеклисту";
      miss.textContent = `Для «${label}» оценка ещё не готова. Нажмите кнопку пересчёта (↻), чтобы сгенерировать оценку по этому чеклисту.`;
    } else {
      miss.textContent = "Оценка ИИ появится после завершения пайплайна.";
    }
    return;
  }
  miss.style.display = "none";
  table.style.display = "table";

  const avg = evaluation.overall_average;
  const avgStr = avg != null ? Number(avg).toFixed(1) : "—";
  const avgCls = scoreClass(avg);
  const iso = evaluation.evaluated_at || "";
  const when = formatEvaluatedAt(iso);
  const dateDd =
    iso
      ? `<dd class="eval-summary__value"><time datetime="${escapeHtml(iso)}">${escapeHtml(when)}</time></dd>`
      : `<dd class="eval-summary__value">—</dd>`;
  summary.innerHTML = `
    <div class="eval-summary-card">
      <div class="eval-summary__file">${escapeHtml(evaluation.video_file || "")}</div>
      <dl class="eval-summary__meta">
        <div class="eval-summary__field">
          <dt class="eval-summary__label">Модель</dt>
          <dd class="eval-summary__value eval-summary__mono">${escapeHtml(evaluation.model || "—")}</dd>
        </div>
        <div class="eval-summary__field">
          <dt class="eval-summary__label">Средний балл</dt>
          <dd class="eval-summary__value eval-summary__avg eval-summary__avg--${avgCls}">${escapeHtml(avgStr)}</dd>
        </div>
        <div class="eval-summary__field eval-summary__field--wide">
          <dt class="eval-summary__label">Дата оценки</dt>
          ${dateDd}
        </div>
      </dl>
    </div>
  `;

  tbody.innerHTML = "";
  for (const c of evaluation.criteria || []) {
    const tr = document.createElement("tr");
    tr.className = scoreClass(c.score);
    tr.dataset.criterionId = c.id;

    const td = document.createElement("td");
    td.colSpan = 3;
    td.className = "eval-criteria-cell";

    const head = document.createElement("div");
    head.className = "eval-criteria-head";

    const nameEl = document.createElement("div");
    nameEl.className = "crit-name";
    nameEl.textContent = c.name || c.id;

    const scoreEl = document.createElement("div");
    scoreEl.className = "score-cell";
    scoreEl.textContent = formatScore(c.score);

    const commentEl = document.createElement("div");
    commentEl.className = "crit-comment";

    const commentText = document.createElement("div");
    commentText.className = "crit-comment-text";
    commentText.textContent = c.comment || "";
    commentEl.appendChild(commentText);

    const segs = c.evidence_segments;
    if (Array.isArray(segs) && segs.length) {
      const chips = document.createElement("div");
      chips.className = "evidence-chips";
      const lab = document.createElement("span");
      lab.className = "evidence-chips-label";
      lab.textContent = "В записи: ";
      chips.appendChild(lab);
      let n = 0;
      for (const seg of segs) {
        if (!seg || seg.start == null || seg.end == null) continue;
        const t0 = Number(seg.start);
        const t1 = Number(seg.end);
        if (!Number.isFinite(t0) || !Number.isFinite(t1)) continue;
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "evidence-chip";
        btn.dataset.evStart = String(t0);
        btn.dataset.evEnd = String(t1);
        btn.textContent = `[${t0.toFixed(1)}–${t1.toFixed(1)}]`;
        btn.title = "Перейти к началу этого отрезка";
        btn.addEventListener("click", (e) => {
          e.preventDefault();
          e.stopPropagation();
          seekMediaToSeconds(t0);
        });
        chips.appendChild(btn);
        n += 1;
      }
      if (n > 0) commentEl.appendChild(chips);
    }

    head.appendChild(nameEl);
    head.appendChild(scoreEl);
    td.appendChild(head);
    td.appendChild(commentEl);
    tr.appendChild(td);
    tbody.appendChild(tr);
  }

  attachEvalPlaybackSync(evaluation);
}

/* --- Eval mode toggle (ИИ / Человек / Сравнить) --- */

function switchEvalMode(mode) {
  _currentEvalMode = mode;
  const aiView = document.getElementById("eval-ai-view");
  const humanView = document.getElementById("eval-human-view");
  const compareView = document.getElementById("eval-compare-view");
  if (aiView) aiView.style.display = mode === "ai" ? "" : "none";
  if (humanView) humanView.style.display = mode === "human" ? "" : "none";
  if (compareView) compareView.style.display = mode === "compare" ? "" : "none";

  const saveBtn = document.getElementById("human-eval-save");
  if (saveBtn) saveBtn.style.display = mode === "human" ? "" : "none";

  document.querySelectorAll("#eval-mode-toggle .eval-mode-btn").forEach((b) => {
    b.classList.toggle("eval-mode-btn--active", b.dataset.mode === mode);
  });
}

function updateEvalToggleState(ws) {
  const compareBtn = document.getElementById("eval-compare");
  if (compareBtn) {
    const hasAi = Boolean(ws && ws.evaluation);
    const hasHuman = Boolean(ws && ws.human_evaluation);
    compareBtn.disabled = !(hasAi && hasHuman);
  }
  const saveBtn = document.getElementById("human-eval-save");
  if (saveBtn) {
    saveBtn.disabled = !ws || !ws.evaluation_criteria;
  }
}

function setupEvalModeToggle() {
  const toggle = document.getElementById("eval-mode-toggle");
  if (toggle) {
    toggle.addEventListener("click", (e) => {
      const btn = e.target.closest(".eval-mode-btn");
      if (!btn) return;
      switchEvalMode(btn.dataset.mode);
    });
  }

  const compareBtn = document.getElementById("eval-compare");
  if (compareBtn) {
    compareBtn.addEventListener("click", async () => {
      if (compareBtn.disabled || !selectedStem) return;
      switchEvalMode("compare");
      await runComparison(selectedStem);
    });
  }

  const saveBtn = document.getElementById("human-eval-save");
  if (saveBtn) {
    saveBtn.addEventListener("click", async () => {
      if (!selectedStem || !_lastWorkspaceData) return;
      await saveHumanEval(selectedStem);
    });
  }
}

/* --- Human eval form --- */

function renderHumanEvalForm(ws) {
  const form = document.getElementById("human-eval-form");
  const summaryEl = document.getElementById("human-eval-summary");
  if (!form) return;
  form.innerHTML = "";
  if (summaryEl) summaryEl.innerHTML = "";

  const humanEval = ws && ws.human_evaluation;
  const aiEval = ws && ws.evaluation;
  const criteria = aiEval ? (aiEval.criteria || []) : [];

  if (!criteria.length) {
    form.innerHTML = '<p class="block-missing">Сначала запустите ИИ-оценку, чтобы получить список критериев.</p>';
    return;
  }

  const humanMap = {};
  if (humanEval && humanEval.criteria) {
    for (const c of humanEval.criteria) {
      humanMap[c.id] = c;
    }
  }

  if (humanEval && humanEval.overall_average != null) {
    const avg = Number(humanEval.overall_average).toFixed(1);
    const cls = scoreClass(humanEval.overall_average);
    const when = formatEvaluatedAt(humanEval.evaluated_at);
    summaryEl.innerHTML = `
      <div class="eval-summary-card">
        <dl class="eval-summary__meta">
          <div class="eval-summary__field">
            <dt class="eval-summary__label">Средний балл (ручной)</dt>
            <dd class="eval-summary__value eval-summary__avg eval-summary__avg--${cls}">${escapeHtml(avg)}</dd>
          </div>
          <div class="eval-summary__field">
            <dt class="eval-summary__label">Сохранено</dt>
            <dd class="eval-summary__value"><time>${escapeHtml(when)}</time></dd>
          </div>
        </dl>
      </div>
    `;
  }

  for (const c of criteria) {
    const existing = humanMap[c.id];
    const row = document.createElement("div");
    row.className = "human-eval-row";
    row.dataset.critId = c.id;
    row.dataset.critName = c.name || c.id;

    const head = document.createElement("div");
    head.className = "human-eval-row-head";

    const nameEl = document.createElement("div");
    nameEl.className = "human-eval-crit-name";
    nameEl.textContent = c.name || c.id;

    const scoreInput = document.createElement("input");
    scoreInput.type = "number";
    scoreInput.min = "0";
    scoreInput.max = "100";
    scoreInput.className = "human-eval-score-input";
    scoreInput.placeholder = "0-100";
    scoreInput.value = existing && existing.score != null ? existing.score : "";

    head.appendChild(nameEl);
    head.appendChild(scoreInput);

    const commentInput = document.createElement("textarea");
    commentInput.className = "human-eval-comment-input";
    commentInput.placeholder = "Комментарий…";
    commentInput.rows = 1;
    commentInput.value = existing ? (existing.comment || "") : "";

    row.appendChild(head);
    row.appendChild(commentInput);
    form.appendChild(row);
  }
}

async function saveHumanEval(stem) {
  const form = document.getElementById("human-eval-form");
  if (!form) return;
  const rows = form.querySelectorAll(".human-eval-row");
  const criteria = [];
  for (const row of rows) {
    const id = row.dataset.critId;
    const name = row.dataset.critName;
    const scoreInput = row.querySelector(".human-eval-score-input");
    const commentInput = row.querySelector(".human-eval-comment-input");
    const rawScore = (scoreInput?.value || "").trim();
    const score = rawScore !== "" ? Math.max(0, Math.min(100, parseInt(rawScore, 10))) : null;
    criteria.push({
      id,
      name,
      score: Number.isNaN(score) ? null : score,
      comment: (commentInput?.value || "").trim(),
    });
  }

  const ws = _lastWorkspaceData;
  const critFile = ws ? ws.evaluation_criteria : null;
  const q = critFile ? `?criteria=${encodeURIComponent(critFile)}` : "";

  try {
    const r = await fetch(`${API}/api/workspace/${encodeURIComponent(stem)}/human-eval${q}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ criteria, criteria_file: critFile }),
    });
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      alert(d.detail || "Не удалось сохранить");
      return;
    }
    await loadWorkspace(stem, true);
  } catch (e) {
    alert(String(e));
  }
}

/* --- Compare view --- */

function diffClass(pct) {
  if (pct == null) return "";
  if (pct <= 20) return "ok";
  if (pct <= 50) return "warn";
  return "danger";
}

async function runComparison(stem) {
  const summaryEl = document.getElementById("compare-summary");
  const tableWrap = document.getElementById("compare-table-wrap");
  const analysisEl = document.getElementById("compare-analysis");
  if (summaryEl) summaryEl.innerHTML = '<p style="color:var(--muted);font-size:0.82rem">Сравниваю…</p>';
  if (tableWrap) tableWrap.innerHTML = "";
  if (analysisEl) analysisEl.innerHTML = "";

  const ws = _lastWorkspaceData;
  const critFile = ws ? ws.evaluation_criteria : null;

  try {
    const r = await fetch(`${API}/api/workspace/${encodeURIComponent(stem)}/compare-eval`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ criteria: critFile }),
    });
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      if (summaryEl) summaryEl.innerHTML = `<p style="color:var(--danger)">${escapeHtml(d.detail || "Ошибка сравнения")}</p>`;
      return;
    }
    const data = await r.json();
    renderCompareResult(data);
  } catch (e) {
    if (summaryEl) summaryEl.innerHTML = `<p style="color:var(--danger)">${escapeHtml(String(e))}</p>`;
  }
}

function renderCompareResult(data) {
  const summaryEl = document.getElementById("compare-summary");
  const tableWrap = document.getElementById("compare-table-wrap");
  const analysisEl = document.getElementById("compare-analysis");

  const oc = diffClass(data.overall_diff);
  summaryEl.innerHTML = `
    <div class="compare-summary-card">
      <span class="compare-overall">ИИ: <strong>${data.ai_overall != null ? Number(data.ai_overall).toFixed(1) : "—"}</strong></span>
      <span class="compare-overall">Человек: <strong>${data.human_overall != null ? Number(data.human_overall).toFixed(1) : "—"}</strong></span>
      <span class="compare-overall compare-diff-${oc}">Δ ${data.overall_diff != null ? Number(data.overall_diff).toFixed(1) : "—"}</span>
    </div>
  `;

  const table = document.createElement("table");
  table.className = "compare-table";
  table.innerHTML = `
    <thead>
      <tr>
        <th>Критерий</th>
        <th class="score-col">ИИ</th>
        <th class="score-col">Человек</th>
        <th class="diff-col">Δ</th>
      </tr>
    </thead>
  `;
  const tbody = document.createElement("tbody");
  for (const row of (data.rows || [])) {
    const dc = diffClass(row.diff_pct);
    const tr = document.createElement("tr");
    if (dc) tr.className = `compare-row-${dc}`;
    tr.innerHTML = `
      <td>
        <div style="font-weight:600;font-size:0.8rem">${escapeHtml(row.name)}</div>
        <div style="font-size:0.72rem;color:var(--muted);margin-top:0.15rem">
          ИИ: ${escapeHtml(row.ai_comment || "—")}
        </div>
        <div style="font-size:0.72rem;color:var(--muted);margin-top:0.1rem">
          Человек: ${escapeHtml(row.human_comment || "—")}
        </div>
      </td>
      <td class="score-col">${row.ai_score != null ? row.ai_score : "—"}</td>
      <td class="score-col">${row.human_score != null ? row.human_score : "—"}</td>
      <td class="diff-col compare-diff-${dc}">${row.diff != null ? row.diff : "—"}</td>
    `;
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  tableWrap.innerHTML = "";
  tableWrap.appendChild(table);

  if (data.llm_analysis) {
    analysisEl.innerHTML = `
      <div class="compare-analysis-title">Анализ ИИ</div>
      <div>${escapeHtml(data.llm_analysis)}</div>
    `;
  } else {
    analysisEl.innerHTML = "";
  }
}

async function uploadFile(file) {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(`${API}/api/upload`, { method: "POST", body: fd });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    alert(`${file.name}: ${data.detail || data.message || "Ошибка загрузки"}`);
    return null;
  }
  return data.stem;
}

async function uploadFiles(files) {
  if (!files || !files.length) return;
  const banner = document.getElementById("workspace-job-banner");
  if (banner) {
    banner.style.display = "block";
    banner.textContent = `Загрузка ${files.length} файл(ов)…`;
  }
  let lastStem = null;
  for (let i = 0; i < files.length; i++) {
    if (banner) banner.textContent = `Загрузка ${i + 1} / ${files.length}: ${files[i].name}…`;
    const stem = await uploadFile(files[i]);
    if (stem) lastStem = stem;
  }
  if (banner) banner.style.display = "none";
  if (lastStem) {
    selectedStem = lastStem;
    await loadLibrary();
    startLibraryPoll();
    await loadWorkspace(lastStem);
    document.querySelectorAll(".library-row").forEach((el) => {
      el.classList.toggle("selected", el.dataset.stem === selectedStem);
    });
  } else {
    await loadLibrary();
  }
}

function setupDropzone() {
  const dz = document.getElementById("dropzone");
  const input = document.getElementById("file-input");

  input.addEventListener("change", () => {
    if (input.files && input.files.length) uploadFiles(Array.from(input.files));
    input.value = "";
  });

  ["dragenter", "dragover"].forEach((ev) => {
    dz.addEventListener(ev, (e) => {
      e.preventDefault();
      dz.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((ev) => {
    dz.addEventListener(ev, (e) => {
      e.preventDefault();
      dz.classList.remove("dragover");
    });
  });
  dz.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files && files.length) uploadFiles(Array.from(files));
  });
}

const SIDEBAR_COLLAPSED_KEY = "fresh-fa-sidebar-collapsed";

function setupSidebarToggle() {
  const shell = document.getElementById("app-shell");
  const collapseBtn = document.getElementById("sidebar-collapse");
  const expandBtn = document.getElementById("sidebar-expand");
  if (!shell || !collapseBtn || !expandBtn) return;

  function apply(collapsed) {
    shell.classList.toggle("sidebar-collapsed", collapsed);
    expandBtn.setAttribute("aria-hidden", collapsed ? "false" : "true");
    collapseBtn.setAttribute("aria-hidden", collapsed ? "true" : "false");
    try {
      if (collapsed) localStorage.setItem(SIDEBAR_COLLAPSED_KEY, "1");
      else localStorage.removeItem(SIDEBAR_COLLAPSED_KEY);
    } catch (_) {
      /* ignore */
    }
  }

  let initial = false;
  try {
    initial = localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === "1";
  } catch (_) {
    /* ignore */
  }
  apply(initial);

  collapseBtn.addEventListener("click", () => apply(true));
  expandBtn.addEventListener("click", () => apply(false));
}

document.addEventListener("DOMContentLoaded", () => {
  setupSidebarToggle();
  setupLibraryControls();
  setupEvalToolbar();
  setupEvalModeToggle();
  setupCriteriaDialogs();
  setupMetaPanel();
  setupDropzone();
  loadLibrary();
});
