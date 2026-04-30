const API = "";

/** Все запросы к API с cookie-сессией. */
function apiFetch(input, init) {
  const opts = init
    ? { ...init, credentials: "same-origin" }
    : { credentials: "same-origin" };
  return globalThis.fetch(input, opts);
}

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
  extract_audio: "1/7 — подготовка WAV из файла",
  diarization: "2/7 — диаризация (pyannote: кто когда говорит)",
  diarization_skip: "2/7 — без HF: диаризация будет грубее (MFCC)",
  diarization_skip_windows: "2/7 — Windows: pyannote выкл., MFCC (см. PYANNOTE_ON_WINDOWS)",
  whisper_load: "3/7 — загрузка модели Whisper",
  asr_whisper: "4/7 — распознавание речи (Whisper, обычно самый долгий шаг)",
  segments_build: "5/7 — спикеры, слова, скорость речи, тон по кускам",
  transcribing: "Транскрибация (устар.)",
  tone: "6/7 — эмоции по аудио (SER)",
  evaluating: "7/7 — оценка по чеклисту (ИИ)",
  done: "Готово",
  error: "Ошибка",
  cancelled: "Остановлено",
  resume: "Возобновление (пропуск готовых шагов)",
};

/** Подписи для задачи «только пересчёт оценки» (не шаг 7/7 полного пайплайна). */
const EVAL_ONLY_STAGE_LABELS = {
  queued: "В очереди",
  eval_prep: "Подготовка к оценке по чеклисту…",
  eval_waiting: "Ожидание: выполняется другая обработка…",
  evaluating: "Запрос к модели оценки (ИИ)…",
  done: "Готово",
  error: "Ошибка",
  cancelled: "Остановлено",
};

const AI_PIPELINE_STEPS = [
  { key: "extract_audio", label: "Подготовка WAV из файла" },
  { key: "diarization", label: "Диаризация и роли спикеров" },
  { key: "asr_whisper", label: "Распознавание речи (Whisper)" },
  { key: "segments_build", label: "Сбор сегментов и таймкодов" },
  { key: "tone", label: "Эмоции по аудио (SER)" },
  { key: "evaluating", label: "Оценка по чеклисту (ИИ)" },
];

function jobStageLabel(job) {
  if (!job) return "…";
  if (job.status === "cancelled") {
    return "Остановлено";
  }
  const stage = job.stage;
  if (job.kind === "eval_only") {
    return EVAL_ONLY_STAGE_LABELS[stage] || STAGE_LABELS[stage] || stage || "…";
  }
  return STAGE_LABELS[stage] || stage || "…";
}

const DEPARTMENT_LABELS = {
  "ОО": "Отдел оценки",
  "ОП": "Отдел продаж",
};

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
let compareStatusPollTimer = null;
let criteriaPopulate = false;
let _currentEvalMode = "ai"; // "ai" | "human" | "compare"
let _lastWorkspaceData = null;
let _showWorkspaceJobLog = false;
let _showCompletedAiPipelineCard = false;
let _humanEvalDirty = false;
let _humanEvalDraftBaseline = "[]";
/** Кеш ответа POST /compare-eval: пересчёт только если ИИ/человек на диске изменились. */
let _compareEvalCache = null;
let _authState = { auth_enabled: false, auth_type: "none", is_admin: false };
let _metaTrainingTypes = [];
let _trainingTypeEditingSlug = null;
let _showDeletedLibrary = false;
let _showForeignLibrary = false;
let _pendingUploadFiles = [];
let _activeUploadItems = new Map();
let _uploadQuota = null;
let _authLocations = [];
let _evalTransientState = {
  aiPending: false,
  comparePending: false,
  compareError: "",
  publishPending: false,
};
const ONBOARDING_TOUR_VERSION = 1;
const ONBOARDING_TOUR_STORAGE_KEY = "fresh-fa-onboarding-version";
const ONBOARDING_TOUR_PREVIEW_FILE = Object.freeze({
  name: "primer-zvonok.mp4",
  size: 24 * 1024 * 1024,
  type: "video/mp4",
});
const ONBOARDING_PREVIEW_POSTER = `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 720">
    <defs>
      <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#0f172a"/>
        <stop offset="55%" stop-color="#142a68"/>
        <stop offset="100%" stop-color="#1d4ed8"/>
      </linearGradient>
    </defs>
    <rect width="1280" height="720" fill="url(#bg)"/>
    <circle cx="1035" cy="152" r="86" fill="rgba(255,255,255,.08)"/>
    <circle cx="1115" cy="108" r="28" fill="rgba(255,255,255,.18)"/>
    <rect x="82" y="84" width="330" height="40" rx="20" fill="rgba(255,255,255,.15)"/>
    <text x="108" y="111" fill="#ffffff" font-family="Arial, sans-serif" font-size="22" font-weight="700">Демо-запись для инструкции</text>
    <rect x="82" y="146" width="510" height="92" rx="24" fill="rgba(255,255,255,.10)" stroke="rgba(255,255,255,.18)"/>
    <text x="112" y="192" fill="#ffffff" font-family="Arial, sans-serif" font-size="34" font-weight="700">Визит в салон: пример звонка</text>
    <text x="112" y="225" fill="rgba(255,255,255,.84)" font-family="Arial, sans-serif" font-size="22">Транскрипт и видео показаны в режиме справки</text>
    <rect x="94" y="562" width="1092" height="52" rx="26" fill="rgba(15,23,42,.55)" stroke="rgba(255,255,255,.18)"/>
    <rect x="122" y="584" width="332" height="10" rx="5" fill="#60a5fa"/>
    <circle cx="454" cy="589" r="12" fill="#ffffff"/>
    <circle cx="142" cy="402" r="44" fill="rgba(255,255,255,.14)"/>
    <polygon points="132,383 132,421 166,402" fill="#ffffff"/>
    <text x="1018" y="596" fill="rgba(255,255,255,.86)" font-family="Arial, sans-serif" font-size="18">01:18 / 02:44</text>
  </svg>`,
)}`;
const ONBOARDING_PREVIEW_CHECKLIST_SLUG = "oo_visit_salon";
const ONBOARDING_PREVIEW_TRANSCRIPT = Object.freeze({
  video_file: ONBOARDING_TOUR_PREVIEW_FILE.name,
  whisper_model: "large-v3-turbo",
  diarization: true,
  diarization_method: "pyannote",
  language: "ru",
  segments: [
    {
      speaker: "SPEAKER_01",
      speaker_role: "Менеджер",
      start: 0.9,
      end: 6.4,
      text: "Добрый день, меня зовут Алина, салон Fresh. Удобно сейчас пару минут поговорить?",
      delivery: { pace: "normal" },
    },
    {
      speaker: "SPEAKER_02",
      speaker_role: "Клиент",
      start: 6.8,
      end: 11.7,
      text: "Да, удобно. Я как раз хотела уточнить по записи на окрашивание.",
      delivery: { pace: "normal" },
    },
    {
      speaker: "SPEAKER_01",
      speaker_role: "Менеджер",
      start: 12.5,
      end: 22.2,
      text: "Подскажите, пожалуйста, какой результат вы хотите получить и были ли уже похожие процедуры раньше?",
      delivery: { pace: "normal" },
    },
    {
      speaker: "SPEAKER_02",
      speaker_role: "Клиент",
      start: 22.9,
      end: 34.8,
      text: "Хочу освежить цвет и скрыть седину. Последний раз делала окрашивание месяца три назад.",
      delivery: { pace: "normal" },
    },
    {
      speaker: "SPEAKER_01",
      speaker_role: "Менеджер",
      start: 35.6,
      end: 49.4,
      text: "Тогда я бы предложила консультацию с мастером и подбор оттенка на месте, чтобы сразу понять точный план работы.",
      delivery: { pace: "normal" },
    },
    {
      speaker: "SPEAKER_02",
      speaker_role: "Клиент",
      start: 50.1,
      end: 56.9,
      text: "Хорошо, а по времени это сколько обычно занимает?",
      delivery: { pace: "normal" },
    },
    {
      speaker: "SPEAKER_01",
      speaker_role: "Менеджер",
      start: 57.5,
      end: 74.2,
      text: "Обычно закладываем около двух с половиной часов. Я могу предложить вам свободное окно в четверг в 18:00 и сразу отправить подтверждение в мессенджер.",
      delivery: { pace: "normal" },
    },
    {
      speaker: "SPEAKER_02",
      speaker_role: "Клиент",
      start: 75.0,
      end: 82.9,
      text: "Четверг в 18:00 мне подходит. Скажите, пожалуйста, сколько примерно будет стоить услуга?",
      delivery: { pace: "normal" },
    },
    {
      speaker: "SPEAKER_01",
      speaker_role: "Менеджер",
      start: 83.6,
      end: 96.8,
      text: "Точную стоимость назовёт мастер после консультации, но ориентир по такой услуге обычно начинается от девяти тысяч рублей в зависимости от длины и расхода материалов.",
      delivery: { pace: "normal" },
    },
    {
      speaker: "SPEAKER_02",
      speaker_role: "Клиент",
      start: 97.3,
      end: 103.9,
      text: "Поняла, спасибо. Тогда давайте запишемся и пришлите, пожалуйста, адрес салона.",
      delivery: { pace: "normal" },
    },
    {
      speaker: "SPEAKER_01",
      speaker_role: "Менеджер",
      start: 104.5,
      end: 118.6,
      text: "Отлично, фиксирую вас на четверг в 18:00. Сейчас отправлю адрес, схему проезда и подтверждение записи в WhatsApp на номер, с которого вы звоните.",
      delivery: { pace: "normal" },
    },
    {
      speaker: "SPEAKER_02",
      speaker_role: "Клиент",
      start: 119.2,
      end: 123.8,
      text: "Да, всё верно. Благодарю за помощь.",
      delivery: { pace: "normal" },
    },
    {
      speaker: "SPEAKER_01",
      speaker_role: "Менеджер",
      start: 124.4,
      end: 131.2,
      text: "Спасибо за обращение. Будем ждать вас в салоне, хорошего дня.",
      delivery: { pace: "normal" },
    },
  ],
});
const ONBOARDING_PREVIEW_TONE = Object.freeze({
  segments: [
    { start: 0.9, top_label: "positive", top_score: 0.86 },
    { start: 6.8, top_label: "neutral", top_score: 0.77 },
    { start: 12.5, top_label: "neutral", top_score: 0.82 },
    { start: 22.9, top_label: "neutral", top_score: 0.74 },
    { start: 35.6, top_label: "positive", top_score: 0.71 },
    { start: 50.1, top_label: "neutral", top_score: 0.69 },
    { start: 57.5, top_label: "positive", top_score: 0.81 },
    { start: 75.0, top_label: "neutral", top_score: 0.76 },
    { start: 83.6, top_label: "positive", top_score: 0.72 },
    { start: 97.3, top_label: "positive", top_score: 0.79 },
    { start: 104.5, top_label: "positive", top_score: 0.88 },
    { start: 119.2, top_label: "positive", top_score: 0.84 },
    { start: 124.4, top_label: "positive", top_score: 0.91 },
  ],
});
const ONBOARDING_PREVIEW_CRITERIA = Object.freeze([
  {
    id: "greeting",
    name: "Поприветствовал клиента и представился",
    weight: 2,
    description: "Проверьте, что менеджер поздоровался, представился и обозначил контекст звонка.",
  },
  {
    id: "need_discovery",
    name: "Выявил потребность",
    weight: 3,
    description: "Проверьте, что менеджер уточнил запрос клиента и собрал ключевые вводные.",
  },
  {
    id: "offer",
    name: "Предложил решение",
    weight: 3,
    description: "Проверьте, что менеджер предложил релевантный вариант и объяснил его клиенту.",
  },
  {
    id: "closing",
    name: "Зафиксировал следующий шаг",
    weight: 2,
    description: "Проверьте, что в конце разговора зафиксирован понятный следующий шаг.",
  },
]);
const ONBOARDING_PREVIEW_AI_EVALUATION = Object.freeze({
  earned_score: 8,
  max_score: 10,
  overall_average: 80,
  evaluated_at: "2026-04-19T08:45:00Z",
  model: "gpt-4.1-mini",
  criteria: [
    {
      id: "greeting",
      name: "Поприветствовал клиента и представился",
      weight: 2,
      passed: true,
      comment: "Менеджер представился и обозначил повод звонка.",
      evidence_segments: [{ start: 4.2, end: 8.8 }],
    },
    {
      id: "need_discovery",
      name: "Выявил потребность",
      weight: 3,
      passed: true,
      comment: "Уточнил текущую ситуацию клиента и ожидаемый результат.",
      evidence_segments: [{ start: 18.1, end: 31.4 }],
    },
    {
      id: "offer",
      name: "Предложил решение",
      weight: 3,
      passed: false,
      comment: "Предложение звучит кратко, без явной привязки к озвученной потребности.",
      evidence_segments: [{ start: 44.7, end: 58.9 }],
    },
    {
      id: "closing",
      name: "Зафиксировал следующий шаг",
      weight: 2,
      passed: true,
      comment: "Следующий шаг проговорён, договорённость понятна.",
      evidence_segments: [{ start: 66.0, end: 73.5 }],
    },
  ],
});
const ONBOARDING_PREVIEW_HUMAN_DRAFT = Object.freeze({
  earned_score: 7,
  max_score: 10,
  overall_average: 70,
  evaluated_at: "2026-04-19T08:52:00Z",
  criteria: [
    {
      id: "greeting",
      name: "Поприветствовал клиента и представился",
      weight: 2,
      passed: true,
      comment: "Представился, но начал разговор немного резко.",
    },
    {
      id: "need_discovery",
      name: "Выявил потребность",
      weight: 3,
      passed: true,
      comment: "Ключевые вопросы заданы.",
    },
    {
      id: "offer",
      name: "Предложил решение",
      weight: 3,
      passed: false,
      comment: "Аргументации не хватило, нужно глубже связать предложение с запросом клиента.",
    },
    {
      id: "closing",
      name: "Зафиксировал следующий шаг",
      weight: 2,
      passed: false,
      comment: "Следующий шаг обозначен не до конца конкретно.",
    },
  ],
});
const ONBOARDING_PREVIEW_HUMAN_PUBLISHED = Object.freeze({
  earned_score: 7,
  max_score: 10,
  overall_average: 70,
  evaluated_at: "2026-04-19T09:02:00Z",
  criteria: [
    {
      id: "greeting",
      name: "Поприветствовал клиента и представился",
      weight: 2,
      passed: true,
      comment: "Представился и обозначил тему разговора.",
    },
    {
      id: "need_discovery",
      name: "Выявил потребность",
      weight: 3,
      passed: true,
      comment: "Запрос клиента уточнён.",
    },
    {
      id: "offer",
      name: "Предложил решение",
      weight: 3,
      passed: false,
      comment: "Предложение недостаточно раскрыто.",
    },
    {
      id: "closing",
      name: "Зафиксировал следующий шаг",
      weight: 2,
      passed: false,
      comment: "Следующий шаг не закреплён жёстко.",
    },
  ],
});
const ONBOARDING_PREVIEW_COMPARE = Object.freeze({
  ai_overall: 8,
  human_overall: 7,
  max_score: 10,
  ai_percent: 80,
  human_percent: 70,
  overall_diff: 25,
  mismatch_count: 1,
  compared_count: 4,
  status_color: "yellow",
  llm_analysis:
    "ИИ и человек в целом согласны по большинству пунктов. Наибольшее внимание стоит обратить на финал разговора: именно там обычно появляются расхождения по трактовке качества следующего шага.",
  rows: [
    {
      id: "greeting",
      name: "Поприветствовал клиента и представился",
      weight: 2,
      ai_passed: true,
      human_passed: true,
      same: true,
      ai_comment: "Приветствие и представление зафиксированы.",
      human_comment: "Этап пройден.",
    },
    {
      id: "need_discovery",
      name: "Выявил потребность",
      weight: 3,
      ai_passed: true,
      human_passed: true,
      same: true,
      ai_comment: "Выявление потребности есть.",
      human_comment: "Потребность уточнена.",
    },
    {
      id: "offer",
      name: "Предложил решение",
      weight: 3,
      ai_passed: false,
      human_passed: false,
      same: true,
      ai_comment: "Решение предложено слабо.",
      human_comment: "Аргументации недостаточно.",
    },
    {
      id: "closing",
      name: "Зафиксировал следующий шаг",
      weight: 2,
      ai_passed: true,
      human_passed: false,
      same: false,
      ai_comment: "Следующий шаг обозначен.",
      human_comment: "Нет жёсткой фиксации следующего шага.",
    },
  ],
});
let _uploadWizardPreviewOnly = false;
let _onboardingTour = {
  active: false,
  stepIndex: 0,
  origin: "auto",
  steps: [],
  target: null,
  metaOpenedByTour: false,
  metaPreviewOnly: false,
  autoStartHandled: false,
  previewActive: false,
  previewSnapshot: null,
};

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

function stopCompareStatusPoll() {
  if (compareStatusPollTimer) {
    clearTimeout(compareStatusPollTimer);
    compareStatusPollTimer = null;
  }
}

function cacheCompareResultForWorkspace(ws, stem, data) {
  if (!ws || !data || !stem) return;
  _compareEvalCache = {
    stem,
    criteria: ws.evaluation_criteria || "",
    aiFp: evaluationFingerprint(ws.evaluation),
    huFp: evaluationFingerprint(ws.human_evaluation),
    data,
  };
}

function applyCompareWorkspaceUpdate(nextWs, stem) {
  if (!nextWs || selectedStem !== stem || !_lastWorkspaceData) return;
  _lastWorkspaceData = {
    ..._lastWorkspaceData,
    comparison_runtime: nextWs.comparison_runtime,
    comparison_state: nextWs.comparison_state,
    human_eval_state: nextWs.human_eval_state || _lastWorkspaceData.human_eval_state,
    permissions: nextWs.permissions || _lastWorkspaceData.permissions,
    auth: nextWs.auth || _lastWorkspaceData.auth,
  };
  _authState = _lastWorkspaceData.auth || _authState;
  if (_lastWorkspaceData.comparison_state && _lastWorkspaceData.comparison_state.payload) {
    _evalTransientState.compareError = "";
    cacheCompareResultForWorkspace(_lastWorkspaceData, stem, _lastWorkspaceData.comparison_state.payload);
  }
  updateWorkspaceHeadStatus(_lastWorkspaceData);
  updateEvalToggleState(_lastWorkspaceData);
  maybeLoadCompareView();
}

function scheduleCompareStatusPoll(stem, criteriaOverride) {
  stopCompareStatusPoll();
  compareStatusPollTimer = setTimeout(async () => {
    if (selectedStem !== stem) return;
    try {
      const crit = resolveWorkspaceCriteriaQuery(stem, criteriaOverride);
      const q = crit ? `?criteria=${encodeURIComponent(crit)}` : "";
      const r = await apiFetch(`${API}/api/workspace/${encodeURIComponent(stem)}${q}`);
      if (!r.ok) return;
      const ws = await r.json();
      if (selectedStem !== stem) return;
      applyCompareWorkspaceUpdate(ws, stem);
      if (ws && ws.comparison_runtime && ws.comparison_runtime.pending) {
        scheduleCompareStatusPoll(stem, criteriaOverride);
      }
    } catch (_) {
      /* keep silent while background compare is still running */
    }
  }, 1500);
}

function clearWorkspaceView() {
  stopWorkspaceJobPoll();
  stopCompareStatusPoll();
  detachTranscriptMedia();
  document.getElementById("workspace-empty").style.display = "flex";
  document.getElementById("workspace").style.display = "none";
  const downloadBtn = document.getElementById("transcript-download");
  if (downloadBtn) {
    downloadBtn.hidden = true;
    downloadBtn.dataset.stem = "";
  }
  lastWorkspaceStem = null;
  lastWorkspaceCriteriaRequested = null;
  _compareEvalCache = null;
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
    if (el === v) {
      el.removeAttribute("poster");
      el.classList.remove("transcript-video--preview");
    }
    el.load();
  }
}

function scoreClass(score) {
  if (score === true) return "score-high";
  if (score === false) return "score-low";
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

function criterionBadgeLabel(weight) {
  const n = Number(weight || 1);
  return `${n} ${n === 1 ? "балл" : n >= 2 && n <= 4 ? "балла" : "баллов"}`;
}

function appendCriterionNameWithBadge(target, name, weight, wrapClassName) {
  const wrap = document.createElement("div");
  wrap.className = wrapClassName;
  const text = document.createElement("span");
  text.className = target.className || "";
  text.textContent = name;
  const badge = document.createElement("span");
  badge.className = "criteria-badge";
  badge.textContent = criterionBadgeLabel(weight);
  wrap.appendChild(text);
  wrap.appendChild(badge);
  target.replaceWith(wrap);
  return wrap;
}

function formatDecisionText(passed) {
  if (passed === true) return "Да";
  if (passed === false) return "НЕТ";
  return "—";
}

function formatDecisionHint(passed) {
  if (passed === true) return "Пункт выполнен";
  if (passed === false) return "Пункт не выполнен";
  return "Нет ответа";
}

function formatShortDate(iso) {
  if (!iso || typeof iso !== "string") return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString("ru-RU", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
  });
}

function isAdmin() {
  return Boolean(_authState && _authState.is_admin);
}

function departmentLabel(code) {
  return DEPARTMENT_LABELS[String(code || "").trim().toUpperCase()] || String(code || "").trim();
}

function currentUserDepartment() {
  return String((_authState && _authState.department) || "").trim().toUpperCase() || "";
}

function itemPermissions(item) {
  return (item && item.permissions) || {};
}

function workspacePermissions(ws) {
  return (ws && ws.permissions) || {};
}

function isReadonlyItem(item) {
  return Boolean(itemPermissions(item).read_only);
}

function checklistBoundToTrainingType(slug) {
  const safeSlug = (slug || "").trim();
  if (!safeSlug) return null;
  return _metaTrainingTypes.find((item) => item.slug === safeSlug) || null;
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

function formatEvaluatedAtParts(iso) {
  const formatted = formatEvaluatedAt(iso);
  if (!formatted || formatted === "—") {
    return { primary: "—", secondary: "" };
  }
  const parts = String(formatted).split(" в ");
  if (parts.length >= 2) {
    return { primary: parts[0], secondary: parts.slice(1).join(" в ") };
  }
  return { primary: formatted, secondary: "" };
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s == null ? "" : String(s);
  return d.innerHTML;
}

function evalStatusIconHtml(state, variant = "") {
  if (state === "pending") {
    return '<span class="eval-status-icon eval-status-icon--spinner" aria-hidden="true"></span>';
  }
  if (state === "waiting") {
    return `
      <span class="eval-status-icon eval-status-icon--waiting" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="9" />
          <path d="M12 7.5v5l3 2" />
        </svg>
      </span>
    `;
  }
  if (state === "success") {
    return `
      <span class="eval-status-icon eval-status-icon--success" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M20 6 9 17l-5-5" />
        </svg>
      </span>
    `;
  }
  if (state === "error") {
    return `
      <span class="eval-status-icon eval-status-icon--error" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
          <path d="m6 6 12 12" />
          <path d="M18 6 6 18" />
        </svg>
      </span>
    `;
  }
  if (variant === "publish") {
    return `
      <span class="eval-status-icon" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M22 2 11 13" />
          <path d="M22 2 15 22 11 13 2 9 22 2Z" />
        </svg>
      </span>
    `;
  }
  return "";
}

function setEvalModeButtonVisual(button, label, state = "", title = "") {
  if (!button) return;
  const icon = evalStatusIconHtml(state);
  button.innerHTML = `
    <span class="eval-mode-btn__label">${escapeHtml(label)}</span>
    <span class="eval-mode-btn__state" aria-hidden="true">${icon}</span>
  `;
  button.title = title || label;
}

function setEvalActionButtonVisual(button, { state = "", title = "", busyTitle = "", icon = "" } = {}) {
  if (!button) return;
  const iconHtml = state === "pending" ? "" : evalStatusIconHtml(state, icon);
  button.innerHTML = `<span class="eval-action-btn__icon" aria-hidden="true">${iconHtml}</span>`;
  if (state === "pending") {
    button.setAttribute("aria-busy", "true");
  } else {
    button.removeAttribute("aria-busy");
  }
  if (state === "pending" && busyTitle) {
    button.title = busyTitle;
    button.setAttribute("aria-label", busyTitle);
  } else if (title) {
    button.title = title;
    button.setAttribute("aria-label", title);
  }
}

function isJobBusy(job) {
  return Boolean(job && (job.status === "queued" || job.status === "running"));
}

function hasAiPipelineFailure(ws) {
  const job = ws && ws.job;
  return Boolean(
    (job && (job.status === "error" || job.status === "cancelled")) ||
      (ws && (ws.transcript_load_error || ws.tone_load_error || ws.criteria_resolution_error)),
  );
}

function isAiPipelineSuccessfullyComplete(ws) {
  const job = ws && ws.job;
  return Boolean(
    ws &&
      ws.video_url &&
      ws.transcript &&
      ws.tone &&
      (ws.ai_evaluation_available || ws.evaluation) &&
      !(job && (job.status === "queued" || job.status === "running")) &&
      !hasAiPipelineFailure(ws),
  );
}

function deriveAiEvalStatus(ws) {
  if (_evalTransientState.aiPending) return "pending";
  const job = ws && ws.job;
  if (isJobBusy(job)) return "pending";
  if (hasAiPipelineFailure(ws)) return "error";
  if (ws && (ws.ai_evaluation_available || ws.evaluation)) return "success";
  if (ws && ws.video_url) return "waiting";
  return "";
}

function deriveCompareStatus(ws) {
  const runtime = (ws && ws.comparison_runtime) || {};
  if (ws && ws.comparison_state && ws.comparison_state.payload) return "success";
  if (runtime.pending || _evalTransientState.publishPending || _evalTransientState.comparePending) return "pending";
  if (runtime.error) return "error";
  if (_evalTransientState.compareError) return "error";
  /* Как у «Человек»: пока сравнения нет — часы ожидания; без чеклиста иконку не показываем */
  if (!ws || !ws.criteria_content || !Array.isArray(ws.criteria_content.criteria) || !ws.criteria_content.criteria.length) {
    return "";
  }
  return "waiting";
}

function deriveHumanStatus(ws) {
  const humanState = (ws && ws.human_eval_state) || {};
  if (humanState.published_at) return "success";
  if (ws && ws.criteria_content && Array.isArray(ws.criteria_content.criteria) && ws.criteria_content.criteria.length) {
    return "waiting";
  }
  return "";
}

function syncEvalModeButtonStates(ws = _lastWorkspaceData) {
  const humanState = (ws && ws.human_eval_state) || {};
  const humanStatus = deriveHumanStatus(ws);
  setEvalModeButtonVisual(
    document.getElementById("eval-mode-human"),
    "Человек",
    humanStatus,
    humanState.published_at ? "Ручной чеклист опубликован" : "Ожидается заполнение ручного чеклиста",
  );
  setEvalModeButtonVisual(
    document.getElementById("eval-mode-ai"),
    "ИИ",
    deriveAiEvalStatus(ws),
    "Оценка ИИ",
  );
  setEvalModeButtonVisual(
    document.getElementById("eval-compare"),
    "Сравнение",
    deriveCompareStatus(ws),
    "Сравнение ИИ и человека",
  );
  setEvalActionButtonVisual(document.getElementById("human-eval-publish"), {
    state: _evalTransientState.publishPending ? "pending" : "",
    icon: "publish",
    title: "Опубликовать ручной чеклист",
    busyTitle: "Публикую ручной чеклист…",
  });
}

let _appMessageResolver = null;

function resolveAppMessageDialog(result) {
  const dlg = document.getElementById("app-message-dialog");
  const resolve = _appMessageResolver;
  _appMessageResolver = null;
  if (dlg && dlg.open) dlg.close();
  if (typeof resolve === "function") resolve(result);
}

function showAppMessageDialog(options = {}) {
  const dlg = document.getElementById("app-message-dialog");
  const titleEl = document.getElementById("app-message-title");
  const textEl = document.getElementById("app-message-text");
  const cancelBtn = document.getElementById("app-message-cancel");
  const okBtn = document.getElementById("app-message-ok");
  if (!dlg || !titleEl || !textEl || !cancelBtn || !okBtn) {
    return Promise.resolve(Boolean(options.confirm) ? false : true);
  }
  if (_appMessageResolver) {
    resolveAppMessageDialog(false);
  }
  const title = String(options.title || (options.confirm ? "Подтверждение" : "Сообщение")).trim();
  const message = String(options.message || "").trim() || "Без текста.";
  const okLabel = String(options.okLabel || (options.confirm ? "Подтвердить" : "Понятно")).trim();
  const cancelLabel = String(options.cancelLabel || "Отмена").trim();
  const isConfirm = Boolean(options.confirm);

  titleEl.textContent = title;
  textEl.textContent = message;
  cancelBtn.hidden = !isConfirm;
  cancelBtn.textContent = cancelLabel;
  okBtn.textContent = okLabel;

  return new Promise((resolve) => {
    _appMessageResolver = resolve;
    dlg.dataset.kind = isConfirm ? "confirm" : "alert";
    if (typeof dlg.showModal === "function") dlg.showModal();
    requestAnimationFrame(() => okBtn.focus());
  });
}

function showAppAlert(message, options = {}) {
  return showAppMessageDialog({ ...options, message, confirm: false });
}

function showAppConfirm(message, options = {}) {
  return showAppMessageDialog({ ...options, message, confirm: true });
}

function setupSystemMessageDialog() {
  const dlg = document.getElementById("app-message-dialog");
  const closeBtn = document.getElementById("app-message-close");
  const cancelBtn = document.getElementById("app-message-cancel");
  const okBtn = document.getElementById("app-message-ok");
  if (!dlg || !closeBtn || !cancelBtn || !okBtn) return;
  closeBtn.addEventListener("click", () => resolveAppMessageDialog(false));
  cancelBtn.addEventListener("click", () => resolveAppMessageDialog(false));
  okBtn.addEventListener("click", () => resolveAppMessageDialog(true));
  dlg.addEventListener("cancel", (e) => {
    e.preventDefault();
    resolveAppMessageDialog(false);
  });
  dlg.addEventListener("click", (e) => {
    if (e.target === dlg) resolveAppMessageDialog(false);
  });
}

function clampNumber(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function currentOnboardingVersion() {
  const raw = Number((_authState && _authState.onboarding_current_version) || ONBOARDING_TOUR_VERSION);
  if (!Number.isFinite(raw)) return ONBOARDING_TOUR_VERSION;
  return Math.max(1, Math.trunc(raw));
}

function getLocalOnboardingSeenVersion() {
  try {
    const raw = window.localStorage.getItem(ONBOARDING_TOUR_STORAGE_KEY);
    const out = Number(raw || 0);
    if (!Number.isFinite(out)) return 0;
    return Math.max(0, Math.trunc(out));
  } catch (_) {
    return 0;
  }
}

function effectiveOnboardingSeenVersion() {
  if (_authState && _authState.auth_enabled && _authState.user) {
    const raw = Number(_authState.onboarding_seen_version || 0);
    return Number.isFinite(raw) ? Math.max(0, Math.trunc(raw)) : 0;
  }
  return getLocalOnboardingSeenVersion();
}

function onboardingCompleted() {
  return effectiveOnboardingSeenVersion() >= currentOnboardingVersion();
}

function updateOnboardingStateLocally(version) {
  const safeVersion = Math.max(0, Math.trunc(Number(version) || 0));
  if (_authState && _authState.auth_enabled && _authState.user) {
    _authState = {
      ..._authState,
      onboarding_seen_version: safeVersion,
      onboarding_current_version: currentOnboardingVersion(),
      onboarding_completed: safeVersion >= currentOnboardingVersion(),
    };
    return;
  }
  try {
    window.localStorage.setItem(ONBOARDING_TOUR_STORAGE_KEY, String(safeVersion));
  } catch (_) {
    /* ignore localStorage failures */
  }
}

async function markOnboardingCompleted() {
  const version = currentOnboardingVersion();
  if (_authState && _authState.auth_enabled && _authState.user) {
    try {
      const r = await apiFetch(`${API}/api/auth/tutorial/complete`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ version }),
      });
      const data = await r.json().catch(() => ({}));
      if (r.ok) {
        _authState = { ..._authState, ...(data || {}) };
      } else {
        updateOnboardingStateLocally(version);
      }
    } catch (_) {
      updateOnboardingStateLocally(version);
    }
  } else {
    updateOnboardingStateLocally(version);
  }
}

function isTourTargetVisible(el) {
  if (!el || typeof el.getBoundingClientRect !== "function") return false;
  const style = window.getComputedStyle(el);
  if (style.display === "none" || style.visibility === "hidden" || style.opacity === "0") return false;
  const rect = el.getBoundingClientRect();
  return rect.width > 0 && rect.height > 0;
}

function resolveFirstVisibleTourTarget(resolvers) {
  for (const resolver of resolvers) {
    const el = typeof resolver === "function" ? resolver() : resolver;
    if (isTourTargetVisible(el)) return el;
  }
  return null;
}

function createOnboardingPreviewFile() {
  return { ...ONBOARDING_TOUR_PREVIEW_FILE };
}

function onboardingPreviewLocationLabel() {
  return (
    (_authState &&
      _authState.location &&
      (_authState.location.crm_name || _authState.location.name || _authState.location.id)) ||
    "Флагман"
  );
}

function onboardingPreviewManagerLabel() {
  return (_authState && (_authState.full_name || _authState.display_name || _authState.user)) || "Новый менеджер";
}

function onboardingPreviewPermissions() {
  return {
    can_save_human_eval: true,
    can_publish_human_eval: true,
    can_compare: true,
    can_re_evaluate: true,
    can_view_ai_log: true,
    can_edit_meta: true,
    can_delete: true,
    can_restore: true,
    can_control_jobs: false,
    can_restart_failed_pipeline: false,
    read_only: false,
    is_admin: isAdmin(),
  };
}

function clonePreviewEvaluation(evaluation) {
  return JSON.parse(JSON.stringify(evaluation));
}

function createOnboardingPreviewWorkspace(kind) {
  const criteriaContent = {
    criteria: ONBOARDING_PREVIEW_CRITERIA.map((item) => ({ ...item })),
  };
  const base = {
    stem: "onboarding-preview",
    video_file: ONBOARDING_TOUR_PREVIEW_FILE.name,
    video_url: "__onboarding_preview__",
    onboarding_preview_media: true,
    meta: {
      display_title: "Пример: запись после загрузки",
      uploaded_at: "2026-04-19T08:30:00Z",
      training_type_name_snapshot: "Визит в салон",
    },
    auth: _authState,
    transcript: clonePreviewEvaluation(ONBOARDING_PREVIEW_TRANSCRIPT),
    tone: clonePreviewEvaluation(ONBOARDING_PREVIEW_TONE),
    job: null,
    permissions: onboardingPreviewPermissions(),
    criteria: { active: ONBOARDING_PREVIEW_CHECKLIST_SLUG, files: [] },
    criteria_content: criteriaContent,
    evaluation_criteria: ONBOARDING_PREVIEW_CHECKLIST_SLUG,
    ai_evaluation_available: true,
    evaluation: clonePreviewEvaluation(ONBOARDING_PREVIEW_AI_EVALUATION),
    human_evaluation: null,
    human_eval_state: {},
    comparison_state: null,
    comparison_runtime: { pending: false, error: null },
    ai_hidden_reason: "",
  };
  switch (kind) {
    case "ai":
      base.human_evaluation = null;
      break;
    case "status":
      base.job = { status: "running", stage: "tone", kind: "pipeline" };
      base.ai_evaluation_available = false;
      base.evaluation = null;
      base.tone = null;
      break;
    case "human":
      base.human_evaluation = null;
      break;
    case "save":
      base.human_evaluation = clonePreviewEvaluation(ONBOARDING_PREVIEW_HUMAN_DRAFT);
      base.human_eval_state = { draft_saved_at: "2026-04-19T08:55:00Z" };
      break;
    case "publish":
      base.human_evaluation = clonePreviewEvaluation(ONBOARDING_PREVIEW_HUMAN_DRAFT);
      base.human_eval_state = { draft_saved_at: "2026-04-19T08:55:00Z" };
      break;
    case "compare":
    case "traffic":
      base.human_evaluation = clonePreviewEvaluation(ONBOARDING_PREVIEW_HUMAN_PUBLISHED);
      base.human_eval_state = {
        draft_saved_at: "2026-04-19T08:55:00Z",
        published_at: "2026-04-19T09:02:00Z",
        compared_at: "2026-04-19T09:04:00Z",
      };
      base.comparison_state = { payload: clonePreviewEvaluation(ONBOARDING_PREVIEW_COMPARE) };
      break;
    default:
      break;
  }
  return base;
}

function syncOnboardingPreviewHumanState(kind) {
  if (kind === "save") {
    _humanEvalDirty = true;
  } else {
    _humanEvalDirty = false;
  }
}

function onboardingPreviewStatusDotClass(kind) {
  if (kind === "status") return "status-dot--spinner";
  if (kind === "compare" || kind === "traffic") return "status-dot--green";
  return "status-dot--off";
}

function renderOnboardingPreviewLibrary(kind) {
  const list = document.getElementById("library-list");
  const empty = document.getElementById("library-empty");
  if (!list || !empty) return;
  list.innerHTML = "";
  empty.style.display = "none";

  const rowWrap = document.createElement("div");
  rowWrap.className = "library-row selected";
  rowWrap.id = "onboarding-preview-library-row";
  rowWrap.dataset.stem = "onboarding-preview";

  const row = document.createElement("button");
  row.type = "button";
  row.className = "library-item";
  row.disabled = true;
  row.title = "Пример записи после загрузки";

  const dotWrap = document.createElement("span");
  dotWrap.className = "status-dot-wrap";
  dotWrap.id = "onboarding-preview-library-status";
  dotWrap.title =
    kind === "status"
      ? "Запись обрабатывается"
      : kind === "compare" || kind === "traffic"
        ? "Сравнение готово"
        : "Запись доступна для ручной оценки";
  const dot = document.createElement("span");
  dot.className = `status-dot ${onboardingPreviewStatusDotClass(kind)}`;
  dotWrap.appendChild(dot);

  const textWrap = document.createElement("div");
  textWrap.className = "library-item-text";
  const name = document.createElement("div");
  name.className = "library-item-name";
  name.textContent = "Пример: запись после загрузки";
  textWrap.appendChild(name);

  row.appendChild(dotWrap);
  row.appendChild(textWrap);
  rowWrap.appendChild(row);

  const deleteBtn = document.createElement("button");
  deleteBtn.type = "button";
  deleteBtn.className = "library-item-delete";
  deleteBtn.id = "onboarding-preview-library-delete";
  deleteBtn.disabled = true;
  deleteBtn.title = "Пометить на удаление";
  deleteBtn.setAttribute("aria-label", "Пометить запись на удаление");
  deleteBtn.innerHTML =
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>';
  rowWrap.appendChild(deleteBtn);

  list.appendChild(rowWrap);
}

function renderOnboardingPreviewWorkspace(kind) {
  const ws = createOnboardingPreviewWorkspace(kind);
  _lastWorkspaceData = ws;
  selectedStem = null;
  lastWorkspaceStem = null;
  lastWorkspaceCriteriaRequested = ws.evaluation_criteria;
  _compareEvalCache = null;
  _showWorkspaceJobLog = false;
  _showCompletedAiPipelineCard = kind === "status";
  _evalTransientState = {
    aiPending: false,
    comparePending: false,
    compareError: "",
    publishPending: false,
  };
  const workspaceEmpty = document.getElementById("workspace-empty");
  const workspace = document.getElementById("workspace");
  const titleEl = document.getElementById("workspace-title");
  const metaEl = document.getElementById("workspace-title-meta");
  if (workspaceEmpty) workspaceEmpty.style.display = "none";
  if (workspace) workspace.style.display = "flex";
  renderOnboardingPreviewLibrary(kind);
  if (titleEl) titleEl.textContent = ws.meta.display_title;
  if (metaEl) {
    renderWorkspaceMetaItems(
      metaEl,
      [
        { text: "Демо-сценарий", tone: "info" },
        { text: onboardingPreviewLocationLabel(), tone: "neutral" },
        { text: "Чеклист: Визит в салон", tone: "neutral" },
      ],
      "Демо-сценарий инструкции",
    );
  }
  updateWorkspaceHeadStatus(ws);
  renderAiPipelinePanel(ws);
  renderMediaAndTranscript(ws);
  renderToneHint(ws.tone, ws.tone_load_error);
  renderChecklistStaleBanner(ws);
  renderEvaluation(ws.evaluation, {
    hasTranscript: kind !== "status",
    criteriaLabel: ws.evaluation_criteria,
    aiAvailable: !!ws.ai_evaluation_available,
    hiddenReason: ws.ai_hidden_reason || "",
  });
  renderHumanEvalForm(ws);
  syncOnboardingPreviewHumanState(kind);
  if (kind === "compare" || kind === "traffic") {
    _currentEvalMode = "compare";
  } else if (kind === "human" || kind === "save" || kind === "publish") {
    _currentEvalMode = "human";
  } else {
    _currentEvalMode = "ai";
  }
  updateEvalToolbar(ws);
  updateEvalToggleState(ws);
  syncEvalModeButtonStates(ws);
  if (kind === "compare" || kind === "traffic") {
    renderCompareResult(clonePreviewEvaluation(ONBOARDING_PREVIEW_COMPARE));
  } else {
    const summaryEl = document.getElementById("compare-summary");
    const tableWrap = document.getElementById("compare-table-wrap");
    const analysisEl = document.getElementById("compare-analysis");
    if (summaryEl) summaryEl.innerHTML = "";
    if (tableWrap) tableWrap.innerHTML = "";
    if (analysisEl) analysisEl.innerHTML = "";
  }
}

function ensureOnboardingWorkspacePreview(kind) {
  const workspace = document.getElementById("workspace");
  const hasRealWorkspace = Boolean(selectedStem && _lastWorkspaceData && workspace && workspace.style.display !== "none");
  if (hasRealWorkspace) {
    _onboardingTour.previewActive = false;
    return false;
  }
  if (!_onboardingTour.previewActive) {
    _onboardingTour.previewSnapshot = {
      selectedStem,
      lastWorkspaceStem,
      lastWorkspaceCriteriaRequested,
      lastWorkspaceData: _lastWorkspaceData,
      currentEvalMode: _currentEvalMode,
      showWorkspaceJobLog: _showWorkspaceJobLog,
      showCompletedAiPipelineCard: _showCompletedAiPipelineCard,
      humanEvalDirty: _humanEvalDirty,
      humanEvalDraftBaseline: _humanEvalDraftBaseline,
      evalTransientState: { ..._evalTransientState },
      compareEvalCache: _compareEvalCache,
    };
    _onboardingTour.previewActive = true;
  }
  renderOnboardingPreviewWorkspace(kind);
  return true;
}

function restoreOnboardingWorkspacePreview() {
  if (!_onboardingTour.previewActive) return;
  const snap = _onboardingTour.previewSnapshot || {};
  selectedStem = snap.selectedStem ?? null;
  lastWorkspaceStem = snap.lastWorkspaceStem ?? null;
  lastWorkspaceCriteriaRequested = snap.lastWorkspaceCriteriaRequested ?? null;
  _lastWorkspaceData = snap.lastWorkspaceData ?? null;
  _currentEvalMode = snap.currentEvalMode || "ai";
  _showWorkspaceJobLog = Boolean(snap.showWorkspaceJobLog);
  _showCompletedAiPipelineCard = Boolean(snap.showCompletedAiPipelineCard);
  _humanEvalDirty = Boolean(snap.humanEvalDirty);
  _humanEvalDraftBaseline = snap.humanEvalDraftBaseline || "[]";
  _evalTransientState = snap.evalTransientState || {
    aiPending: false,
    comparePending: false,
    compareError: "",
    publishPending: false,
  };
  _compareEvalCache = snap.compareEvalCache || null;
  _onboardingTour.previewSnapshot = null;
  _onboardingTour.previewActive = false;
  if (selectedStem && _lastWorkspaceData) {
    const workspaceEmpty = document.getElementById("workspace-empty");
    const workspace = document.getElementById("workspace");
    const titleEl = document.getElementById("workspace-title");
    const metaEl = document.getElementById("workspace-title-meta");
    if (workspaceEmpty) workspaceEmpty.style.display = "none";
    if (workspace) workspace.style.display = "flex";
    if (titleEl) {
      const disp = _lastWorkspaceData.meta && _lastWorkspaceData.meta.display_title;
      titleEl.textContent = (disp && String(disp).trim()) || _lastWorkspaceData.video_file || _lastWorkspaceData.stem || "—";
    }
    if (metaEl) renderWorkspaceMetaLine(metaEl, _lastWorkspaceData);
    renderLibrary(_allLibraryItems);
    updateWorkspaceHeadStatus(_lastWorkspaceData);
    renderAiPipelinePanel(_lastWorkspaceData);
    renderEvaluation(_lastWorkspaceData.evaluation, {
      hasTranscript: !!_lastWorkspaceData.transcript,
      criteriaLabel: _lastWorkspaceData.evaluation_criteria || _lastWorkspaceData.criteria?.active || "",
      aiAvailable: !!_lastWorkspaceData.ai_evaluation_available,
      hiddenReason: _lastWorkspaceData.ai_hidden_reason || "",
    });
    renderHumanEvalForm(_lastWorkspaceData);
    updateEvalToolbar(_lastWorkspaceData);
    updateEvalToggleState(_lastWorkspaceData);
    syncEvalModeButtonStates(_lastWorkspaceData);
    maybeLoadCompareView();
  } else {
    clearWorkspaceView();
    renderLibrary(_allLibraryItems);
  }
}

function closeOnboardingUploadPreview() {
  const dlg = document.getElementById("upload-wizard-dialog");
  if (dlg && dlg.open && _uploadWizardPreviewOnly) dlg.close();
  _uploadWizardPreviewOnly = false;
}

function closeOnboardingMetaDialog() {
  const dlg = document.getElementById("meta-dialog");
  const saveBtn = document.getElementById("meta-save");
  if (saveBtn && _onboardingTour.metaPreviewOnly) {
    saveBtn.hidden = false;
    saveBtn.disabled = false;
  }
  if (dlg && dlg.open && (_onboardingTour.metaOpenedByTour || _onboardingTour.metaPreviewOnly)) dlg.close();
  _onboardingTour.metaOpenedByTour = false;
  _onboardingTour.metaPreviewOnly = false;
}

async function openOnboardingUploadPreview() {
  const dlg = document.getElementById("upload-wizard-dialog");
  if (dlg && dlg.open && !_uploadWizardPreviewOnly) return;
  await openUploadWizard([createOnboardingPreviewFile()], { previewOnly: true });
}

function openOnboardingMetaPreview() {
  const dlg = document.getElementById("meta-dialog");
  const stemCode = document.getElementById("meta-dialog-stem-code");
  const saveBtn = document.getElementById("meta-save");
  if (!dlg || !stemCode || !saveBtn) return;
  fillMetaForm({
    managers: [{ id: "demo-manager", name: onboardingPreviewManagerLabel() }],
    locations: [{ id: "demo-location", name: onboardingPreviewLocationLabel() }],
    trainingTypes: [
      {
        slug: "visit_salon",
        name: "Визит в салон",
        department: currentUserDepartment() || "ОО",
        checklist_slug: ONBOARDING_PREVIEW_CHECKLIST_SLUG,
      },
    ],
    meta: {
      display_title: "Пример: запись после загрузки",
      manager_id: "demo-manager",
      location_id: "demo-location",
      training_type_slug: "visit_salon",
      interaction_date: "2026-04-19",
      tags: ["демо", "инструкция"],
    },
    videoFileFallback: ONBOARDING_TOUR_PREVIEW_FILE.name,
    checklistSlug: ONBOARDING_PREVIEW_CHECKLIST_SLUG,
  });
  stemCode.textContent = "demo-preview";
  saveBtn.hidden = true;
  saveBtn.disabled = true;
  _onboardingTour.metaPreviewOnly = true;
  if (typeof dlg.showModal === "function" && !dlg.open) dlg.showModal();
}

function canShowOnboardingMetaDialog() {
  return Boolean(selectedStem && _lastWorkspaceData && workspacePermissions(_lastWorkspaceData).can_edit_meta);
}

function resolveOnboardingStatusTarget() {
  return resolveFirstVisibleTourTarget([
    () => document.getElementById("ai-pipeline-card"),
    () => document.getElementById("workspace-head-status-wrap"),
    () => document.getElementById("workspace"),
  ]);
}

function resolveOnboardingLibraryRowTarget() {
  return resolveFirstVisibleTourTarget([
    () => document.getElementById("onboarding-preview-library-row"),
    () => document.querySelector("#library-list .library-row.selected"),
    () => document.getElementById("library-list"),
  ]);
}

function resolveOnboardingLibraryDeleteTarget() {
  return resolveFirstVisibleTourTarget([
    () => document.querySelector("#onboarding-preview-library-delete svg"),
    () => document.getElementById("onboarding-preview-library-delete"),
    () => document.querySelector("#library-list .library-item-delete svg"),
    () => document.querySelector("#library-list .library-item-delete"),
    resolveOnboardingLibraryRowTarget,
  ]);
}

function resolveOnboardingWorkspaceLayoutTarget() {
  return resolveFirstVisibleTourTarget([
    () => document.querySelector(".workspace-layout"),
    () => document.getElementById("workspace"),
  ]);
}

function resolveOnboardingTabsTarget() {
  return resolveFirstVisibleTourTarget([
    () => document.getElementById("eval-mode-toggle"),
    resolveOnboardingWorkspaceLayoutTarget,
  ]);
}

function resolveOnboardingAiChecklistTarget() {
  return resolveFirstVisibleTourTarget([
    () => document.getElementById("eval-criteria-table"),
    () => document.getElementById("eval-ai-view"),
    resolveOnboardingWorkspaceLayoutTarget,
  ]);
}

function resolveOnboardingAuthSettingsTarget() {
  return resolveFirstVisibleTourTarget([
    () => document.querySelector(".auth-settings"),
    () => document.getElementById("auth-user-bar"),
    resolveOnboardingReplayTarget,
  ]);
}

function resolveOnboardingChecklistTarget() {
  return resolveFirstVisibleTourTarget([
    () => document.getElementById("criteria-select"),
    () => document.getElementById("meta-dialog"),
    () => document.getElementById("workspace"),
  ]);
}

function resolveOnboardingManualTarget() {
  return resolveFirstVisibleTourTarget([
    () => document.getElementById("human-eval-form"),
    () => document.getElementById("eval-human-view"),
    () => document.getElementById("workspace"),
  ]);
}

function resolveOnboardingSaveTarget() {
  return resolveFirstVisibleTourTarget([
    () => document.getElementById("human-eval-save"),
    resolveOnboardingManualTarget,
  ]);
}

function resolveOnboardingPublishTarget() {
  return resolveFirstVisibleTourTarget([
    () => document.getElementById("human-eval-publish"),
    resolveOnboardingManualTarget,
  ]);
}

function resolveOnboardingCompareTarget() {
  return resolveFirstVisibleTourTarget([
    () => document.getElementById("eval-compare-view"),
    () => document.getElementById("eval-compare"),
    () => document.getElementById("workspace"),
  ]);
}

function resolveOnboardingTrafficLightTarget() {
  return resolveFirstVisibleTourTarget([
    () => document.getElementById("workspace-head-status-wrap"),
    () => document.querySelector("#library-list .status-dot-wrap"),
    () => document.getElementById("workspace"),
  ]);
}

function resolveOnboardingReplayTarget() {
  return resolveFirstVisibleTourTarget([
    () => document.getElementById("auth-tour-button"),
    () => document.getElementById("workspace-tour-replay"),
    () => document.getElementById("workspace-empty"),
  ]);
}

function buildOnboardingSteps() {
  return [
    {
      id: "welcome",
      title: "Добро пожаловать",
      text:
        "Система проведет вас по полному циклу работы с записью: загрузка, ручная проверка по чеклисту, публикация результата и сравнение с оценкой ИИ. На каждом шаге будет подсвечена нужная область интерфейса.",
      target: () => null,
      nextLabel: "Начать",
      prepare: async () => {
        closeOnboardingMetaDialog();
        closeOnboardingUploadPreview();
        restoreOnboardingWorkspacePreview();
      },
    },
    {
      id: "upload",
      title: "Шаг 1. Загрузка записи",
      text:
        "Начните отсюда. Выберите видео или аудио либо перетащите файл в эту область. После выбора файла откроется окно загрузки, где нужно указать параметры записи.",
      prepare: async () => {
        closeOnboardingMetaDialog();
        closeOnboardingUploadPreview();
        restoreOnboardingWorkspacePreview();
      },
      target: () => document.getElementById("dropzone"),
    },
    {
      id: "wizard",
      title: "Параметры записи",
      text:
        "Здесь указываются отдел и тип тренировки. От этого выбора зависит, какой чеклист будет применяться для ручной оценки и для сравнения с ИИ.",
      detail:
        "Перед загрузкой система проверяет, нет ли такого файла в базе, чтобы избежать дублей. В инструкции окно показано как пример и не запускает реальную загрузку.",
      target: () => document.getElementById("upload-wizard-dialog"),
      prepare: async () => {
        closeOnboardingMetaDialog();
        restoreOnboardingWorkspacePreview();
        await openOnboardingUploadPreview();
      },
    },
    {
      id: "library-row",
      title: "После загрузки запись появится слева",
      text:
        "Каждая загруженная запись появляется в библиотеке слева. Здесь видно название записи, текущий статус обработки и можно быстро открыть нужный разговор в рабочей области.",
      detail:
        "Когда запись обрабатывается, статус показывает, что система ещё строит транскрипт, анализирует аудио или готовит оценку ИИ.",
      target: resolveOnboardingLibraryRowTarget,
      prepare: async () => {
        closeOnboardingUploadPreview();
        closeOnboardingMetaDialog();
        ensureOnboardingWorkspacePreview("status");
      },
    },
    {
      id: "status",
      title: "Что происходит после загрузки",
      text:
        "После загрузки запись поступает в обработку. Здесь можно отслеживать статус транскрипции, анализа аудио и подготовки оценки ИИ.",
      detail:
        "Пока запись обрабатывается, вы уже можете перейти к ручному чеклисту и начать проверку. Инструкция не симулирует прогресс: она показывает, где смотреть реальные статусы.",
      target: resolveOnboardingStatusTarget,
      prepare: async () => {
        closeOnboardingUploadPreview();
        closeOnboardingMetaDialog();
        ensureOnboardingWorkspacePreview("status");
      },
    },
    {
      id: "workspace-layout",
      title: "Рабочие области записи",
      text:
        "После открытия записи основная работа идёт в двух областях: слева находятся плеер и транскрипт, справа — вся оценка по чеклисту.",
      detail:
        "Это основной экран проверки: обычно менеджер смотрит запись и транскрипт слева, а решения по чеклисту принимает справа.",
      cardPlacement: "center-target",
      target: resolveOnboardingWorkspaceLayoutTarget,
      prepare: async () => {
        closeOnboardingMetaDialog();
        closeOnboardingUploadPreview();
        ensureOnboardingWorkspacePreview("ai");
      },
    },
    {
      id: "tabs",
      title: "Три вкладки оценки",
      text:
        "В правой колонке есть три вкладки. “Человек” — это ручной чеклист, который менеджер заполняет сам. Вкладки “ИИ” и “Сравнение” становятся доступны по смыслу после публикации ручной оценки.",
      detail:
        "Правильный сценарий работы такой: сначала менеджер просматривает запись и заполняет свою оценку во вкладке “Человек”. Только после публикации он видит, как ИИ оценил диалог, и может открыть сравнение двух результатов.",
      target: resolveOnboardingTabsTarget,
      prepare: async () => {
        closeOnboardingMetaDialog();
        closeOnboardingUploadPreview();
        ensureOnboardingWorkspacePreview("ai");
      },
    },
    {
      id: "ai-checklist",
      title: "Вкладка ИИ и таймкоды",
      text:
        "После публикации ручной оценки во вкладке “ИИ” становится видна автоматическая оценка по чеклисту. Здесь можно быстро посмотреть, как система трактует каждый критерий.",
      detail:
        "Если рядом с пунктом есть таймкоды, по ним можно нажимать и сразу переходить к нужному месту записи. Это удобно для перепроверки спорных моментов уже после того, как менеджер зафиксировал собственное решение.",
      target: resolveOnboardingAiChecklistTarget,
      prepare: async () => {
        closeOnboardingMetaDialog();
        closeOnboardingUploadPreview();
        const usingPreview = ensureOnboardingWorkspacePreview("ai");
        if (!usingPreview && selectedStem && _lastWorkspaceData) switchEvalMode("ai");
      },
    },
    {
      id: "human",
      title: "Шаг 2. Ручной чеклист",
      text:
        "Во вкладке “Человек” заполняется ручная оценка. По каждому критерию выберите результат и при необходимости добавьте комментарий.",
      detail:
        "Комментарий особенно полезен там, где решение может вызвать вопросы при последующем сравнении или внутреннем ревью.",
      target: resolveOnboardingManualTarget,
      prepare: async () => {
        closeOnboardingUploadPreview();
        closeOnboardingMetaDialog();
        const usingPreview = ensureOnboardingWorkspacePreview("human");
        if (!usingPreview && selectedStem && _lastWorkspaceData) switchEvalMode("human");
      },
    },
    {
      id: "save",
      title: "Сохранение черновика",
      text:
        "Кнопка “Сохранить” фиксирует текущий вариант оценки как черновик. Это удобно, если нужно вернуться позже, показать результат коллеге или сначала сделать дополнительную проверку.",
      detail:
        "Кнопка появляется, когда в ручной оценке есть изменения. Сначала сохраните черновик, затем станет доступна публикация.",
      target: resolveOnboardingSaveTarget,
      prepare: async () => {
        closeOnboardingUploadPreview();
        closeOnboardingMetaDialog();
        const usingPreview = ensureOnboardingWorkspacePreview("save");
        if (!usingPreview && selectedStem && _lastWorkspaceData) switchEvalMode("human");
      },
    },
    {
      id: "publish",
      title: "Шаг 3. Публикация",
      text:
        "Публикация завершает ручную оценку. После этого редактирование будет заблокировано, а система автоматически подготовит сравнение с ИИ.",
      detail:
        "Перед публикацией убедитесь, что чеклист заполнен полностью и проверен: вернуться к редактированию уже нельзя.",
      target: resolveOnboardingPublishTarget,
      prepare: async () => {
        closeOnboardingUploadPreview();
        closeOnboardingMetaDialog();
        const usingPreview = ensureOnboardingWorkspacePreview("publish");
        if (!usingPreview && selectedStem && _lastWorkspaceData) switchEvalMode("human");
      },
    },
    {
      id: "compare",
      title: "Шаг 4. Сравнение с ИИ",
      text:
        "После публикации становится доступен режим сравнения. Здесь показываются итоговые баллы ИИ и человека, доля несовпавших ответов по чеклисту и детализация по каждому критерию.",
      detail:
        "Процент расхождения считается просто: сколько ответов ИИ и человека не совпало, делим на общее число критериев с ответами. Несогласованные критерии видны в таблице ниже.",
      target: resolveOnboardingCompareTarget,
      prepare: async () => {
        closeOnboardingUploadPreview();
        closeOnboardingMetaDialog();
        const usingPreview = ensureOnboardingWorkspacePreview("compare");
        if (!usingPreview && selectedStem && _lastWorkspaceData) switchEvalMode("compare");
      },
    },
    {
      id: "account-settings",
      title: "Настройки аккаунта",
      text:
        "Внизу слева находятся действия аккаунта. Здесь можно открыть настройки профиля, повторно запустить инструкцию и при необходимости выйти из системы.",
      target: resolveOnboardingAuthSettingsTarget,
      prepare: async () => {
        closeOnboardingUploadPreview();
        closeOnboardingMetaDialog();
        ensureOnboardingWorkspacePreview("traffic");
      },
    },
    {
      id: "delete-video",
      title: "Пометка на удаление",
      text:
        "Обычный пользователь не удаляет запись окончательно. Он только помечает её на удаление, чтобы убрать из основного рабочего списка.",
      detail:
        "Такую запись потом можно вернуть. Окончательное удаление выполняет модератор, поэтому для обычного пользователя это безопасный soft-delete, а не безвозвратное удаление.",
      target: resolveOnboardingLibraryDeleteTarget,
      prepare: async () => {
        closeOnboardingUploadPreview();
        closeOnboardingMetaDialog();
        ensureOnboardingWorkspacePreview("traffic");
      },
    },
    {
      id: "traffic-light",
      title: "Как читать светофор",
      text:
        "Светофор показывает долю несовпавших ответов между опубликованной ручной оценкой и оценкой ИИ.",
      bullets: [
        "Серый — ручная оценка еще не опубликована.",
        "Индикатор обработки — запись, оценка или сравнение еще считаются.",
        "Зеленый — не совпало менее 20% ответов.",
        "Желтый — не совпало от 20% до 40% ответов.",
        "Красный — не совпало 40% ответов и более.",
      ],
      target: resolveOnboardingTrafficLightTarget,
      prepare: async () => {
        closeOnboardingUploadPreview();
        closeOnboardingMetaDialog();
        ensureOnboardingWorkspacePreview("traffic");
      },
    },
    {
      id: "finish",
      title: "Готово",
      text:
        "Теперь вы знаете полный сценарий работы: загрузка записи, ручной чеклист, публикация результата и сравнение с ИИ. Инструкцию можно открыть повторно в любой момент из меню аккаунта.",
      target: resolveOnboardingReplayTarget,
      nextLabel: "Завершить",
      prepare: async () => {
        closeOnboardingUploadPreview();
        closeOnboardingMetaDialog();
        ensureOnboardingWorkspacePreview("traffic");
      },
    },
  ];
}

function positionOnboardingCard(card, targetRect) {
  if (!card) return;
  const margin = 12;
  const gap = 18;
  const currentStep =
    _onboardingTour && Array.isArray(_onboardingTour.steps)
      ? _onboardingTour.steps[_onboardingTour.stepIndex] || null
      : null;
  const cardRect = card.getBoundingClientRect();
  const cardWidth = cardRect.width || Math.min(384, window.innerWidth - margin * 2);
  const cardHeight = cardRect.height || 260;
  let top = (window.innerHeight - cardHeight) / 2;
  let left = (window.innerWidth - cardWidth) / 2;
  if (targetRect) {
    if (currentStep && currentStep.cardPlacement === "center-target") {
      left = clampNumber(
        targetRect.left + (targetRect.width - cardWidth) / 2,
        margin,
        window.innerWidth - cardWidth - margin,
      );
      top = clampNumber(
        targetRect.top + (targetRect.height - cardHeight) / 2,
        margin,
        window.innerHeight - cardHeight - margin,
      );
    } else {
      const rightSpace = window.innerWidth - targetRect.right - gap - margin;
      const leftSpace = targetRect.left - gap - margin;
      if (rightSpace >= cardWidth) {
        left = targetRect.right + gap;
        top = clampNumber(targetRect.top, margin, window.innerHeight - cardHeight - margin);
      } else if (leftSpace >= cardWidth) {
        left = targetRect.left - cardWidth - gap;
        top = clampNumber(targetRect.top, margin, window.innerHeight - cardHeight - margin);
      } else {
        left = clampNumber(
          targetRect.left + (targetRect.width - cardWidth) / 2,
          margin,
          window.innerWidth - cardWidth - margin,
        );
        if (window.innerHeight - targetRect.bottom - gap - margin >= cardHeight) {
          top = targetRect.bottom + gap;
        } else {
          top = targetRect.top - cardHeight - gap;
        }
        top = clampNumber(top, margin, window.innerHeight - cardHeight - margin);
      }
    }
  }
  card.style.left = `${Math.round(left)}px`;
  card.style.top = `${Math.round(top)}px`;
}

function setOnboardingShadeRect(el, left, top, width, height) {
  if (!el) return;
  el.style.left = `${Math.max(0, Math.round(left))}px`;
  el.style.top = `${Math.max(0, Math.round(top))}px`;
  el.style.width = `${Math.max(0, Math.round(width))}px`;
  el.style.height = `${Math.max(0, Math.round(height))}px`;
}

function updateOnboardingOverlayLayout() {
  const root = document.getElementById("onboarding-tour");
  const card = document.getElementById("onboarding-tour-card");
  const spotlight = document.getElementById("onboarding-tour-spotlight");
  const topShade = document.getElementById("onboarding-tour-shade-top");
  const rightShade = document.getElementById("onboarding-tour-shade-right");
  const bottomShade = document.getElementById("onboarding-tour-shade-bottom");
  const leftShade = document.getElementById("onboarding-tour-shade-left");
  const topLeftCornerShade = document.getElementById("onboarding-tour-shade-corner-tl");
  const topRightCornerShade = document.getElementById("onboarding-tour-shade-corner-tr");
  const bottomRightCornerShade = document.getElementById("onboarding-tour-shade-corner-br");
  const bottomLeftCornerShade = document.getElementById("onboarding-tour-shade-corner-bl");
  if (!root || !root.open || !card) return;
  const target = isTourTargetVisible(_onboardingTour.target) ? _onboardingTour.target : null;
  const clearSecondaryShades = () => {
    setOnboardingShadeRect(rightShade, 0, 0, 0, 0);
    setOnboardingShadeRect(bottomShade, 0, 0, 0, 0);
    setOnboardingShadeRect(leftShade, 0, 0, 0, 0);
    setOnboardingShadeRect(topLeftCornerShade, 0, 0, 0, 0);
    setOnboardingShadeRect(topRightCornerShade, 0, 0, 0, 0);
    setOnboardingShadeRect(bottomRightCornerShade, 0, 0, 0, 0);
    setOnboardingShadeRect(bottomLeftCornerShade, 0, 0, 0, 0);
  };
  if (!target) {
    if (spotlight) spotlight.hidden = true;
    setOnboardingShadeRect(topShade, 0, 0, window.innerWidth, window.innerHeight);
    if (topShade) {
      topShade.style.clipPath = "none";
      topShade.style.webkitClipPath = "none";
    }
    clearSecondaryShades();
    positionOnboardingCard(card, null);
    return;
  }
  const rect = target.getBoundingClientRect();
  const targetRadius = Math.max(14, parseFloat(window.getComputedStyle(target).borderRadius || "18") || 18);
  const padding = 10;
  const top = clampNumber(rect.top - padding, 0, window.innerHeight);
  const left = clampNumber(rect.left - padding, 0, window.innerWidth);
  const right = clampNumber(rect.right + padding, 0, window.innerWidth);
  const bottom = clampNumber(rect.bottom + padding, 0, window.innerHeight);
  const width = Math.max(0, right - left);
  const height = Math.max(0, bottom - top);
  const spotlightRadius = Math.round(targetRadius + 4);
  if (spotlight) {
    spotlight.hidden = false;
    spotlight.style.left = `${Math.round(left)}px`;
    spotlight.style.top = `${Math.round(top)}px`;
    spotlight.style.width = `${Math.round(width)}px`;
    spotlight.style.height = `${Math.round(height)}px`;
    spotlight.style.borderRadius = `${spotlightRadius}px`;
  }
  setOnboardingShadeRect(topShade, 0, 0, window.innerWidth, window.innerHeight);
  if (topShade) {
    const path = [
      `M 0 0 H ${window.innerWidth} V ${window.innerHeight} H 0 Z`,
      `M ${left + spotlightRadius} ${top}`,
      `H ${right - spotlightRadius}`,
      `A ${spotlightRadius} ${spotlightRadius} 0 0 1 ${right} ${top + spotlightRadius}`,
      `V ${bottom - spotlightRadius}`,
      `A ${spotlightRadius} ${spotlightRadius} 0 0 1 ${right - spotlightRadius} ${bottom}`,
      `H ${left + spotlightRadius}`,
      `A ${spotlightRadius} ${spotlightRadius} 0 0 1 ${left} ${bottom - spotlightRadius}`,
      `V ${top + spotlightRadius}`,
      `A ${spotlightRadius} ${spotlightRadius} 0 0 1 ${left + spotlightRadius} ${top}`,
      "Z",
    ].join(" ");
    const clipPath = `path(evenodd, "${path}")`;
    topShade.style.clipPath = clipPath;
    topShade.style.webkitClipPath = clipPath;
  }
  clearSecondaryShades();
  positionOnboardingCard(card, { top, left, right, bottom, width, height });
}

function syncOnboardingReplayUi() {
  const label = onboardingCompleted() ? "Повторить инструкцию" : "Показать инструкцию";
  const authBtn = document.getElementById("auth-tour-button");
  const replayBtn = document.getElementById("workspace-tour-replay");
  if (authBtn) {
    authBtn.title = label;
    authBtn.setAttribute("aria-label", label);
    authBtn.disabled = _onboardingTour.active;
  }
  if (replayBtn) {
    replayBtn.textContent = label;
    replayBtn.disabled = _onboardingTour.active;
  }
}

async function renderOnboardingStep(index) {
  if (!_onboardingTour.active) return;
  const steps = _onboardingTour.steps;
  if (!steps.length || index < 0 || index >= steps.length) return;
  const root = document.getElementById("onboarding-tour");
  const stepEl = document.getElementById("onboarding-tour-step");
  const titleEl = document.getElementById("onboarding-tour-title");
  const textEl = document.getElementById("onboarding-tour-text");
  const detailEl = document.getElementById("onboarding-tour-detail");
  const legendEl = document.getElementById("onboarding-tour-legend");
  const prevBtn = document.getElementById("onboarding-tour-prev");
  const nextBtn = document.getElementById("onboarding-tour-next");
  const skipBtn = document.getElementById("onboarding-tour-skip");
  if (!root || !stepEl || !titleEl || !textEl || !detailEl || !legendEl || !prevBtn || !nextBtn || !skipBtn) return;
  _onboardingTour.stepIndex = index;
  const step = steps[index];
  if (typeof step.prepare === "function") {
    await step.prepare();
  }
  if (!_onboardingTour.active || _onboardingTour.stepIndex !== index) return;
  await new Promise((resolve) => requestAnimationFrame(resolve));
  const target = typeof step.target === "function" ? step.target() : null;
  _onboardingTour.target = target || null;
  stepEl.textContent = `${index + 1} из ${steps.length}`;
  titleEl.textContent = step.title || "Инструкция";
  textEl.textContent = step.text || "";
  if (step.detail) {
    detailEl.hidden = false;
    detailEl.textContent = step.detail;
  } else {
    detailEl.hidden = true;
    detailEl.textContent = "";
  }
  if (Array.isArray(step.bullets) && step.bullets.length) {
    legendEl.hidden = false;
    legendEl.innerHTML = `<ul>${step.bullets.map((item) => `<li>${escapeHtml(String(item))}</li>`).join("")}</ul>`;
  } else {
    legendEl.hidden = true;
    legendEl.innerHTML = "";
  }
  prevBtn.hidden = index === 0;
  prevBtn.disabled = index === 0;
  skipBtn.hidden = index === steps.length - 1;
  nextBtn.textContent = step.nextLabel || (index === steps.length - 1 ? "Завершить" : "Далее");
  try {
    if (root.open) root.close();
    if (typeof root.showModal === "function") root.showModal();
  } catch (_) {
    /* ignore dialog errors */
  }
  if (isTourTargetVisible(target)) {
    try {
      target.scrollIntoView({ block: "center", inline: "nearest", behavior: "smooth" });
    } catch (_) {
      /* noop */
    }
  }
  requestAnimationFrame(() => {
    updateOnboardingOverlayLayout();
    nextBtn.focus();
  });
}

async function stopOnboardingTour(options = {}) {
  const { complete = false } = options;
  const root = document.getElementById("onboarding-tour");
  const spotlight = document.getElementById("onboarding-tour-spotlight");
  if (!_onboardingTour.active) return;
  _onboardingTour.active = false;
  _onboardingTour.target = null;
  _onboardingTour.autoStartHandled = true;
  closeOnboardingUploadPreview();
  closeOnboardingMetaDialog();
  restoreOnboardingWorkspacePreview();
  if (root && root.open) root.close();
  if (spotlight) spotlight.hidden = true;
  if (complete) {
    await markOnboardingCompleted();
  }
  syncOnboardingReplayUi();
}

async function startOnboardingTour(options = {}) {
  const { origin = "auto" } = options;
  if (_onboardingTour.active) return;
  if (_authState && _authState.auth_enabled && !_authState.user) return;
  if (_authState && _authState.auth_enabled && _authState.profile_complete === false) {
    await openProfileDialog();
    return;
  }
  _onboardingTour.origin = origin;
  _onboardingTour.steps = buildOnboardingSteps();
  _onboardingTour.stepIndex = 0;
  _onboardingTour.target = null;
  _onboardingTour.metaOpenedByTour = false;
  _onboardingTour.active = true;
  syncOnboardingReplayUi();
  await renderOnboardingStep(0);
}

async function goToOnboardingStep(nextIndex) {
  if (!_onboardingTour.active) return;
  const maxIndex = _onboardingTour.steps.length - 1;
  const safeIndex = clampNumber(nextIndex, 0, maxIndex);
  await renderOnboardingStep(safeIndex);
}

async function advanceOnboardingTour() {
  if (!_onboardingTour.active) return;
  const isLast = _onboardingTour.stepIndex >= _onboardingTour.steps.length - 1;
  if (isLast) {
    await stopOnboardingTour({ complete: true });
    return;
  }
  await goToOnboardingStep(_onboardingTour.stepIndex + 1);
}

function canAutoStartOnboardingTour() {
  if (_onboardingTour.active || _onboardingTour.autoStartHandled) return false;
  if (_authState && _authState.auth_enabled && !_authState.user) return false;
  if (_authState && _authState.auth_enabled && _authState.profile_complete === false) return false;
  if (document.getElementById("profile-dialog")?.open) return false;
  return !onboardingCompleted();
}

async function maybeAutoStartOnboardingTour() {
  if (!canAutoStartOnboardingTour()) return false;
  _onboardingTour.autoStartHandled = true;
  await startOnboardingTour({ origin: "auto" });
  return true;
}

function setupOnboardingTour() {
  const authBtn = document.getElementById("auth-tour-button");
  const replayBtn = document.getElementById("workspace-tour-replay");
  const prevBtn = document.getElementById("onboarding-tour-prev");
  const nextBtn = document.getElementById("onboarding-tour-next");
  const skipBtn = document.getElementById("onboarding-tour-skip");
  const root = document.getElementById("onboarding-tour");
  authBtn?.addEventListener("click", () => {
    _onboardingTour.autoStartHandled = true;
    void startOnboardingTour({ origin: "manual" });
  });
  replayBtn?.addEventListener("click", () => {
    _onboardingTour.autoStartHandled = true;
    void startOnboardingTour({ origin: "manual" });
  });
  prevBtn?.addEventListener("click", () => void goToOnboardingStep(_onboardingTour.stepIndex - 1));
  nextBtn?.addEventListener("click", () => void advanceOnboardingTour());
  skipBtn?.addEventListener("click", () => void stopOnboardingTour());
  root?.addEventListener("cancel", (e) => {
    e.preventDefault();
    void stopOnboardingTour();
  });
  document.addEventListener(
    "keydown",
    (e) => {
      if (!_onboardingTour.active) return;
      if (e.key === "Escape") {
        e.preventDefault();
        void stopOnboardingTour();
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        void advanceOnboardingTour();
      } else if (e.key === "ArrowLeft" && _onboardingTour.stepIndex > 0) {
        e.preventDefault();
        void goToOnboardingStep(_onboardingTour.stepIndex - 1);
      }
    },
    true,
  );
  window.addEventListener("resize", () => {
    if (_onboardingTour.active) updateOnboardingOverlayLayout();
  });
  document.addEventListener(
    "scroll",
    () => {
      if (_onboardingTour.active) updateOnboardingOverlayLayout();
    },
    true,
  );
  syncOnboardingReplayUi();
}

function appendEvidenceChipButtons(wrap, segments) {
  let n = 0;
  for (const { t0, t1 } of segments) {
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
    wrap.appendChild(btn);
    n += 1;
  }
  return n;
}

function makeEvidenceChipsWrapFromSegments(segments) {
  const wrap = document.createElement("span");
  wrap.className = "evidence-chips evidence-chips--inline";
  const lab = document.createElement("span");
  lab.className = "evidence-chips-label";
  lab.textContent = "В записи: ";
  wrap.appendChild(lab);
  if (appendEvidenceChipButtons(wrap, segments) === 0) return null;
  return wrap;
}

/**
 * Один фрагмент: `[11.82, 13.58]` или `[11.82–13.58]` / `[11.82-13.58]` (запятая или тире/en-dash/U+2212).
 */
const BRACKET_TIME_RANGE_RE =
  /^\[\s*([\d.]+)\s*(?:,\s*|\s*[\u2013\u2212\-]\s*)([\d.]+)\s*\]/;

/** Сканирует от позиции `[` цепочку диапазонов через ` и ` или `,` */
function extractBracketTupleRun(s, bracketIndex) {
  let i = bracketIndex;
  const segments = [];
  while (i < s.length) {
    const m = s.slice(i).match(BRACKET_TIME_RANGE_RE);
    if (!m) break;
    segments.push({ t0: Number(m[1]), t1: Number(m[2]) });
    i += m[0].length;
    const sep = s.slice(i).match(/^\s*(?:и|,)\s*/);
    if (sep) i += sep[0].length;
    else break;
  }
  if (segments.length === 0) return null;
  return { start: bracketIndex, end: i, segments };
}

function findNextTupleBracketIndex(s, from) {
  const re = /\[\s*[\d.]/g;
  re.lastIndex = from;
  const m = re.exec(s);
  return m ? m.index : -1;
}

/**
 * Заменяет в тексте узла:
 * - `evidence_segments: [{"start":…,"end":…}, …]`
 * - пары `[205.38, 207.74]` или `[11.82–13.58]` и цепочки через ` и ` / `,`
 */
function replaceEvidenceSegmentsInTextNode(wholeText) {
  if (!wholeText) return null;
  const frag = document.createDocumentFragment();
  let pos = 0;
  let replaced = false;

  while (pos < wholeText.length) {
    const idxSeg = wholeText.indexOf("evidence_segments", pos);
    const idxTuple = findNextTupleBracketIndex(wholeText, pos);
    const segAt = idxSeg >= 0 ? idxSeg : Infinity;
    const tupAt = idxTuple >= 0 ? idxTuple : Infinity;

    if (segAt === Infinity && tupAt === Infinity) {
      frag.appendChild(document.createTextNode(wholeText.slice(pos)));
      break;
    }

    if (segAt <= tupAt) {
      frag.appendChild(document.createTextNode(wholeText.slice(pos, idxSeg)));
      const rest = wholeText.slice(idxSeg);
      const cm = rest.match(/^evidence_segments\s*:\s*/);
      if (!cm) {
        frag.appendChild(document.createTextNode(wholeText[idxSeg]));
        pos = idxSeg + 1;
        continue;
      }
      let j = cm[0].length;
      while (j < rest.length && /\s/.test(rest[j])) j++;
      if (rest[j] !== "[") {
        frag.appendChild(document.createTextNode(wholeText[idxSeg]));
        pos = idxSeg + 1;
        continue;
      }
      let depth = 0;
      let k = j;
      for (; k < rest.length; k++) {
        if (rest[k] === "[") depth++;
        else if (rest[k] === "]") {
          depth--;
          if (depth === 0) break;
        }
      }
      if (k >= rest.length || depth !== 0) {
        frag.appendChild(document.createTextNode(wholeText[idxSeg]));
        pos = idxSeg + 1;
        continue;
      }
      const jsonStr = rest.slice(j, k + 1);
      const matchLen = cm[0].length + (k - j + 1);
      let arr;
      try {
        arr = JSON.parse(jsonStr);
      } catch {
        frag.appendChild(document.createTextNode(wholeText.slice(idxSeg, idxSeg + matchLen)));
        pos = idxSeg + matchLen;
        continue;
      }
      if (!Array.isArray(arr)) {
        frag.appendChild(document.createTextNode(wholeText.slice(idxSeg, idxSeg + matchLen)));
        pos = idxSeg + matchLen;
        continue;
      }
      const segments = [];
      for (const seg of arr) {
        if (!seg || seg.start == null || seg.end == null) continue;
        const t0 = Number(seg.start);
        const t1 = Number(seg.end);
        if (!Number.isFinite(t0) || !Number.isFinite(t1)) continue;
        segments.push({ t0, t1 });
      }
      const wrap = makeEvidenceChipsWrapFromSegments(segments);
      if (wrap) {
        frag.appendChild(wrap);
        replaced = true;
      } else {
        frag.appendChild(document.createTextNode(wholeText.slice(idxSeg, idxSeg + matchLen)));
      }
      pos = idxSeg + matchLen;
    } else {
      frag.appendChild(document.createTextNode(wholeText.slice(pos, idxTuple)));
      const run = extractBracketTupleRun(wholeText, idxTuple);
      if (!run) {
        frag.appendChild(document.createTextNode(wholeText[idxTuple]));
        pos = idxTuple + 1;
        continue;
      }
      const wrap = makeEvidenceChipsWrapFromSegments(run.segments);
      if (wrap) {
        frag.appendChild(wrap);
        replaced = true;
      } else {
        frag.appendChild(document.createTextNode(wholeText.slice(run.start, run.end)));
      }
      pos = run.end;
    }
  }
  return replaced ? frag : null;
}

function upgradeEvalReasoningEvidenceSegments(root) {
  if (!root) return;
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const textNodes = [];
  let n;
  while ((n = walker.nextNode())) {
    const t = n.textContent;
    if (
      !t ||
      (!t.includes("evidence_segments") &&
        !/\[\s*[\d.]+\s*(?:,\s*|[\u2013\u2212\-]\s*)[\d.]+\s*\]/.test(t))
    ) {
      continue;
    }
    const p = n.parentElement;
    if (!p || p.closest("pre, code")) continue;
    textNodes.push(n);
  }
  for (const textNode of textNodes) {
    const frag = replaceEvidenceSegmentsInTextNode(textNode.textContent);
    if (frag && textNode.parentNode) {
      textNode.parentNode.replaceChild(frag, textNode);
    }
  }
}

/** Блок «ход рассуждений»: Markdown → безопасный HTML или plain pre. */
function renderEvalReasoningBody(raw) {
  const text = raw == null ? "" : String(raw);
  const pre = document.createElement("pre");
  pre.className = "eval-reasoning-body eval-reasoning-body--plain";
  pre.textContent = text;

  const g = globalThis;
  const parse = g.marked && typeof g.marked.parse === "function" ? g.marked.parse.bind(g.marked) : null;
  const purify = g.DOMPurify && typeof g.DOMPurify.sanitize === "function" ? g.DOMPurify.sanitize : null;
  if (!parse || !purify || !text.trim()) {
    return pre;
  }
  try {
    let html = parse(text, { async: false, breaks: true, gfm: true });
    if (html && typeof html.then === "function") {
      return pre;
    }
    const clean = purify(html, {
      ALLOWED_TAGS: [
        "p",
        "br",
        "strong",
        "em",
        "b",
        "i",
        "del",
        "s",
        "ul",
        "ol",
        "li",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "blockquote",
        "code",
        "pre",
        "hr",
        "a",
        "table",
        "thead",
        "tbody",
        "tr",
        "th",
        "td",
      ],
      ALLOWED_ATTR: ["href", "title", "colspan", "rowspan"],
      ALLOW_DATA_ATTR: false,
    });
    const div = document.createElement("div");
    div.className = "eval-reasoning-body eval-reasoning-md";
    div.innerHTML = clean;
    upgradeEvalReasoningEvidenceSegments(div);
    return div;
  } catch {
    return pre;
  }
}

/** Текст анализа ИИ: безопасный HTML — абзацы, **жирный**, списки. */
function formatCompareAnalysisHtml(raw) {
  let s = raw == null ? "" : String(raw).trim();
  if (!s) return "";
  let esc = escapeHtml(s);
  esc = esc.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  const blocks = esc.split(/\n{2,}/).map((b) => b.trim()).filter(Boolean);
  return blocks.map((block) => formatCompareAnalysisBlock(block)).join("");
}

function formatCompareAnalysisBlock(block) {
  const lines = block.split("\n");
  const nonEmpty = lines.map((l) => l.trim()).filter((l) => l.length);
  if (nonEmpty.length >= 2) {
    const allBullet = nonEmpty.every((l) => /^[-*•]\s+/.test(l));
    if (allBullet) {
      const items = nonEmpty
        .map((l) => `<li>${l.replace(/^[-*•]\s+/, "").trim()}</li>`)
        .join("");
      return `<ul class="compare-analysis-ul">${items}</ul>`;
    }
    const allNum = nonEmpty.every((l) => /^\d+\.\s+/.test(l));
    if (allNum) {
      const items = nonEmpty
        .map((l) => `<li>${l.replace(/^\d+\.\s+/, "").trim()}</li>`)
        .join("");
      return `<ol class="compare-analysis-ol">${items}</ol>`;
    }
  }
  const inner = lines.join("<br />");
  return `<p class="compare-analysis-p">${inner}</p>`;
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

  const aiView = document.getElementById("eval-ai-view");
  if (aiView) {
    aiView.querySelectorAll(".evidence-chip").forEach((b) => b.classList.remove("evidence-chip--active"));
  }

  if (tbody) {
    const rows = tbody.querySelectorAll("tr[data-criterion-id]");
    rows.forEach((tr) => {
      tr.classList.remove("eval-row--evidence-active");
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

  if (aiView) {
    aiView.querySelectorAll(".eval-reasoning-md .evidence-chip").forEach((btn) => {
      const a = Number(btn.dataset.evStart);
      const b = Number(btn.dataset.evEnd);
      if (Number.isFinite(a) && Number.isFinite(b) && t >= a && t < b) {
        btn.classList.add("evidence-chip--active");
      }
    });
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
  const activeChip =
    (aiView && aiView.querySelector(".evidence-chip--active")) ||
    (tbody && tbody.querySelector(".evidence-chip--active"));
  const activeRow = tbody && tbody.querySelector("tr.eval-row--evidence-active");
  /* Сначала таймкод (чеклист или «ход рассуждений»), иначе строка, иначе транскрипт */
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
  if (item && item.upload_phase) {
    if (item.upload_phase === "uploading") {
      const progress = Math.max(0, Math.min(Number(item.upload_progress) || 0, 100));
      return `Загрузка файла: ${Math.round(progress)}%`;
    }
    if (item.upload_phase === "processing") {
      return "Файл загружен, запускается обработка";
    }
    if (item.upload_phase === "error") {
      return item.upload_error || "Ошибка загрузки";
    }
  }
  if (item && item.status_summary && item.status_summary.tooltip) {
    return item.status_summary.tooltip;
  }
  const j = item.job;
  if (j && (j.status === "queued" || j.status === "running")) {
    const label = jobStageLabel(j);
    return `${label} (нажмите для просмотра)`;
  }
  if (j && j.status === "cancelled") {
    return "Обработка остановлена";
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
  if (item && item.upload_phase === "uploading") return "status-dot--upload";
  if (item && item.upload_phase === "processing") return "status-dot--spinner";
  if (item && item.upload_phase === "error") return "status-dot--red";
  const statusCode = item && item.status_summary && item.status_summary.code;
  if (statusCode) return `status-dot--${statusCode}`;
  const j = item.job;
  if (j && j.status === "queued") return "status-dot--queued";
  if (j && j.status === "running") return "status-dot--spinner";
  if (j && j.status === "cancelled") return "status-dot--off";
  if (j && j.status === "error") return "status-dot--red";
  if (item.has_transcript && item.has_tone && item.has_evaluation) return "status-dot--green";
  if (item.has_transcript) return "status-dot--yellow";
  return "status-dot--off";
}

function workspaceArtifactParts(ws) {
  const parts = [];
  if (ws && ws.transcript) parts.push("текст");
  if (ws && ws.tone) parts.push("тон");
  if (ws && ws.evaluation) parts.push("оценка");
  return parts;
}

function buildWorkspaceMetaItems(ws) {
  if (!ws || !ws.video_url) {
    return [{ text: "Файл отсутствует", tone: "warn" }];
  }
  const items = [];
  const permissions = workspacePermissions(ws);
  if (ws.training_type && ws.training_type.name) items.push({ text: ws.training_type.name, tone: "primary" });
  if (permissions.read_only) items.push({ text: "Только чтение", tone: "warn" });
  if (ws.meta && ws.meta.uploaded_at) items.push({ text: formatShortDate(ws.meta.uploaded_at), tone: "neutral" });
  return items;
}

/**
 * Строка под заголовком: только статичные метаданные записи, без live-статуса пайплайна.
 */
function formatWorkspaceMetaLine(ws) {
  return buildWorkspaceMetaItems(ws)
    .map((item) => item.text)
    .filter(Boolean)
    .join(" · ");
}

function renderWorkspaceMetaItems(target, items, title) {
  if (!target) return;
  const list = Array.isArray(items) ? items.filter((item) => item && item.text) : [];
  target.replaceChildren();
  target.title = title || list.map((item) => item.text).join(" · ");
  for (const item of list) {
    const chip = document.createElement("span");
    chip.className = `workspace-meta-chip workspace-meta-chip--${item.tone || "neutral"}`;
    chip.textContent = item.text;
    target.appendChild(chip);
  }
}

function renderWorkspaceMetaLine(target, ws) {
  renderWorkspaceMetaItems(target, buildWorkspaceMetaItems(ws), formatWorkspaceMetaLine(ws));
}

function syncWorkspaceJobActionsVisibility() {
  const actions = document.getElementById("workspace-job-actions");
  const details = document.getElementById("workspace-job-details");
  const cancel = document.getElementById("workspace-job-cancel");
  const afterStop = document.getElementById("workspace-job-after-stop");
  const restart = document.getElementById("workspace-job-restart-full");
  const compareRestart = document.getElementById("workspace-compare-restart");
  if (!actions) return;
  const hasVisibleControl =
    Boolean(details && !details.hidden) ||
    Boolean(cancel && !cancel.hidden) ||
    Boolean(afterStop && !afterStop.hidden) ||
    Boolean(restart && !restart.hidden) ||
    Boolean(compareRestart && !compareRestart.hidden);
  actions.hidden = !hasVisibleControl;
}

function aiPipelineStepIndexFromStage(stage) {
  switch (stage) {
    case "starting":
    case "extract_audio":
    case "resume":
      return 0;
    case "diarization":
    case "diarization_skip":
    case "diarization_skip_windows":
      return 1;
    case "whisper_load":
    case "asr_whisper":
    case "transcribing":
      return 2;
    case "segments_build":
      return 3;
    case "tone":
      return 4;
    case "eval_prep":
    case "eval_waiting":
    case "evaluating":
      return 5;
    default:
      return -1;
  }
}

function markAiPipelineStepsDone(steps, lastIndex) {
  if (lastIndex < 0) return;
  const limit = Math.min(lastIndex, steps.length - 1);
  for (let i = 0; i <= limit; i += 1) {
    steps[i].state = "success";
  }
}

function firstIncompleteAiPipelineStepIndex(steps) {
  return steps.findIndex((step) => step.state !== "success");
}

function deriveAiPipelineSteps(ws) {
  const steps = AI_PIPELINE_STEPS.map((step) => ({ ...step, state: "waiting" }));
  if (!ws) return steps;

  if (ws.transcript) markAiPipelineStepsDone(steps, 3);
  if (ws.tone) markAiPipelineStepsDone(steps, 4);
  if (ws.ai_evaluation_available || ws.evaluation) markAiPipelineStepsDone(steps, 5);

  const job = ws.job;
  if (job && job.status === "running") {
    const stepIndex = job.kind === "eval_only" ? 5 : aiPipelineStepIndexFromStage(job.stage);
    if (stepIndex >= 0) {
      for (let i = 0; i < stepIndex; i += 1) {
        steps[i].state = "success";
      }
      if (steps[stepIndex].state !== "success") {
        steps[stepIndex].state = "pending";
      }
    }
  }

  if (job && (job.status === "error" || job.status === "cancelled")) {
    const mappedIndex = job.kind === "eval_only" ? 5 : aiPipelineStepIndexFromStage(job.stage);
    const stepIndex =
      mappedIndex >= 0 ? mappedIndex : Math.max(0, firstIncompleteAiPipelineStepIndex(steps));
    for (let i = 0; i < stepIndex; i += 1) {
      steps[i].state = "success";
    }
    steps[Math.min(stepIndex, steps.length - 1)].state = "error";
  }

  if (ws.transcript_load_error) steps[3].state = "error";
  if (ws.tone_load_error) steps[4].state = "error";
  if (ws.criteria_resolution_error && !(ws.ai_evaluation_available || ws.evaluation)) {
    steps[5].state = "error";
  }

  return steps;
}

function deriveAiPipelineSummary(ws, steps) {
  const job = ws && ws.job;
  if (job && job.status === "running") {
    return { state: "pending", text: jobStageLabel(job) };
  }
  if (job && job.status === "queued") {
    return { state: "waiting", text: jobStageLabel(job) };
  }
  if (job && job.status === "cancelled") {
    return { state: "error", text: "Пайплайн остановлен" };
  }
  if (job && job.status === "error") {
    return { state: "error", text: "Пайплайн завершился с ошибкой" };
  }
  if (steps.every((step) => step.state === "success")) {
    return { state: "success", text: "Пайплайн завершён" };
  }
  if (steps.some((step) => step.state === "error")) {
    return { state: "error", text: "В пайплайне есть ошибка" };
  }
  if (steps.some((step) => step.state === "success")) {
    return { state: "waiting", text: "Есть промежуточные результаты" };
  }
  return { state: "waiting", text: "Ожидает запуска пайплайна" };
}

function deriveAiPipelineNote(ws) {
  const job = ws && ws.job;
  if (job && job.status === "cancelled") {
    return "Обработка остановлена вручную. Можно продолжить с текущего места или запустить заново.";
  }
  if (job && job.status === "error") {
    const err = job.error && String(job.error).trim();
    return err ? `Ошибка пайплайна: ${err}` : "Пайплайн завершился с ошибкой.";
  }
  if (ws && ws.transcript_load_error) {
    return "Не удалось прочитать сохранённый транскрипт.";
  }
  if (ws && ws.tone_load_error) {
    return "Не удалось прочитать сохранённый файл эмоций.";
  }
  if (ws && ws.criteria_resolution_error) {
    return String(ws.criteria_resolution_error);
  }
  return "";
}

function renderAiPipelinePanel(ws = _lastWorkspaceData) {
  const summaryEl = document.getElementById("ai-pipeline-summary");
  const stepsEl = document.getElementById("ai-pipeline-steps");
  const noteEl = document.getElementById("ai-pipeline-note");
  if (!summaryEl || !stepsEl || !noteEl) return;

  const steps = deriveAiPipelineSteps(ws);
  const note = deriveAiPipelineNote(ws);

  summaryEl.replaceChildren();

  const reasoningHost = document.getElementById("eval-reasoning-host");
  if (reasoningHost) reasoningHost.remove();

  const stepRowHtml = (step, index) => `
          <span class="ai-pipeline-step__icon" aria-hidden="true">${evalStatusIconHtml(step.state)}</span>
          <span class="ai-pipeline-step__body">
            <span class="ai-pipeline-step__index">${index + 1}.</span>
            <span class="ai-pipeline-step__label">${escapeHtml(step.label)}</span>
          </span>`;

  stepsEl.innerHTML = steps
    .map((step, index) => {
      if (step.key === "evaluating") {
        return `
        <li class="ai-pipeline-step ai-pipeline-step--${step.state} ai-pipeline-step--evaluating" data-ai-pipeline-step="${escapeHtml(step.key)}">
          <div class="ai-pipeline-step__line">${stepRowHtml(step, index)}</div>
          <div class="eval-reasoning-slot" aria-hidden="true"></div>
        </li>`;
      }
      return `
        <li class="ai-pipeline-step ai-pipeline-step--${step.state}" data-ai-pipeline-step="${escapeHtml(step.key)}">
          ${stepRowHtml(step, index)}
        </li>`;
    })
    .join("");

  const slot = stepsEl.querySelector(".eval-reasoning-slot");
  if (reasoningHost && slot) {
    slot.replaceWith(reasoningHost);
  } else if (reasoningHost) {
    const evalLi = stepsEl.querySelector('[data-ai-pipeline-step="evaluating"]');
    if (evalLi) evalLi.appendChild(reasoningHost);
  }

  noteEl.hidden = !note;
  noteEl.classList.toggle("ai-pipeline-card__note--error", Boolean(note));
  noteEl.textContent = note;
}

function normalizeHumanEvalCriteria(criteria) {
  if (!Array.isArray(criteria)) return [];
  return criteria.map((item) => ({
    id: item && item.id != null ? String(item.id) : "",
    name: item && item.name != null ? String(item.name) : "",
    passed:
      item && item.passed === true ? true : item && item.passed === false ? false : null,
    weight: Number(item && item.weight != null ? item.weight : 1),
    comment: item && item.comment != null ? String(item.comment).trim() : "",
  }));
}

function humanEvalDraftFingerprint(criteria) {
  return JSON.stringify(normalizeHumanEvalCriteria(criteria));
}

function resetHumanEvalDraftState(criteria = []) {
  _humanEvalDraftBaseline = humanEvalDraftFingerprint(criteria);
  _humanEvalDirty = false;
}

function collectHumanEvalDraftFromForm(form) {
  if (!form) return [];
  const rows = form.querySelectorAll(".human-eval-row");
  const criteria = [];
  for (const row of rows) {
    const commentInput = row.querySelector(".human-eval-comment-input");
    const rawPassed = row.dataset.passed;
    criteria.push({
      id: row.dataset.critId || "",
      name: row.dataset.critName || "",
      passed: rawPassed === "true" ? true : rawPassed === "false" ? false : null,
      weight: Number(row.dataset.critWeight || 1),
      comment: (commentInput?.value || "").trim(),
    });
  }
  return criteria;
}

function syncHumanEvalDirtyFromForm(form = document.getElementById("human-eval-form")) {
  const dirty = humanEvalDraftFingerprint(collectHumanEvalDraftFromForm(form)) !== _humanEvalDraftBaseline;
  if (_humanEvalDirty === dirty) return;
  _humanEvalDirty = dirty;
  updateEvalToggleState(_lastWorkspaceData);
}

function syncEvalPanelView() {
  const aiView = document.getElementById("eval-ai-view");
  const humanView = document.getElementById("eval-human-view");
  const compareView = document.getElementById("eval-compare-view");
  const logView = document.getElementById("workspace-job-stream-wrap");
  const pipelineCard = document.getElementById("ai-pipeline-card");
  const pipelineToggleBtn = document.getElementById("ai-pipeline-toggle");
  const title = document.getElementById("eval-panel-title");
  const toggle = document.getElementById("eval-mode-toggle");
  const refreshBtn = document.getElementById("eval-refresh");
  const saveBtn = document.getElementById("human-eval-save");
  const publishBtn = document.getElementById("human-eval-publish");
  const humanState = (_lastWorkspaceData && _lastWorkspaceData.human_eval_state) || {};
  const permissions = workspacePermissions(_lastWorkspaceData);
  const humanPublished = Boolean(humanState.published_at);
  const hasSavedHumanDraft = Boolean(
    _lastWorkspaceData &&
      _lastWorkspaceData.human_evaluation &&
      Array.isArray(_lastWorkspaceData.human_evaluation.criteria)
  );
  const job = (_lastWorkspaceData && _lastWorkspaceData.job) || null;
  const aiStatus = deriveAiEvalStatus(_lastWorkspaceData);
  const showLog = Boolean(
    _showWorkspaceJobLog &&
    permissions.can_view_ai_log &&
    workspaceJobShowsReasoningStream(job),
  );
  const showRetry =
    !showLog &&
    _currentEvalMode === "ai" &&
    permissions.can_re_evaluate &&
    aiStatus === "error";
  const canCollapsePipelineCard = isAiPipelineSuccessfullyComplete(_lastWorkspaceData);
  if (!canCollapsePipelineCard) {
    _showCompletedAiPipelineCard = false;
  }
  const showPipelineCard =
    !showLog &&
    _currentEvalMode === "ai" &&
    (!canCollapsePipelineCard || _showCompletedAiPipelineCard);

  if (title) {
    title.textContent = showLog ? "Лог выполнения" : "Оценка по чеклисту";
  }
  if (toggle) {
    toggle.style.display = showLog ? "none" : "";
  }
  if (aiView) aiView.style.display = !showLog && _currentEvalMode === "ai" ? "" : "none";
  if (humanView) humanView.style.display = !showLog && _currentEvalMode === "human" ? "" : "none";
  if (compareView) compareView.style.display = !showLog && _currentEvalMode === "compare" ? "" : "none";
  if (pipelineCard) {
    pipelineCard.hidden = !showPipelineCard;
  }
  if (pipelineToggleBtn) {
    const showPipelineToggle = !showLog && _currentEvalMode === "ai" && canCollapsePipelineCard;
    pipelineToggleBtn.style.display = showPipelineToggle ? "" : "none";
    pipelineToggleBtn.classList.toggle("eval-action-btn--active", showPipelineToggle && _showCompletedAiPipelineCard);
    const toggleTitle = _showCompletedAiPipelineCard ? "Скрыть этапы пайплайна" : "Показать этапы пайплайна";
    pipelineToggleBtn.title = toggleTitle;
    pipelineToggleBtn.setAttribute("aria-label", toggleTitle);
    pipelineToggleBtn.setAttribute(
      "aria-pressed",
      showPipelineToggle && _showCompletedAiPipelineCard ? "true" : "false",
    );
  }
  if (logView) {
    logView.hidden = !showLog;
  }
  if (saveBtn) {
    const canShowSave =
      !humanPublished &&
      permissions.can_save_human_eval &&
      (_humanEvalDirty || !hasSavedHumanDraft);
    saveBtn.style.display =
      !showLog && _currentEvalMode === "human" && canShowSave
        ? ""
        : "none";
  }
  if (publishBtn) {
    const canShowPublish =
      _evalTransientState.publishPending ||
      (!humanPublished &&
        !_humanEvalDirty &&
        hasSavedHumanDraft &&
        permissions.can_publish_human_eval);
    publishBtn.style.display =
      !showLog &&
      _currentEvalMode === "human" &&
      canShowPublish
        ? ""
        : "none";
  }
  if (refreshBtn) refreshBtn.style.display = showRetry ? "" : "none";

  document.querySelectorAll("#eval-mode-toggle .eval-mode-btn").forEach((b) => {
    b.classList.toggle("eval-mode-btn--active", b.dataset.mode === _currentEvalMode);
  });
  syncEvalModeButtonStates(_lastWorkspaceData);
}

function workspaceToStatusItem(ws) {
  let statusSummary = null;
  const diff =
    ws &&
    ws.comparison_state &&
    ws.comparison_state.payload &&
    ws.comparison_state.payload.overall_diff != null
      ? Number(ws.comparison_state.payload.overall_diff)
      : null;
  if (ws && ws.job && ws.job.status === "running") {
    statusSummary = { code: "spinner", tooltip: "Идёт обработка записи, оценка или сравнение." };
  } else if (ws && ws.job && ws.job.status === "queued") {
    statusSummary = { code: "queued", tooltip: "Запись в очереди на обработку." };
  } else if (!(ws && ws.human_eval_state && ws.human_eval_state.published_at)) {
    statusSummary = { code: "off", tooltip: "Ручная оценка ещё не опубликована." };
  } else if (diff != null) {
    const code = diffClass(diff) === "danger" ? "red" : diffClass(diff) === "warn" ? "yellow" : "green";
    statusSummary = { code, tooltip: `Расхождение ИИ и человека: ${formatComparisonDiff(ws.comparison_state.payload)}` };
  }
  return {
    job: ws && ws.job,
    has_transcript: Boolean(ws && ws.transcript),
    has_tone: Boolean(ws && ws.tone),
    has_evaluation: Boolean(ws && ws.evaluation),
    status_summary: statusSummary,
  };
}

function updateWorkspaceHeadStatus(ws) {
  const wrap = document.getElementById("workspace-head-status-wrap");
  const dot = document.getElementById("workspace-head-status-dot");
  if (!wrap || !dot) return;
  const item = workspaceToStatusItem(ws);
  dot.className = "status-dot " + statusDotClass(item);
  wrap.title = jobTooltip({
    ...item,
    has_video_file: Boolean(ws && ws.video_url),
  });
}

async function fetchLibrary() {
  const qs = new URLSearchParams();
  if (_showDeletedLibrary) qs.set("include_deleted", "1");
  if (_showForeignLibrary) qs.set("include_foreign", "1");
  const q = qs.toString() ? `?${qs.toString()}` : "";
  const r = await apiFetch(`${API}/api/library${q}`);
  if (!r.ok) return [];
  return r.json();
}

function getLibrarySortValue() {
  const sel = document.getElementById("library-sort-select");
  const v = sel?.value;
  return v && String(v).trim() !== "" ? v : "date_desc";
}

function syncLibraryViewMenuState() {
  const root = document.getElementById("library-sort");
  const select = document.getElementById("library-sort-select");
  const label = document.getElementById("library-sort-trigger-label");
  const deletedToggle = document.getElementById("library-show-deleted");
  const foreignToggle = document.getElementById("library-show-foreign");
  if (label) {
    label.textContent = "Вид списка";
  }
  if (!root) return;
  const hasNonDefaultSort = (select?.value || "date_desc") !== "date_desc";
  const hasDeletedFilter = Boolean(deletedToggle?.checked);
  const hasForeignFilter = Boolean(foreignToggle?.checked && !foreignToggle?.disabled);
  root.classList.toggle(
    "library-sort--nondefault",
    hasNonDefaultSort || hasDeletedFilter || hasForeignFilter
  );
}

function setupLibrarySortCustom() {
  const root = document.getElementById("library-sort");
  const select = document.getElementById("library-sort-select");
  const trigger = document.getElementById("library-sort-trigger");
  const label = document.getElementById("library-sort-trigger-label");
  const menu = document.getElementById("library-sort-listbox");
  if (!root || !select || !trigger || !label || !menu) {
    return { syncFromSelect: () => {}, syncUiState: () => {}, closeMenu: () => {} };
  }

  const getOptions = () => [...menu.querySelectorAll(".library-sort-option[data-value]")];

  function syncFromSelect() {
    label.textContent = "Вид списка";
    const v = select.value;
    getOptions().forEach((el) => {
      const on = el.dataset.value === v;
      el.setAttribute("aria-checked", on ? "true" : "false");
      el.classList.toggle("library-sort-option--selected", on);
    });
    syncLibraryViewMenuState();
  }

  function layoutMenuPosition() {
    const r = trigger.getBoundingClientRect();
    menu.style.position = "fixed";
    menu.style.left = `${Math.max(8, r.left)}px`;
    menu.style.top = `${r.bottom + 4}px`;
    menu.style.width = `${Math.max(r.width, 248)}px`;
    menu.style.right = "auto";
    menu.style.maxHeight = `min(320px, calc(100vh - ${r.bottom + 12}px))`;
    menu.style.overflowY = "auto";
  }

  function clearMenuPosition() {
    menu.style.position = "";
    menu.style.left = "";
    menu.style.top = "";
    menu.style.width = "";
    menu.style.right = "";
    menu.style.maxHeight = "";
    menu.style.overflowY = "";
  }

  function closeMenu() {
    if (menu.hidden) return;
    menu.hidden = true;
    clearMenuPosition();
    root.classList.remove("library-sort--open");
    trigger.setAttribute("aria-expanded", "false");
    getOptions().forEach((el) => {
      el.tabIndex = -1;
    });
  }

  function openMenu() {
    menu.hidden = false;
    root.classList.add("library-sort--open");
    trigger.setAttribute("aria-expanded", "true");
    layoutMenuPosition();
    getOptions().forEach((el) => {
      el.tabIndex = -1;
    });
  }

  window.addEventListener("resize", () => {
    if (!menu.hidden) layoutMenuPosition();
  });
  document.addEventListener(
    "scroll",
    () => {
      if (!menu.hidden) layoutMenuPosition();
    },
    true
  );

  function selectValue(value) {
    if (value == null || value === "") return;
    if (select.value !== value) {
      select.value = value;
      select.dispatchEvent(new Event("change", { bubbles: true }));
    }
    syncFromSelect();
    closeMenu();
    trigger.focus();
  }

  trigger.addEventListener("click", (e) => {
    e.stopPropagation();
    if (menu.hidden) openMenu();
    else closeMenu();
  });

  getOptions().forEach((el) => {
    el.addEventListener("click", (e) => {
      e.stopPropagation();
      selectValue(el.dataset.value);
    });
    el.addEventListener("keydown", (e) => {
      const opts = getOptions();
      const i = opts.indexOf(el);
      if (e.key === "ArrowDown") {
        e.preventDefault();
        if (i < opts.length - 1) {
          el.tabIndex = -1;
          opts[i + 1].tabIndex = 0;
          opts[i + 1].focus();
        }
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        if (i > 0) {
          el.tabIndex = -1;
          opts[i - 1].tabIndex = 0;
          opts[i - 1].focus();
        } else {
          closeMenu();
          trigger.focus();
        }
      } else if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        selectValue(el.dataset.value);
      } else if (e.key === "Escape") {
        e.preventDefault();
        closeMenu();
        trigger.focus();
      } else if (e.key === "Home") {
        e.preventDefault();
        el.tabIndex = -1;
        opts[0].tabIndex = 0;
        opts[0].focus();
      } else if (e.key === "End") {
        e.preventDefault();
        el.tabIndex = -1;
        opts[opts.length - 1].tabIndex = 0;
        opts[opts.length - 1].focus();
      }
    });
  });

  trigger.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      if (menu.hidden) openMenu();
      const opts = getOptions();
      opts.forEach((o) => {
        o.tabIndex = -1;
      });
      if (opts.length) {
        opts[0].tabIndex = 0;
        opts[0].focus();
      }
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      if (menu.hidden) openMenu();
      const opts = getOptions();
      opts.forEach((o) => {
        o.tabIndex = -1;
      });
      if (opts.length) {
        const last = opts[opts.length - 1];
        last.tabIndex = 0;
        last.focus();
      }
    } else if (e.key === "Escape" && !menu.hidden) {
      e.preventDefault();
      closeMenu();
    }
  });

  menu.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      e.preventDefault();
      closeMenu();
      trigger.focus();
    }
  });

  document.addEventListener("click", (e) => {
    if (!root.contains(e.target)) closeMenu();
  });

  syncFromSelect();
  select.addEventListener("change", syncFromSelect);

  return { syncFromSelect, syncUiState: syncLibraryViewMenuState, closeMenu };
}

function filterAndSortLibrary(items) {
  const searchEl = document.getElementById("library-search");
  const q = (searchEl?.value || "").trim().toLowerCase();
  const sort = getLibrarySortValue();

  let filtered = items;
  if (q) {
    filtered = items.filter((item) => {
      const haystack = [
        item.display_title,
        item.video_file,
        item.stem,
        item.training_type_name,
        item.manager_name,
        item.location_name,
        item.department_label,
        item.interaction_date,
        item.added_at,
        ...(item.tags || []),
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      return haystack.includes(q);
    });
  }

  const sorted = [...filtered];
  const addedAtValue = (item) => {
    const ts = Date.parse(item?.added_at || "");
    return Number.isFinite(ts) ? ts : (item?.mtime || 0);
  };
  switch (sort) {
    case "date_asc":
      sorted.sort((a, b) => addedAtValue(a) - addedAtValue(b));
      break;
    case "name_asc":
      sorted.sort((a, b) =>
        (a.display_title || a.video_file || a.stem || "").localeCompare(
          b.display_title || b.video_file || b.stem || "",
          "ru",
        ),
      );
      break;
    case "name_desc":
      sorted.sort((a, b) =>
        (b.display_title || b.video_file || b.stem || "").localeCompare(
          a.display_title || a.video_file || a.stem || "",
          "ru",
        ),
      );
      break;
    default:
      sorted.sort((a, b) => addedAtValue(b) - addedAtValue(a));
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

function basenameWithoutExtension(filename) {
  return String(filename || "").replace(/\.[^.]+$/, "") || String(filename || "");
}

function uploadTrainingTypeName(slug) {
  const item = _metaTrainingTypes.find((x) => String(x.slug || "") === String(slug || ""));
  return item ? item.name || item.slug || "" : "";
}

function renderLibraryWithUploads(items) {
  const uploads = Array.from(_activeUploadItems.values());
  if (!uploads.length) return items;
  const stems = new Set((items || []).map((item) => item && item.stem).filter(Boolean));
  const visibleUploads = uploads.filter((item) => !item.server_stem || !stems.has(item.server_stem));
  return [...visibleUploads, ...(items || [])];
}

function createUploadLibraryItem(file, trainingTypeSlug, departmentCode) {
  const id = `upload-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  const name = basenameWithoutExtension(file && file.name);
  const department = String(departmentCode || currentUserDepartment() || "").trim().toUpperCase();
  return {
    stem: `__upload_${id}`,
    upload_id: id,
    upload_phase: "uploading",
    upload_progress: 0,
    display_title: name,
    video_file: file && file.name,
    added_at: new Date().toISOString(),
    training_type_name: uploadTrainingTypeName(trainingTypeSlug),
    manager_name: _authState.full_name || _authState.display_name || _authState.user || "",
    location_name: (_authState.location && (_authState.location.crm_name || _authState.location.name)) || "",
    department,
    department_label: departmentLabel(department),
    permissions: {},
  };
}

function upsertUploadLibraryItem(item, patch = {}) {
  if (!item || !item.upload_id) return;
  _activeUploadItems.set(item.upload_id, { ...item, ...patch });
  renderLibrary(_allLibraryItems);
}

function removeUploadLibraryItem(item) {
  if (!item || !item.upload_id) return;
  _activeUploadItems.delete(item.upload_id);
  renderLibrary(_allLibraryItems);
}

/** Тип тренировки → локация → менеджер → список записей. */
function buildLibraryGroupMap(visible) {
  const byTraining = new Map();
  for (const item of visible) {
    const tk = libraryGroupKey(item.training_type_name);
    const lk = libraryGroupKey(item.location_name);
    const mk = libraryGroupKey(item.manager_name);
    if (!byTraining.has(tk)) byTraining.set(tk, new Map());
    const byLoc = byTraining.get(tk);
    if (!byLoc.has(lk)) byLoc.set(lk, new Map());
    const byMan = byLoc.get(lk);
    if (!byMan.has(mk)) byMan.set(mk, []);
    byMan.get(mk).push(item);
  }
  return byTraining;
}

function createLibraryRow(item) {
  const wrap = document.createElement("div");
  const isUploadItem = Boolean(item.upload_phase);
  const isMarkedDeleted = Boolean(item.delete_requested_at || item.deleted_at);
  const permissions = itemPermissions(item);
  const readOnly = Boolean(permissions.read_only);
  wrap.className =
    "library-row" +
    (item.stem === selectedStem ? " selected" : "") +
    (isMarkedDeleted ? " library-row--deleted" : "") +
    (readOnly ? " library-row--readonly" : "") +
    (isUploadItem ? " library-row--uploading" : "");
  wrap.dataset.stem = item.stem;

  const row = document.createElement("button");
  row.type = "button";
  row.className = "library-item";

  const dotWrap = document.createElement("span");
  dotWrap.className = "status-dot-wrap";
  const dot = document.createElement("span");
  dot.className = "status-dot " + statusDotClass(item);
  if (item.upload_phase === "uploading") {
    const progress = Math.max(0, Math.min(Number(item.upload_progress) || 0, 100));
    dot.style.setProperty("--upload-progress", `${progress}%`);
  }
  dotWrap.appendChild(dot);
  dotWrap.title = jobTooltip(item);

  const textWrap = document.createElement("div");
  textWrap.className = "library-item-text";

  const name = document.createElement("div");
  name.className = "library-item-name";
  name.textContent = item.display_title || item.video_file || item.stem;

  const metaLine = [
    item.manager_name,
    item.location_name,
    item.department_label || departmentLabel(item.department || ""),
    readOnly ? "только чтение" : "",
  ]
    .filter(Boolean)
    .join(" · ");

  textWrap.appendChild(name);
  if (isUploadItem) {
    const sub = document.createElement("div");
    sub.className = "library-item-sub";
    const progress = Math.max(0, Math.min(Number(item.upload_progress) || 0, 100));
    sub.textContent =
      item.upload_phase === "uploading"
        ? `Загрузка ${Math.round(progress)}%`
        : item.upload_phase === "processing"
          ? "Файл загружен · обработка запускается"
          : item.upload_error || "Ошибка загрузки";
    textWrap.appendChild(sub);
  }
  row.title = [
    item.display_title || item.video_file || item.stem,
    item.training_type_name || "",
    item.added_at ? `добавлено ${formatShortDate(item.added_at)}` : "",
    metaLine,
    isMarkedDeleted ? "помечено на удаление" : "",
  ]
    .filter(Boolean)
    .join(" · ");

  row.appendChild(dotWrap);
  row.appendChild(textWrap);
  row.addEventListener("click", () => {
    if (isUploadItem) return;
    selectedStem = item.stem;
    document.querySelectorAll(".library-row").forEach((el) => {
      el.classList.toggle("selected", el.dataset.stem === selectedStem);
    });
    collapseSidebarOnMobileIfNeeded();
    loadWorkspace(item.stem);
  });

  const delBtn = document.createElement("button");
  delBtn.type = "button";
  delBtn.className = "library-item-delete";
  const isHardDelete = isAdmin();
  const canRestore = Boolean(!isHardDelete && permissions.can_restore && isMarkedDeleted);
  const canDelete = Boolean(permissions.can_delete && (isHardDelete || !isMarkedDeleted));
  delBtn.setAttribute(
    "aria-label",
    canRestore ? "Вернуть в библиотеку" : isHardDelete ? "Удалить навсегда" : "Пометить на удаление",
  );
  delBtn.title = canRestore
    ? "Вернуть в библиотеку"
    : isHardDelete
      ? "Удалить навсегда"
      : "Пометить на удаление";
  delBtn.innerHTML = canRestore
    ? '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 3v6h6"/></svg>'
    : '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>';
  delBtn.addEventListener("click", async (e) => {
    e.stopPropagation();
    e.preventDefault();
    if (!canRestore && !canDelete) return;
    const label = item.display_title || item.video_file || item.stem;
    const prompt = canRestore
      ? `Вернуть «${label}» в основную библиотеку?`
      : isHardDelete
        ? `Удалить «${label}» окончательно? Будут удалены файл, транскрипт, тон, ручные и ИИ-оценки, сравнение и история задач.`
        : `Пометить «${label}» на удаление? Запись исчезнет из основной библиотеки и останется доступной для восстановления.`;
    if (!(await showAppConfirm(prompt, { title: canRestore ? "Восстановление записи" : "Удаление записи" }))) return;
    const url = canRestore
      ? `${API}/api/library/${encodeURIComponent(item.stem)}/restore`
      : `${API}/api/library/${encodeURIComponent(item.stem)}`;
    const r = await apiFetch(url, { method: canRestore ? "POST" : "DELETE" });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      await showAppAlert(
        data.detail || data.message || (canRestore ? "Не удалось восстановить" : "Не удалось удалить"),
      );
      return;
    }
    if (!canRestore && selectedStem === item.stem) {
      selectedStem = null;
      clearWorkspaceView();
    }
    await loadLibrary();
    if (canRestore && selectedStem === item.stem) {
      await loadWorkspace(item.stem, true);
    }
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
  settingsBtn.hidden = isUploadItem || !isAdmin() || !permissions.can_edit_meta;

  wrap.appendChild(row);
  if (!settingsBtn.hidden) wrap.appendChild(settingsBtn);
  delBtn.hidden = isUploadItem || (!canRestore && !canDelete);
  if (!delBtn.hidden) wrap.appendChild(delBtn);
  return wrap;
}

function renderLibrary(items) {
  const list = document.getElementById("library-list");
  const empty = document.getElementById("library-empty");
  const prevScroll = list ? list.scrollTop : 0;
  list.innerHTML = "";
  const mergedItems = renderLibraryWithUploads(items || []);
  const visible = filterAndSortLibrary(mergedItems);
  if (!visible.length) {
    empty.style.display = "block";
    empty.textContent = mergedItems.length
      ? "Нет записей по фильтру"
      : "Пока нет записей — загрузите видео или аудио выше.";
    return;
  }
  empty.style.display = "none";

  const byTraining = buildLibraryGroupMap(visible);
  for (const trainingKey of sortLibraryGroupKeys([...byTraining.keys()])) {
    const byLoc = byTraining.get(trainingKey);
    const trainingDetails = document.createElement("details");
    trainingDetails.className = "library-folder";
    trainingDetails.open = true;
    const trainingSum = document.createElement("summary");
    trainingSum.className = "library-folder-summary";
    trainingSum.textContent = libraryGroupLabel(trainingKey);
    trainingDetails.appendChild(trainingSum);

    for (const locKey of sortLibraryGroupKeys([...byLoc.keys()])) {
      const byMan = byLoc.get(locKey);
      const locDetails = document.createElement("details");
      locDetails.className = "library-subfolder";
      locDetails.open = true;
      const locSum = document.createElement("summary");
      locSum.className = "library-subfolder-summary";
      locSum.textContent = libraryGroupLabel(locKey);
      locDetails.appendChild(locSum);

      for (const manKey of sortLibraryGroupKeys([...byMan.keys()])) {
        const groupItems = byMan.get(manKey);
        const manDetails = document.createElement("details");
        manDetails.className = "library-subfolder library-subfolder--third";
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
      trainingDetails.appendChild(locDetails);
    }
    list.appendChild(trainingDetails);
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

function syncLibraryForeignToggle() {
  const foreignToggle = document.getElementById("library-show-foreign");
  if (!foreignToggle) return;
  const canUse = _authState.auth_enabled && !isAdmin() && !!currentUserDepartment();
  foreignToggle.checked = canUse ? _showForeignLibrary : false;
  foreignToggle.disabled = !canUse;
  if (!canUse) _showForeignLibrary = false;
  syncLibraryViewMenuState();
}

function setupLibraryControls() {
  const searchEl = document.getElementById("library-search");
  const searchWrap = document.getElementById("library-search-wrap");
  const searchToggle = document.getElementById("library-search-toggle");
  const deletedToggle = document.getElementById("library-show-deleted");
  const foreignToggle = document.getElementById("library-show-foreign");
  const sortRoot = document.getElementById("library-sort");
  const sortTrigger = document.getElementById("library-sort-trigger");
  const sortSelect = document.getElementById("library-sort-select");
  const sortUi = setupLibrarySortCustom();
  const filterIcon = document.getElementById("library-search-filter-icon");
  const labelRow = document.getElementById("library-toolbar-label-row");

  function isSearchPanelOpen() {
    return Boolean(searchWrap?.classList.contains("library-toolbar-slide--open"));
  }

  function setSearchPanelOpen(open) {
    if (!searchWrap || !searchEl) return;
    searchWrap.classList.toggle("library-toolbar-slide--open", open);
    searchWrap.setAttribute("aria-hidden", open ? "false" : "true");
    searchEl.tabIndex = open ? 0 : -1;
    labelRow?.classList.toggle("sidebar-label-row--search-open", open);
    if (open) sortUi.closeMenu();
    if (sortRoot) {
      sortRoot.setAttribute("aria-hidden", open ? "true" : "false");
    }
    if (sortTrigger) {
      sortTrigger.tabIndex = open ? -1 : 0;
    }
    if (sortSelect) {
      sortSelect.tabIndex = -1;
    }
    if (filterIcon) {
      filterIcon.setAttribute("aria-hidden", open ? "false" : "true");
    }
  }

  function syncLibrarySearchToggle() {
    if (!searchToggle) return;
    const q = (searchEl?.value || "").trim();
    const open = isSearchPanelOpen();
    searchToggle.setAttribute("aria-expanded", open ? "true" : "false");
    searchToggle.classList.toggle("library-search-toggle--filter-active", q.length > 0);
  }

  function syncLibrarySortSelect() {
    sortUi.syncFromSelect();
    sortUi.syncUiState();
  }

  if (searchEl) {
    searchEl.addEventListener("input", () => {
      renderLibrary(_allLibraryItems);
      syncLibrarySearchToggle();
    });
  }

  if (searchToggle && searchWrap && searchEl) {
    searchToggle.addEventListener("click", () => {
      setSearchPanelOpen(!isSearchPanelOpen());
      syncLibrarySearchToggle();
      if (isSearchPanelOpen()) {
        searchEl.focus();
      }
    });
    searchEl.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        setSearchPanelOpen(false);
        syncLibrarySearchToggle();
        searchToggle.focus();
      }
    });
  }

  if (sortSelect) {
    sortSelect.addEventListener("change", () => {
      renderLibrary(_allLibraryItems);
      syncLibrarySortSelect();
    });
  }

  if (deletedToggle) {
    deletedToggle.checked = _showDeletedLibrary;
    deletedToggle.addEventListener("change", async () => {
      _showDeletedLibrary = deletedToggle.checked;
      sortUi.syncUiState();
      await loadLibrary();
    });
  }

  if (foreignToggle) {
    syncLibraryForeignToggle();
    foreignToggle.addEventListener("change", async () => {
      _showForeignLibrary = foreignToggle.checked;
      sortUi.syncUiState();
      await loadLibrary();
    });
  }

  if (searchEl && searchWrap && (searchEl.value || "").trim()) {
    setSearchPanelOpen(true);
    syncLibrarySearchToggle();
  } else {
    setSearchPanelOpen(false);
    syncLibrarySearchToggle();
  }

  syncLibrarySortSelect();
  syncLibraryForeignToggle();
  sortUi.syncUiState();
}

function workspaceJobShowsReasoningStream(j) {
  if (!j || j.status === "error" || j.status === "cancelled") return false;
  const run = j.status === "queued" || j.status === "running";
  if (!run) return false;
  if (j.kind === "eval_only") return true;
  if (j.kind === "pipeline" && j.stage === "evaluating") return true;
  return false;
}

function updateWorkspaceJobStreamPanel(j) {
  const btn = document.getElementById("workspace-job-details");
  const wrap = document.getElementById("workspace-job-stream-wrap");
  const pre = document.getElementById("workspace-job-stream-text");
  if (!btn || !wrap || !pre) return;
  const canViewLog = Boolean(
    (_lastWorkspaceData && workspacePermissions(_lastWorkspaceData).can_view_ai_log) || isAdmin(),
  );
  if (!canViewLog || !j || !workspaceJobShowsReasoningStream(j)) {
    _showWorkspaceJobLog = false;
    btn.hidden = true;
    wrap.hidden = true;
    btn.setAttribute("aria-expanded", "false");
    btn.classList.remove("btn-job-details--open");
    pre.textContent = "";
    syncWorkspaceJobActionsVisibility();
    syncEvalPanelView();
    return;
  }
  btn.hidden = false;
  const log = j.stream_log != null ? String(j.stream_log) : "";
  pre.textContent = log || "Лог появится, как только модель начнёт писать ход выполнения…";
  wrap.hidden = !_showWorkspaceJobLog;
  btn.setAttribute("aria-expanded", _showWorkspaceJobLog ? "true" : "false");
  btn.classList.toggle("btn-job-details--open", _showWorkspaceJobLog);
  if (!wrap.hidden) {
    pre.scrollTop = pre.scrollHeight;
  }
  syncWorkspaceJobActionsVisibility();
  syncEvalPanelView();
}

function setupWorkspaceJobStreamToggle() {
  const btn = document.getElementById("workspace-job-details");
  const wrap = document.getElementById("workspace-job-stream-wrap");
  const pre = document.getElementById("workspace-job-stream-text");
  const closeBtn = document.getElementById("workspace-job-log-close");
  if (!btn || !wrap || !pre || !closeBtn) return;
  const syncState = () => {
    wrap.hidden = !_showWorkspaceJobLog;
    btn.setAttribute("aria-expanded", _showWorkspaceJobLog ? "true" : "false");
    btn.classList.toggle("btn-job-details--open", _showWorkspaceJobLog);
    syncEvalPanelView();
    if (_showWorkspaceJobLog && pre.textContent) {
      pre.scrollTop = pre.scrollHeight;
    }
  };
  btn.addEventListener("click", () => {
    if (btn.hidden) return;
    _showWorkspaceJobLog = !_showWorkspaceJobLog;
    syncState();
  });
  closeBtn.addEventListener("click", () => {
    _showWorkspaceJobLog = false;
    syncState();
  });
}

function setupTranscriptDownload() {
  const btn = document.getElementById("transcript-download");
  if (!btn) return;
  btn.addEventListener("click", () => {
    const stem = btn.dataset.stem || (_lastWorkspaceData && _lastWorkspaceData.stem) || selectedStem;
    if (!stem || btn.hidden) return;
    window.location.href = `${API}/api/transcripts/${encodeURIComponent(stem)}/download?format=txt`;
  });
}

async function loadWorkspace(stem, silent, criteriaOverride) {
  stopCompareStatusPoll();
  if (lastWorkspaceStem && lastWorkspaceStem !== stem) {
    _showWorkspaceJobLog = false;
    _showCompletedAiPipelineCard = false;
  }
  if (_compareEvalCache && _compareEvalCache.stem !== stem) {
    _compareEvalCache = null;
  }
  const crit = resolveWorkspaceCriteriaQuery(stem, criteriaOverride);
  const q = crit ? `?criteria=${encodeURIComponent(crit)}` : "";
  const r = await apiFetch(`${API}/api/workspace/${encodeURIComponent(stem)}${q}`);
  if (!r.ok) return;
  const ws = await r.json();
  _lastWorkspaceData = ws;
  _authState = ws.auth || _authState;
  _evalTransientState.aiPending = false;
  _evalTransientState.publishPending = false;
  if (ws && ws.comparison_state && ws.comparison_state.payload) {
    _evalTransientState.compareError = "";
  }
  lastWorkspaceStem = stem;
  lastWorkspaceCriteriaRequested = ws.evaluation_criteria || ws.criteria?.active || null;

  document.getElementById("workspace-empty").style.display = "none";
  document.getElementById("workspace").style.display = "flex";

  const titleEl = document.getElementById("workspace-title");
  const metaEl = document.getElementById("workspace-title-meta");
  const disp = ws.meta && ws.meta.display_title;
  titleEl.textContent = (disp && String(disp).trim()) || ws.video_file || ws.stem;
  if (metaEl) {
    renderWorkspaceMetaLine(metaEl, ws);
  }
  updateWorkspaceHeadStatus(ws);

  const jobCancel = document.getElementById("workspace-job-cancel");
  const j = ws.job;
  const permissions = workspacePermissions(ws);
  stopWorkspaceJobPoll();

  function setJobCancelVisible(show, jobId) {
    if (!jobCancel) return;
    if (show && jobId && permissions.can_control_jobs) {
      jobCancel.hidden = false;
      jobCancel.dataset.jobId = jobId;
    } else {
      jobCancel.hidden = true;
      delete jobCancel.dataset.jobId;
    }
    syncWorkspaceJobActionsVisibility();
  }

  if (j && (j.status === "queued" || j.status === "running")) {
    setJobCancelVisible(true, j.id);
    updateWorkspaceJobStreamPanel(j);
    workspaceJobPollTimer = setInterval(async () => {
      const cPoll = resolveWorkspaceCriteriaQuery(stem, undefined);
      const qPoll = cPoll ? `?criteria=${encodeURIComponent(cPoll)}` : "";
      const r2 = await apiFetch(`${API}/api/workspace/${encodeURIComponent(stem)}${qPoll}`);
      if (!r2.ok) return;
      const w2 = await r2.json();
      _lastWorkspaceData = w2;
      const j2 = w2.job;
      if (!j2 || (j2.status !== "queued" && j2.status !== "running")) {
        stopWorkspaceJobPoll();
        await loadWorkspace(stem, true, lastWorkspaceCriteriaRequested || undefined);
        return;
      }
      const metaPoll = document.getElementById("workspace-title-meta");
      if (metaPoll) renderWorkspaceMetaLine(metaPoll, w2);
      updateWorkspaceHeadStatus(w2);
      renderAiPipelinePanel(w2);
      syncEvalModeButtonStates(w2);
      setJobCancelVisible(Boolean(j2 && j2.id), j2 && j2.id);
      updateWorkspaceJobStreamPanel(j2);
    }, 500);
  } else if (j && j.status === "error") {
    setJobCancelVisible(false);
    updateWorkspaceJobStreamPanel(null);
  } else if (j && j.status === "cancelled") {
    setJobCancelVisible(false);
    updateWorkspaceJobStreamPanel(null);
  } else {
    setJobCancelVisible(false);
    updateWorkspaceJobStreamPanel(null);
  }

  const afterStop = document.getElementById("workspace-job-after-stop");
  const resumeBtn = document.getElementById("workspace-job-resume");
  const restartBtn = document.getElementById("workspace-job-restart-full");
  const compareRestartBtn = document.getElementById("workspace-compare-restart");
  const pipelineComplete = Boolean(ws.transcript && ws.tone && (ws.ai_evaluation_available || ws.evaluation));
  const canControlPipeline = Boolean(ws.video_url) && permissions.can_control_jobs;
  const canRestartComparison = Boolean(
    permissions.can_restart_comparison &&
      ws.ai_evaluation_available &&
      ws.human_evaluation &&
      ws.human_eval_state &&
      ws.human_eval_state.published_at &&
      !(j && (j.status === "queued" || j.status === "running")) &&
      !(ws.comparison_runtime && ws.comparison_runtime.pending),
  );
  const canRestartFailedPipelineAsUser = Boolean(
    ws.video_url &&
      permissions.can_restart_failed_pipeline &&
      j &&
      j.kind === "pipeline" &&
      j.status === "error",
  );
  const canResumePipeline =
    canControlPipeline && j && j.kind === "pipeline" && (j.status === "cancelled" || j.status === "error");
  const canRestartPipeline =
    canRestartFailedPipelineAsUser ||
    (canControlPipeline &&
      (canResumePipeline || pipelineComplete || Boolean(j && j.kind === "pipeline" && j.status === "done")));
  if (afterStop) {
    afterStop.hidden = !canResumePipeline;
  }
  if (resumeBtn) resumeBtn.hidden = !canResumePipeline;
  if (restartBtn) restartBtn.hidden = !canRestartPipeline;
  if (compareRestartBtn) {
    compareRestartBtn.hidden = !canRestartComparison;
    compareRestartBtn.disabled = !canRestartComparison;
  }
  syncWorkspaceJobActionsVisibility();
  renderAiPipelinePanel(ws);

  renderMediaAndTranscript(ws);
  renderToneHint(ws.tone, ws.tone_load_error);
  renderChecklistStaleBanner(ws);
  renderEvaluation(ws.evaluation, {
    hasTranscript: !!ws.transcript,
    criteriaLabel: ws.evaluation_criteria || ws.criteria?.active || "",
    aiAvailable: !!ws.ai_evaluation_available,
    hiddenReason: ws.ai_hidden_reason || "",
  });
  renderHumanEvalForm(ws);
  updateEvalToggleState(ws);
  updateEvalToolbar(ws);
  syncEvalModeButtonStates(ws);
  maybeLoadCompareView();
  if (ws && ws.comparison_runtime && ws.comparison_runtime.pending && selectedStem === stem) {
    scheduleCompareStatusPoll(stem, lastWorkspaceCriteriaRequested || undefined);
  }
  if (!isAdmin() && !ws.evaluation && _currentEvalMode === "ai" && !_showWorkspaceJobLog) {
    switchEvalMode("human");
  }

}

function renderMediaAndTranscript(ws) {
  detachTranscriptMedia();

  const segs = document.getElementById("transcript-segments");
  const videoEl = document.getElementById("transcript-video");
  const audioEl = document.getElementById("transcript-audio");
  const mediaErr = document.getElementById("transcript-media-error");
  const miss = document.getElementById("transcript-missing");
  const downloadBtn = document.getElementById("transcript-download");

  const data = ws.transcript;
  if (downloadBtn) {
    const canDownload = Boolean(isAdmin() && data && ws && ws.stem && !ws.onboarding_preview);
    downloadBtn.hidden = !canDownload;
    downloadBtn.dataset.stem = canDownload ? ws.stem : "";
  }
  mediaErr.style.display = "none";
  mediaErr.textContent = "";

  const vfName = ((data && data.video_file) || ws.video_file || "").toLowerCase();
  const isAudio = /\.(m4a|mp3|aac|wav|flac|ogg|oga|opus|wma|amr)$/.test(vfName);

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

  if (ws && ws.onboarding_preview_media && mediaEl === videoEl) {
    videoEl.poster = ONBOARDING_PREVIEW_POSTER;
    videoEl.classList.add("transcript-video--preview");
    mediaErr.style.display = "none";
    mediaErr.textContent = "";
  } else if (ws.video_url) {
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
    miss.textContent = ws.transcript_load_error
      ? "Файл транскрипта повреждён или не читается. Запустите пайплайн заново или «С начала»."
      : "Транскрипт появится после этапа распознавания речи.";
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
    if (ws && ws.onboarding_preview_media) return;
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
        passed: c.passed,
        weight: c.weight,
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
    item.className = "seg-eval-item " + scoreClass(c.passed);
    item.dataset.criterionId = c.id;

    const head = document.createElement("div");
    head.className = "seg-eval-head";
    const nameEl = document.createElement("span");
    nameEl.className = "seg-eval-name";
    nameEl.textContent = c.name;
    const scoreEl = document.createElement("span");
    scoreEl.className = "seg-eval-score";
    scoreEl.title = formatDecisionHint(c.passed);
    scoreEl.textContent = formatDecisionText(c.passed);
    head.appendChild(nameEl);
    appendCriterionNameWithBadge(nameEl, c.name, c.weight || 1, "crit-name-wrap");
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

function renderToneHint(tone, toneLoadError) {
  const hint = document.getElementById("transcript-ser-hint");
  if (!hint) return;
  if (toneLoadError && !tone) {
    hint.textContent =
      "Файл данных тона повреждён или не читается. Перезапустите обработку или «С начала».";
    hint.style.display = "block";
    syncTranscriptHintsPopover();
    return;
  }
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

function normalizeChecklistName(rawName) {
  const safe = slugifyMetaId(rawName);
  return safe.replace(/^[_-]+|[_-]+$/g, "");
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

function fillMetaForm({
  managers = [],
  locations = [],
  trainingTypes = [],
  meta = {},
  videoFileFallback = "",
  checklistSlug = "",
}) {
  const managerSel = document.getElementById("meta-manager");
  const locationSel = document.getElementById("meta-location");
  const trainingSel = document.getElementById("meta-training-type");
  const checklistSel = document.getElementById("criteria-select");
  const dateIn = document.getElementById("meta-date");
  const tagsIn = document.getElementById("meta-tags");
  const titleIn = document.getElementById("meta-display-title");
  if (!managerSel || !locationSel || !trainingSel || !checklistSel) return;

  _metaTrainingTypes = Array.isArray(trainingTypes) ? trainingTypes.slice() : [];
  const visibleTrainingTypes = _metaTrainingTypes.filter((item) => {
    const hasDepartment = String(item.department || "").trim();
    const usableChecklist = String(item.checklist_slug || "").trim() && String(item.checklist_slug || "").trim() !== "criteria";
    return (hasDepartment && usableChecklist) || item.slug === meta.training_type_slug;
  });

  if (titleIn) {
    const custom = (meta.display_title && String(meta.display_title).trim()) || "";
    titleIn.value = custom || videoFileFallback || "";
  }

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

  trainingSel.innerHTML = '<option value="">—</option>';
  for (const t of visibleTrainingTypes) {
    const o = document.createElement("option");
    o.value = t.slug;
    o.textContent = t.name || t.slug;
    if (t.slug === meta.training_type_slug) o.selected = true;
    trainingSel.appendChild(o);
  }

  const boundType = checklistBoundToTrainingType(meta.training_type_slug);
  const boundChecklist = (boundType && boundType.checklist_slug) || checklistSlug || "";
  checklistSel.innerHTML = "";
  const option = document.createElement("option");
  option.value = boundChecklist;
  option.textContent = boundChecklist || "Не привязан";
  option.selected = true;
  checklistSel.appendChild(option);
  checklistSel.disabled = true;

  if (dateIn) dateIn.value = meta.interaction_date || "";
  if (tagsIn) tagsIn.value = (meta.tags || []).join(", ");
}

function refreshMetaChecklistPreview() {
  const trainingSel = document.getElementById("meta-training-type");
  const checklistSel = document.getElementById("criteria-select");
  if (!trainingSel || !checklistSel) return;
  const boundType = checklistBoundToTrainingType(trainingSel.value);
  const checklistSlug = (boundType && boundType.checklist_slug) || "";
  checklistSel.innerHTML = "";
  const option = document.createElement("option");
  option.value = checklistSlug;
  option.textContent = checklistSlug || "Не привязан";
  option.selected = true;
  checklistSel.appendChild(option);
}

async function openMetaDialog(stem) {
  const dlg = document.getElementById("meta-dialog");
  const stemCode = document.getElementById("meta-dialog-stem-code");
  if (!dlg || !stem) return;
  metaEditStem = stem;
  if (stemCode) stemCode.textContent = stem;
  try {
    const [metaR, mR, lR, ttR] = await Promise.all([
      apiFetch(`${API}/api/workspace/${encodeURIComponent(stem)}/meta`),
      apiFetch(`${API}/api/managers`),
      apiFetch(`${API}/api/locations`),
      apiFetch(`${API}/api/training-types`),
    ]);
    if (!metaR.ok) {
      const d = await metaR.json().catch(() => ({}));
      throw new Error(d.detail || "Не удалось загрузить метаданные");
    }
    const meta = await metaR.json();
    const managers = await mR.json();
    const locations = await lR.json();
    const trainingTypes = ttR.ok ? await ttR.json() : [];
    let videoFileFallback = stem;
    let checklistSlug = "";
    try {
      const wsR = await apiFetch(`${API}/api/workspace/${encodeURIComponent(stem)}`);
      if (wsR.ok) {
        const ws = await wsR.json();
        videoFileFallback = ws.video_file || stem;
        checklistSlug = ws.evaluation_criteria || "";
        _authState = ws.auth || _authState;
        updateEvalToolbar(ws);
      }
    } catch (_) {
      /* чеклист подтянется при следующем loadWorkspace */
    }
    fillMetaForm({ managers, locations, trainingTypes, meta, videoFileFallback, checklistSlug });
    const trainingAddBtn = document.getElementById("meta-training-type-add");
    const trainingEditBtn = document.getElementById("meta-training-type-edit");
    if (trainingAddBtn) trainingAddBtn.hidden = !isAdmin();
    if (trainingEditBtn) trainingEditBtn.hidden = !isAdmin();
    if (typeof dlg.showModal === "function") dlg.showModal();
  } catch (e) {
    metaEditStem = null;
    await showAppAlert(String(e));
  }
}

function setupMetaPanel() {
  const saveBtn = document.getElementById("meta-save");
  const dlg = document.getElementById("meta-dialog");
  const closeBtn = document.getElementById("meta-dialog-close");
  const trainingSel = document.getElementById("meta-training-type");

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
        const r = await apiFetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id, name }),
        });
        if (!r.ok) {
          const d = await r.json().catch(() => ({}));
          await showAppAlert(d.detail || "Не удалось добавить");
          return;
        }
        if (metaEditStem) await openMetaDialog(metaEditStem);
        else if (selectedStem) await loadWorkspace(selectedStem, true);
      } catch (err) {
        await showAppAlert(String(err));
      }
    });
  }
  if (closeBtn && dlg) {
    closeBtn.addEventListener("click", () => dlg.close());
  }

  if (trainingSel) {
    trainingSel.addEventListener("change", refreshMetaChecklistPreview);
  }

  if (saveBtn) {
    saveBtn.addEventListener("click", async () => {
      const stem = metaEditStem;
      if (!stem) return;
      const managerSel = document.getElementById("meta-manager");
      const locationSel = document.getElementById("meta-location");
      const trainingSel = document.getElementById("meta-training-type");
      const dateIn = document.getElementById("meta-date");
      const tagsIn = document.getElementById("meta-tags");
      const managerId = managerSel?.value || null;
      const managerName = managerId ? managerSel.selectedOptions[0]?.textContent || null : null;
      const locationId = locationSel?.value || null;
      const locationName = locationId ? locationSel.selectedOptions[0]?.textContent || null : null;
      const tagsRaw = (tagsIn?.value || "").split(",").map((s) => s.trim()).filter(Boolean);
      const titleIn = document.getElementById("meta-display-title");
      const displayTitleRaw = (titleIn?.value || "").trim();
      try {
        const r = await apiFetch(`${API}/api/workspace/${encodeURIComponent(stem)}/meta`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            display_title: displayTitleRaw || null,
            manager_id: managerId,
            manager_name: managerName,
            location_id: locationId,
            location_name: locationName,
            interaction_date: dateIn?.value || null,
            training_type_slug: trainingSel?.value || null,
            tags: tagsRaw,
          }),
        });
        if (!r.ok) {
          const d = await r.json().catch(() => ({}));
          await showAppAlert(d.detail || "Не удалось сохранить");
          return;
        }
        if (dlg) dlg.close();
        await loadLibrary();
        if (selectedStem === stem) await loadWorkspace(stem, true);
      } catch (e) {
        await showAppAlert(String(e));
      }
    });
  }
}

function fillTrainingTypeChecklistOptions(selected = "", departmentCode = "") {
  const sel = document.getElementById("training-type-checklist");
  if (!sel) return;
  sel.innerHTML = '<option value="">—</option>';
  const files = (_lastWorkspaceData && _lastWorkspaceData.criteria && _lastWorkspaceData.criteria.files) || [];
  const code = String(departmentCode || "").trim().toUpperCase();
  for (const file of files) {
    const name = (file && file.name) || file;
    if (!name) continue;
    const fileDepartment = String((file && file.department) || "").trim().toUpperCase();
    if (code && name !== selected && fileDepartment !== code) continue;
    const o = document.createElement("option");
    o.value = name;
    o.textContent = file && file.display_name ? file.display_name : name;
    if (name === selected) o.selected = true;
    sel.appendChild(o);
  }
}

function setupTrainingTypeDialog() {
  const dlg = document.getElementById("training-type-dialog");
  const addBtn = document.getElementById("meta-training-type-add");
  const editBtn = document.getElementById("meta-training-type-edit");
  const closeBtn = document.getElementById("training-type-cancel");
  const saveBtn = document.getElementById("training-type-save");
  const deleteBtn = document.getElementById("training-type-delete");
  const nameIn = document.getElementById("training-type-name");
  const departmentSel = document.getElementById("training-type-department");
  const checklistSel = document.getElementById("training-type-checklist");
  const titleEl = document.getElementById("training-type-title");
  const slugHint = document.getElementById("training-type-slug-hint");
  const currentTypeSelect = document.getElementById("meta-training-type");
  if (!dlg || !addBtn || !editBtn || !saveBtn || !nameIn || !departmentSel || !checklistSel || !titleEl) return;

  function openCreate() {
    _trainingTypeEditingSlug = null;
    titleEl.textContent = "Новый тип тренировки";
    nameIn.value = "";
    departmentSel.value = "";
    fillTrainingTypeChecklistOptions(document.getElementById("criteria-select")?.value || "", departmentSel.value);
    if (slugHint) slugHint.textContent = "";
    if (deleteBtn) deleteBtn.hidden = true;
    dlg.showModal();
  }

  function openEdit(slug) {
    const item = checklistBoundToTrainingType(slug);
    if (!item) {
      void showAppAlert("Сначала выберите существующий тип тренировки.");
      return;
    }
    _trainingTypeEditingSlug = item.slug;
    titleEl.textContent = "Тип тренировки";
    nameIn.value = item.name || item.slug;
    departmentSel.value = item.department || "";
    fillTrainingTypeChecklistOptions(item.checklist_slug || "", departmentSel.value);
    if (slugHint) slugHint.textContent = `slug: ${item.slug}`;
    if (deleteBtn) deleteBtn.hidden = false;
    dlg.showModal();
  }

  addBtn.addEventListener("click", () => {
    if (!isAdmin()) return;
    openCreate();
  });
  editBtn.addEventListener("click", () => {
    if (!isAdmin()) return;
    openEdit(currentTypeSelect?.value || "");
  });
  departmentSel.addEventListener("change", () => {
    fillTrainingTypeChecklistOptions(checklistSel.value || "", departmentSel.value);
  });
  closeBtn?.addEventListener("click", () => dlg.close());
  dlg.addEventListener("click", (e) => {
    if (e.target === dlg) dlg.close();
  });

  saveBtn.addEventListener("click", async () => {
    const name = (nameIn.value || "").trim();
    if (!name) {
      await showAppAlert("Введите название типа тренировки.");
      nameIn.focus();
      return;
    }
    const slug = _trainingTypeEditingSlug || slugifyMetaId(name);
    const payload = {
      slug,
      name,
      department: departmentSel.value || null,
      checklist_slug: checklistSel.value || null,
    };
    const url = _trainingTypeEditingSlug
      ? `${API}/api/training-types/${encodeURIComponent(_trainingTypeEditingSlug)}`
      : `${API}/api/training-types`;
    const method = _trainingTypeEditingSlug ? "PUT" : "POST";
    try {
      const r = await apiFetch(url, {
        method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        await showAppAlert(data.detail || "Не удалось сохранить тип тренировки");
        return;
      }
      dlg.close();
      if (metaEditStem) await openMetaDialog(metaEditStem);
      else if (selectedStem) await loadWorkspace(selectedStem, true);
    } catch (e) {
      await showAppAlert(String(e));
    }
  });

  deleteBtn?.addEventListener("click", async () => {
    if (!_trainingTypeEditingSlug) return;
    if (
      !(await showAppConfirm(`Удалить тип тренировки «${_trainingTypeEditingSlug}»?`, {
        title: "Удаление типа тренировки",
        okLabel: "Удалить",
      }))
    ) return;
    try {
      const r = await apiFetch(
        `${API}/api/training-types/${encodeURIComponent(_trainingTypeEditingSlug)}`,
        { method: "DELETE" }
      );
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        await showAppAlert(data.detail || "Не удалось удалить тип тренировки");
        return;
      }
      dlg.close();
      if (metaEditStem) await openMetaDialog(metaEditStem);
      else if (selectedStem) await loadWorkspace(selectedStem, true);
    } catch (e) {
      await showAppAlert(String(e));
    }
  });
}

function updateEvalToolbar(ws) {
  const sel = document.getElementById("criteria-select");
  const btn = document.getElementById("eval-refresh");
  if (!sel || !btn) return;

  const crit = ws.criteria || { active: "criteria", files: [] };
  const active = ws.evaluation_criteria || crit.bound || crit.active || "criteria";
  const permissions = workspacePermissions(ws);

  criteriaPopulate = true;
  sel.innerHTML = "";
  const o = document.createElement("option");
  o.value = active;
  o.textContent = active || "—";
  o.selected = true;
  sel.appendChild(o);
  criteriaPopulate = false;

  const j = ws.job;
  const busy = j && (j.status === "queued" || j.status === "running");
  const hasTr = !!ws.transcript;
  btn.disabled = !hasTr || busy || !permissions.can_re_evaluate;
  sel.disabled = true;
  btn.toggleAttribute("aria-busy", Boolean(busy && hasTr));

  btn.title = busy
    ? "Дождитесь окончания задачи"
    : !hasTr
      ? "Сначала нужен транскрипт"
      : !permissions.can_re_evaluate
        ? "Чужую запись можно только просматривать"
      : "Сгенерировать или пересчитать оценку по чеклисту, привязанному к типу тренировки";

  const editBtn = document.getElementById("criteria-edit-btn");
  const newBtn = document.getElementById("criteria-new-btn");
  if (editBtn) {
    editBtn.disabled = busy || !isAdmin();
    editBtn.hidden = !isAdmin();
  }
  if (newBtn) {
    newBtn.disabled = busy || !isAdmin();
    newBtn.hidden = !isAdmin();
  }
}

function setupEvalToolbar() {
  const sel = document.getElementById("criteria-select");
  const btn = document.getElementById("eval-refresh");
  if (!sel || !btn) return;

  btn.addEventListener("click", async () => {
    if (!selectedStem || btn.disabled) return;
    btn.disabled = true;
    sel.disabled = true;
    btn.setAttribute("aria-busy", "true");
    btn.title = "Выполняется оценка…";
    _evalTransientState.aiPending = true;
    syncEvalModeButtonStates(_lastWorkspaceData);
    try {
      const r = await apiFetch(
        `${API}/api/workspace/${encodeURIComponent(selectedStem)}/re-evaluate`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: "{}",
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
        _evalTransientState.aiPending = false;
        syncEvalModeButtonStates(_lastWorkspaceData);
        await showAppAlert(msg);
        if (_lastWorkspaceData) updateEvalToolbar(_lastWorkspaceData);
        else {
          btn.removeAttribute("aria-busy");
          btn.disabled = false;
          sel.disabled = false;
          btn.title =
            "Сгенерировать или пересчитать оценку по выбранному в списке чеклисту";
        }
        return;
      }
      loadLibrary();
      await loadWorkspace(selectedStem, true);
    } catch (e) {
      _evalTransientState.aiPending = false;
      syncEvalModeButtonStates(_lastWorkspaceData);
      await showAppAlert(String(e));
      if (_lastWorkspaceData) updateEvalToolbar(_lastWorkspaceData);
      else {
        btn.removeAttribute("aria-busy");
        btn.disabled = false;
        sel.disabled = false;
        btn.title =
          "Сгенерировать или пересчитать оценку по выбранному в списке чеклисту";
      }
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
  const weightIn = document.createElement("input");
  weightIn.type = "number";
  weightIn.className = "criteria-row-weight";
  weightIn.min = "1";
  weightIn.step = "1";
  weightIn.placeholder = "Вес";
  weightIn.value = data.weight != null ? data.weight : 1;
  const rm = document.createElement("button");
  rm.type = "button";
  rm.className = "criteria-row-remove";
  rm.setAttribute("aria-label", "Удалить критерий");
  rm.textContent = "×";
  rm.addEventListener("click", () => row.remove());
  head.appendChild(idIn);
  head.appendChild(nameIn);
  head.appendChild(weightIn);
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
    const fn = normalizeChecklistName(newName.value || "");
    const copyRaw = (newCopy.value || "").trim();
    const copyFrom = copyRaw ? copyRaw : null;
    if (!fn) {
      await showAppAlert("Укажите имя чеклиста.");
      newName.focus();
      return;
    }
    newName.value = fn;
    try {
      const r = await apiFetch(`${API}/api/criteria`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: fn, copy_from: copyFrom }),
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        await showAppAlert(flattenApiDetail(data.detail) || "Не удалось создать чеклист");
        return;
      }
      const created = data.filename || fn;
      await apiFetch(`${API}/api/criteria/active`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file: created }),
      });
      newDlg.close();
      newName.value = "";
      if (selectedStem) await loadWorkspace(selectedStem, true);
    } catch (err) {
      await showAppAlert(String(err));
    }
  });

  if (editBtn) {
    editBtn.addEventListener("click", async () => {
      const sel = document.getElementById("criteria-select");
      if (!sel || !sel.value || editBtn.disabled) return;
      const filename = sel.value;
      try {
        const r = await apiFetch(`${API}/api/criteria/content/${encodeURIComponent(filename)}`);
        const data = await r.json().catch(() => ({}));
        if (!r.ok) {
          await showAppAlert(flattenApiDetail(data.detail) || "Не удалось загрузить чеклист");
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
        await showAppAlert(String(err));
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
        const weight = Math.max(1, parseInt(row.querySelector(".criteria-row-weight")?.value || "1", 10) || 1);
        const description = row.querySelector(".criteria-row-desc")?.value?.trim() || "";
        if (!id && !name && !description) return;
        i += 1;
        const idFinal = id || `criterion_${i}`;
        criteria.push({
          id: idFinal,
          name: name || idFinal,
          weight,
          description,
        });
      });
      if (!criteria.length) {
        await showAppAlert("Добавьте хотя бы один критерий.");
        return;
      }
      try {
        const r = await apiFetch(
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
          await showAppAlert(flattenApiDetail(data.detail) || "Не удалось сохранить");
          return;
        }
        editDlg.close();
        if (selectedStem) await loadWorkspace(selectedStem, true);
      } catch (err) {
        await showAppAlert(String(err));
      }
    });
  }

  if (editDelete) {
    editDelete.addEventListener("click", async () => {
      if (!criteriaEditingFilename || editDelete.hidden) return;
      if (
        !(await showAppConfirm(`Удалить файл «${criteriaEditingFilename}» с диска?`, {
          title: "Удаление чеклиста",
          okLabel: "Удалить",
        }))
      ) return;
      try {
        const r = await apiFetch(
          `${API}/api/criteria/content/${encodeURIComponent(criteriaEditingFilename)}`,
          { method: "DELETE" }
        );
        const data = await r.json().catch(() => ({}));
        if (!r.ok) {
          await showAppAlert(flattenApiDetail(data.detail) || "Не удалось удалить");
          return;
        }
        editDlg.close();
        if (selectedStem) await loadWorkspace(selectedStem, true);
      } catch (err) {
        await showAppAlert(String(err));
      }
    });
  }
}

function renderEvaluation(evaluation, options = {}) {
  const { hasTranscript = false, criteriaLabel = "", aiAvailable = false, hiddenReason = "" } = options;
  const summary = document.getElementById("eval-summary");
  const reasoningHost = document.getElementById("eval-reasoning-host");
  const miss = document.getElementById("eval-missing");
  const table = document.getElementById("eval-criteria-table");
  const tbody = table.querySelector("tbody");

  if (!evaluation) {
    detachEvalPlaybackSync();
    summary.innerHTML = "";
    if (reasoningHost) reasoningHost.replaceChildren();
    miss.style.display = "block";
    table.style.display = "none";
    if (aiAvailable && hiddenReason) {
      miss.textContent = hiddenReason;
    } else if (hasTranscript) {
      const label = criteriaLabel || "чеклисту";
      miss.textContent = `Для «${label}» оценка ещё не готова. Нажмите кнопку пересчёта (↻), чтобы сгенерировать оценку по этому чеклисту.`;
    } else {
      miss.textContent = "Оценка ИИ появится после завершения пайплайна.";
    }
    return;
  }
  miss.style.display = "none";
  table.style.display = "table";

  const pct = evaluation.overall_average;
  const pctStr = pct != null ? `${Number(pct).toFixed(1)}%` : "—";
  const pctCls = scoreClass(pct);
  const earned = evaluation.earned_score != null ? evaluation.earned_score : "—";
  const maxScore = evaluation.max_score != null ? evaluation.max_score : "—";
  const dateInfo = formatEvaluatedAtParts(evaluation.evaluated_at || "");
  summary.innerHTML = `
    <div class="eval-summary-metrics">
      <div class="compare-overall compare-overall--ai">
        <span class="compare-overall-label">ИИ</span>
        <strong class="compare-overall-value eval-summary__score eval-summary__score--${pctCls}">${escapeHtml(String(earned))} / ${escapeHtml(String(maxScore))}</strong>
        <span class="compare-overall-sub">${escapeHtml(pctStr)}</span>
      </div>
      <div class="compare-overall compare-overall--meta">
        <span class="compare-overall-label">Дата оценки</span>
        <strong class="compare-overall-value eval-summary__meta-value">${escapeHtml(dateInfo.primary)}</strong>
        <span class="compare-overall-sub">${escapeHtml(dateInfo.secondary || " ")}</span>
      </div>
      <div class="compare-overall compare-overall--meta">
        <span class="compare-overall-label">Модель</span>
        <strong class="compare-overall-value eval-summary__meta-value eval-summary__meta-value--mono">${escapeHtml(evaluation.model || "—")}</strong>
      </div>
    </div>
  `;
  if (reasoningHost) reasoningHost.replaceChildren();

  const traceRaw = evaluation.reasoning_trace;
  if (reasoningHost && traceRaw != null && String(traceRaw).trim()) {
    const det = document.createElement("details");
    det.className = "eval-reasoning";
    const summ = document.createElement("summary");
    summ.className = "eval-reasoning-summary";
    summ.textContent = "Ход рассуждений модели";
    det.appendChild(summ);
    det.appendChild(renderEvalReasoningBody(traceRaw));
    reasoningHost.appendChild(det);
  }

  tbody.innerHTML = "";
  for (const c of evaluation.criteria || []) {
    const tr = document.createElement("tr");
    tr.className = scoreClass(c.passed);
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
    scoreEl.title = formatDecisionHint(c.passed);
    scoreEl.textContent = formatDecisionText(c.passed);

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
    appendCriterionNameWithBadge(nameEl, c.name || c.id, c.weight || 1, "crit-name-wrap");
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
  _showWorkspaceJobLog = false;
  syncEvalPanelView();
  maybeLoadCompareView();
}

function maybeLoadCompareView() {
  if (_currentEvalMode !== "compare" || !selectedStem || _evalTransientState.comparePending) return;
  const ws = _lastWorkspaceData;
  const runtime = (ws && ws.comparison_runtime) || {};
  if (ws && compareEvalCacheMatches(ws, selectedStem)) {
    renderCompareResult(_compareEvalCache.data);
    return;
  }
  const stored = ws && ws.comparison_state && ws.comparison_state.payload;
  const hasStoredAnalysis = Boolean(stored && stored.llm_analysis);
  if (stored && ws && ws.human_eval_state && ws.human_eval_state.compared_at && hasStoredAnalysis) {
    renderCompareResult(stored);
    return;
  }
  if (runtime.pending) {
    const summaryEl = document.getElementById("compare-summary");
    const tableWrap = document.getElementById("compare-table-wrap");
    const analysisEl = document.getElementById("compare-analysis");
    if (summaryEl) {
      summaryEl.innerHTML =
        '<p style="color:var(--muted);font-size:0.82rem">Сравнение запущено и считается в фоне…</p>';
    }
    if (tableWrap) tableWrap.innerHTML = "";
    if (analysisEl) analysisEl.innerHTML = "";
    return;
  }
  if (runtime.error) {
    const summaryEl = document.getElementById("compare-summary");
    const tableWrap = document.getElementById("compare-table-wrap");
    const analysisEl = document.getElementById("compare-analysis");
    if (summaryEl) {
      summaryEl.innerHTML = `<p style="color:var(--danger)">${escapeHtml(runtime.error)}</p>`;
    }
    if (tableWrap) tableWrap.innerHTML = "";
    if (analysisEl) analysisEl.innerHTML = "";
    return;
  }
  if (stored || (ws && ws.human_evaluation && ws.ai_evaluation_available)) {
    void runComparison(selectedStem);
  }
}

function updateEvalToggleState(ws) {
  const compareBtn = document.getElementById("eval-compare");
  const state = (ws && ws.human_eval_state) || {};
  const permissions = workspacePermissions(ws);
  const humanPublished = Boolean(state.published_at);
  if (compareBtn) {
    const hasAi = Boolean(ws && ws.ai_evaluation_available);
    const hasHuman = Boolean(ws && ws.human_evaluation);
    const hasComparison = Boolean(ws && ws.comparison_state && ws.comparison_state.payload);
    compareBtn.disabled = !(
      !_evalTransientState.publishPending &&
      (
        hasComparison ||
        (permissions.can_compare && hasAi && hasHuman && (humanPublished || isAdmin()))
      )
    );
  }
  const saveBtn = document.getElementById("human-eval-save");
  if (saveBtn) {
    saveBtn.disabled =
      !ws ||
      !ws.criteria_content ||
      humanPublished ||
      !_humanEvalDirty ||
      !permissions.can_save_human_eval;
  }
  const publishBtn = document.getElementById("human-eval-publish");
  if (publishBtn) {
    publishBtn.disabled =
      !ws ||
      !ws.criteria_content ||
      humanPublished ||
      _humanEvalDirty ||
      !ws.human_evaluation ||
      !permissions.can_publish_human_eval ||
      _evalTransientState.publishPending;
  }
  syncEvalPanelView();
}

function setupEvalModeToggle() {
  const toggle = document.getElementById("eval-mode-toggle");
  if (toggle) {
    toggle.addEventListener("click", (e) => {
      const btn = e.target.closest(".eval-mode-btn");
      if (!btn) return;
      if (btn.disabled) return;
      switchEvalMode(btn.dataset.mode);
    });
  }

  const compareBtn = document.getElementById("eval-compare");
  if (compareBtn) {
    compareBtn.addEventListener("click", async () => {
      if (compareBtn.disabled || !selectedStem) return;
      switchEvalMode("compare");
    });
  }

  const pipelineToggleBtn = document.getElementById("ai-pipeline-toggle");
  if (pipelineToggleBtn) {
    pipelineToggleBtn.addEventListener("click", () => {
      if (!isAiPipelineSuccessfullyComplete(_lastWorkspaceData)) return;
      _showCompletedAiPipelineCard = !_showCompletedAiPipelineCard;
      syncEvalPanelView();
    });
  }

  const saveBtn = document.getElementById("human-eval-save");
  if (saveBtn) {
    saveBtn.addEventListener("click", async () => {
      if (!selectedStem || !_lastWorkspaceData) return;
      await saveHumanEval(selectedStem);
    });
  }
  const publishBtn = document.getElementById("human-eval-publish");
  if (publishBtn) {
    publishBtn.addEventListener("click", async () => {
      if (!selectedStem || !_lastWorkspaceData) return;
      await saveHumanEval(selectedStem, { publish: true });
    });
  }
}

/* --- Human eval form --- */

function renderHumanSummary(summaryEl, humanEval, state, options = {}) {
  if (!summaryEl) return;
  summaryEl.innerHTML = "";
  if (!humanEval) return;
  const { readOnly = false } = options;
  const pct = humanEval.overall_average != null ? `${Number(humanEval.overall_average).toFixed(1)}%` : "—";
  const cls = scoreClass(humanEval.overall_average);
  const dateInfo = formatEvaluatedAtParts(humanEval.evaluated_at || "");
  const statusText = state && state.published_at ? "Опубликован" : "Черновик";
  summaryEl.innerHTML = `
    <div class="eval-summary-metrics">
      <div class="compare-overall compare-overall--human">
        <span class="compare-overall-label">Человек</span>
        <strong class="compare-overall-value eval-summary__score eval-summary__score--${cls}">${escapeHtml(String(humanEval.earned_score ?? "—"))} / ${escapeHtml(String(humanEval.max_score ?? "—"))}</strong>
        <span class="compare-overall-sub">${escapeHtml(pct)}</span>
      </div>
      <div class="compare-overall compare-overall--meta">
        <span class="compare-overall-label">Дата оценки</span>
        <strong class="compare-overall-value eval-summary__meta-value">${escapeHtml(dateInfo.primary)}</strong>
        <span class="compare-overall-sub">${escapeHtml(dateInfo.secondary || " ")}</span>
      </div>
      <div class="compare-overall compare-overall--meta">
        <span class="compare-overall-label">Статус</span>
        <strong class="compare-overall-value eval-summary__meta-value">${escapeHtml(statusText)}</strong>
        ${readOnly ? '<span class="compare-overall-sub">Только чтение</span>' : ""}
      </div>
    </div>
  `;
}

function createHumanChoiceButton(row, label, value, current, disabled) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "human-eval-choice" + (current === value ? " human-eval-choice--active" : "");
  btn.dataset.value = value === true ? "true" : "false";
  btn.textContent = label;
  btn.disabled = disabled;
  btn.addEventListener("click", () => {
    if (disabled) return;
    row.dataset.passed = String(value);
    row.querySelectorAll(".human-eval-choice").forEach((node) => {
      node.classList.toggle("human-eval-choice--active", node.dataset.value === String(value));
    });
    syncHumanEvalDirtyFromForm(row.closest(".human-eval-form"));
  });
  return btn;
}

function renderHumanEvalForm(ws) {
  const form = document.getElementById("human-eval-form");
  const summaryEl = document.getElementById("human-eval-summary");
  if (!form) return;
  form.innerHTML = "";

  const humanEval = ws && ws.human_evaluation;
  const criteria = ws && ws.criteria_content ? (ws.criteria_content.criteria || []) : [];
  const state = (ws && ws.human_eval_state) || {};
  const readOnly = Boolean(workspacePermissions(ws).read_only);
  const locked = Boolean(state.published_at || readOnly);
  const draftBaseline = [];
  renderHumanSummary(summaryEl, humanEval, state, { readOnly });

  if (!criteria.length) {
    resetHumanEvalDraftState([]);
    if (ws && ws.criteria_resolution_error) {
      form.innerHTML = `<p class="block-missing">${escapeHtml(ws.criteria_resolution_error)}</p>`;
    } else {
      form.innerHTML = '<p class="block-missing">Для этой записи не найден чеклист типа тренировки.</p>';
    }
    return;
  }

  const humanMap = {};
  if (humanEval && humanEval.criteria) {
    for (const c of humanEval.criteria) {
      humanMap[c.id] = c;
    }
  }

  for (const c of criteria) {
    const existing = humanMap[c.id];
    const row = document.createElement("div");
    row.className = "human-eval-row" + (locked ? " human-eval-row--locked" : "");
    row.dataset.critId = c.id;
    row.dataset.critName = c.name || c.id;
    row.dataset.critWeight = String(c.weight || 1);
    row.dataset.passed = existing && existing.passed != null ? String(existing.passed) : "";
    draftBaseline.push({
      id: c.id,
      name: c.name || c.id,
      passed: existing && existing.passed != null ? existing.passed : null,
      weight: Number(c.weight || 1),
      comment: existing ? existing.comment || "" : "",
    });

    const head = document.createElement("div");
    head.className = "human-eval-row-head";

    const nameEl = document.createElement("div");
    nameEl.className = "human-eval-crit-name";
    nameEl.textContent = c.name || c.id;

    const choiceWrap = document.createElement("div");
    choiceWrap.className = "human-eval-choice-wrap";
    const current = existing && existing.passed != null ? existing.passed : null;
    choiceWrap.appendChild(createHumanChoiceButton(row, "Да", true, current, locked));
    choiceWrap.appendChild(createHumanChoiceButton(row, "Нет", false, current, locked));

    head.appendChild(nameEl);
    appendCriterionNameWithBadge(nameEl, c.name || c.id, c.weight || 1, "human-eval-crit-name-wrap");
    head.appendChild(choiceWrap);

    const commentInput = document.createElement("textarea");
    row.appendChild(head);
    if (locked) {
      const commentText = document.createElement("div");
      commentText.className = "human-eval-comment-text crit-comment-text";
      commentText.textContent = (existing && existing.comment) || "—";
      row.appendChild(commentText);
    } else {
      commentInput.className = "human-eval-comment-input";
      commentInput.placeholder = "Комментарий…";
      commentInput.rows = 1;
      commentInput.value = existing ? (existing.comment || "") : "";
      commentInput.disabled = false;
      commentInput.addEventListener("input", () => {
        syncHumanEvalDirtyFromForm(form);
      });
      row.appendChild(commentInput);
    }
    form.appendChild(row);
  }
  resetHumanEvalDraftState(draftBaseline);
}

async function saveHumanEval(stem, options = {}) {
  const { publish = false } = options;
  const form = document.getElementById("human-eval-form");
  if (!form) return;
  if (publish) {
    const ok = await showAppConfirm(
      "Опубликовать чеклист можно только один раз. После публикации редактирование будет заблокировано, а сравнение запустится автоматически. Продолжить?",
      { title: "Публикация чеклиста", okLabel: "Опубликовать" },
    );
    if (!ok) return;
  }
  const criteria = collectHumanEvalDraftFromForm(form);

  const ws = _lastWorkspaceData;
  const critFile = ws ? ws.evaluation_criteria : null;
  const q = critFile ? `?criteria=${encodeURIComponent(critFile)}` : "";

  try {
    if (publish) {
      _evalTransientState.publishPending = true;
      _evalTransientState.compareError = "";
      syncEvalPanelView();
    }
    const r = await apiFetch(`${API}/api/workspace/${encodeURIComponent(stem)}/human-eval${q}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ criteria, criteria_file: critFile }),
    });
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      await showAppAlert(d.detail || "Не удалось сохранить");
      return;
    }
    if (publish) {
      const pr = await apiFetch(
        `${API}/api/workspace/${encodeURIComponent(stem)}/human-eval/publish${q}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: "{}",
        }
      );
      if (!pr.ok) {
        const pd = await pr.json().catch(() => ({}));
        await showAppAlert(pd.detail || "Не удалось опубликовать");
        return;
      }
      await pr.json().catch(() => ({}));
    }
    await loadWorkspace(stem, true);
    if (publish) switchEvalMode("compare");
  } catch (e) {
    await showAppAlert(String(e));
  } finally {
    if (publish) {
      _evalTransientState.publishPending = false;
      syncEvalPanelView();
    }
  }
}

/* --- Compare view --- */

function evaluationFingerprint(ev) {
  if (!ev || typeof ev !== "object") return "";
  const crit = Array.isArray(ev.criteria) ? ev.criteria : [];
  const sorted = crit
    .slice()
    .sort((a, b) => String(a.id).localeCompare(String(b.id)))
    .map((c) => [c.id, c.passed, c.weight, c.comment != null ? String(c.comment) : ""]);
  return JSON.stringify({
    es: ev.earned_score,
    ms: ev.max_score,
    oa: ev.overall_average,
    at: ev.evaluated_at != null ? String(ev.evaluated_at) : "",
    c: sorted,
  });
}

function compareEvalCacheMatches(ws, stem) {
  if (!_compareEvalCache || !ws || stem !== selectedStem) return false;
  const crit = ws.evaluation_criteria || "";
  const aiFp = evaluationFingerprint(ws.evaluation);
  const huFp = evaluationFingerprint(ws.human_evaluation);
  const c = _compareEvalCache;
  return (
    c.stem === stem &&
    c.criteria === crit &&
    c.aiFp === aiFp &&
    c.huFp === huFp
  );
}

function diffClass(diff) {
  if (diff == null) return "";
  const value = Math.abs(Number(diff));
  if (Number.isNaN(value)) return "";
  if (value < 20) return "ok";
  if (value < 40) return "warn";
  return "danger";
}

function formatComparisonDiff(data) {
  if (!data || data.overall_diff == null) return "—";
  const diff = `${Number(data.overall_diff).toFixed(1)}%`;
  const mismatchCount = Number(data.mismatch_count);
  const comparedCount = Number(data.compared_count);
  if (Number.isFinite(mismatchCount) && Number.isFinite(comparedCount) && comparedCount > 0) {
    return `${diff} (${Math.round(mismatchCount)} из ${Math.round(comparedCount)} критериев)`;
  }
  return diff;
}

function formatComparisonDiffValue(data) {
  if (!data || data.overall_diff == null) return "—";
  return `${Number(data.overall_diff).toFixed(1)}%`;
}

async function runComparison(stem, options = {}) {
  const force = Boolean(options.force);
  if (_evalTransientState.comparePending) return;
  const summaryEl = document.getElementById("compare-summary");
  const tableWrap = document.getElementById("compare-table-wrap");
  const analysisEl = document.getElementById("compare-analysis");
  if (summaryEl) summaryEl.innerHTML = '<p style="color:var(--muted);font-size:0.82rem">Сравниваю…</p>';
  if (tableWrap) tableWrap.innerHTML = "";
  if (analysisEl) analysisEl.innerHTML = "";

  const ws = _lastWorkspaceData;
  const critFile = ws ? ws.evaluation_criteria : null;
  const runtime = (ws && ws.comparison_runtime) || {};
  const stored = ws && ws.comparison_state && ws.comparison_state.payload;
  if (!force && stored && ws && ws.human_eval_state && ws.human_eval_state.compared_at && stored.llm_analysis) {
    renderCompareResult(stored);
    return;
  }
  if (runtime.pending) {
    if (summaryEl) {
      summaryEl.innerHTML =
        '<p style="color:var(--muted);font-size:0.82rem">Сравнение уже запущено и считается в фоне…</p>';
    }
    scheduleCompareStatusPoll(stem, critFile || undefined);
    return;
  }

  _evalTransientState.comparePending = true;
  _evalTransientState.compareError = "";
  syncEvalModeButtonStates(ws);
  try {
    const r = await apiFetch(`${API}/api/workspace/${encodeURIComponent(stem)}/compare-eval`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ criteria: critFile, force }),
    });
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      _evalTransientState.compareError = String(d.detail || "Ошибка сравнения");
      if (summaryEl) summaryEl.innerHTML = `<p style="color:var(--danger)">${escapeHtml(d.detail || "Ошибка сравнения")}</p>`;
      return;
    }
    const data = await r.json();
    _evalTransientState.compareError = "";
    const wsSnap = _lastWorkspaceData;
    if (wsSnap && selectedStem === stem) {
      _lastWorkspaceData = {
        ...wsSnap,
        comparison_runtime: { pending: false, error: null },
        comparison_state: {
          ...((wsSnap && wsSnap.comparison_state) || {}),
          payload: data,
        },
      };
      cacheCompareResultForWorkspace(wsSnap, stem, data);
      updateWorkspaceHeadStatus(_lastWorkspaceData);
      updateEvalToggleState(_lastWorkspaceData);
    }
    renderCompareResult(data);
  } catch (e) {
    _evalTransientState.compareError = String(e);
    if (summaryEl) summaryEl.innerHTML = `<p style="color:var(--danger)">${escapeHtml(String(e))}</p>`;
  } finally {
    _evalTransientState.comparePending = false;
    syncEvalModeButtonStates(_lastWorkspaceData);
  }
}

function renderCompareResult(data) {
  const summaryEl = document.getElementById("compare-summary");
  const tableWrap = document.getElementById("compare-table-wrap");
  const analysisEl = document.getElementById("compare-analysis");

  const oc = diffClass(data.overall_diff);
  const diffLabel = formatComparisonDiffValue(data);
  const mismatchCount = Number(data.mismatch_count);
  const comparedCount = Number(data.compared_count);
  const diffSubtitle =
    data.overall_diff == null
      ? "—"
      : Number.isFinite(mismatchCount) && Number.isFinite(comparedCount) && comparedCount > 0
        ? `${Math.round(mismatchCount)} из ${Math.round(comparedCount)} критериев не совпали`
        : oc === "danger"
          ? "Сильное расхождение"
          : oc === "warn"
            ? "Среднее расхождение"
            : "Незначительное расхождение";
  const analysisHtml = data.llm_analysis
    ? `<div class="compare-summary-note"><div class="compare-summary-note-label">Комментарий ИИ</div><div class="compare-summary-note-body">${formatCompareAnalysisHtml(data.llm_analysis)}</div></div>`
    : "";
  summaryEl.innerHTML = `
    <div class="compare-summary-card">
      <div class="compare-summary-metrics">
        <div class="compare-overall compare-overall--ai">
          <span class="compare-overall-label">ИИ</span>
          <strong class="compare-overall-value">${data.ai_overall != null ? data.ai_overall : "—"} / ${data.max_score != null ? data.max_score : "—"}</strong>
          <span class="compare-overall-sub">${data.ai_percent != null ? Number(data.ai_percent).toFixed(1) + "%" : "—"}</span>
        </div>
        <div class="compare-overall compare-overall--human">
          <span class="compare-overall-label">Человек</span>
          <strong class="compare-overall-value">${data.human_overall != null ? data.human_overall : "—"} / ${data.max_score != null ? data.max_score : "—"}</strong>
          <span class="compare-overall-sub">${data.human_percent != null ? Number(data.human_percent).toFixed(1) + "%" : "—"}</span>
        </div>
        <div class="compare-overall compare-overall--diff compare-diff-${oc}">
          <span class="compare-overall-label">Δ</span>
          <strong class="compare-overall-value">${diffLabel}</strong>
          <span class="compare-overall-sub">${diffSubtitle}</span>
        </div>
      </div>
      ${analysisHtml}
    </div>
  `;

  const table = document.createElement("table");
  table.className = "compare-table";
  table.innerHTML = `
    <thead>
      <tr>
        <th>Критерий</th>
        <th class="score-col">Вес</th>
        <th class="score-col">ИИ</th>
        <th class="score-col">Человек</th>
        <th class="diff-col">Статус</th>
      </tr>
    </thead>
  `;
  const tbody = document.createElement("tbody");
  for (const row of (data.rows || [])) {
    const dc = row.same === true ? "ok" : row.same === false ? "danger" : "";
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
      <td class="score-col">${row.weight != null ? row.weight : "—"}</td>
      <td class="score-col">${formatDecisionText(row.ai_passed)}</td>
      <td class="score-col">${formatDecisionText(row.human_passed)}</td>
      <td class="diff-col compare-diff-${dc}">${row.same === true ? "Совпало" : row.same === false ? "Разошлось" : "—"}</td>
    `;
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  tableWrap.innerHTML = "";
  tableWrap.appendChild(table);

  analysisEl.innerHTML = "";
}

function renderChecklistStaleBanner(ws) {
  const banner = document.getElementById("checklist-stale-banner");
  if (!banner) return;
  const info = ws && ws.checklist_stale;
  if (!info || !info.is_stale) {
    banner.style.display = "none";
    banner.textContent = "";
    return;
  }
  const currentName = info.current_display_name || info.current_slug || "новый чеклист";
  banner.textContent = `Оценка сделана по устаревшей версии чеклиста (${info.snapshot_version}). Хотите обновить оценку по «${currentName}» (${info.current_version})? Нажмите кнопку обновления.`;
  banner.style.display = "block";
}

function formatUploadWizardFiles(files) {
  return (files || []).map((file) => `${file.name} (${Math.round(file.size / 1024 / 1024 * 10) / 10} MB)`).join(", ");
}

function trainingTypesForDepartment(departmentCode) {
  const code = String(departmentCode || "").trim().toUpperCase();
  return (_metaTrainingTypes || []).filter((item) => {
    const itemDept = String(item.department || "").trim().toUpperCase();
    return code ? itemDept === code : false;
  });
}

async function fetchTrainingTypesForWizard() {
  const r = await apiFetch(`${API}/api/training-types`);
  if (!r.ok) return [];
  const items = await r.json().catch(() => []);
  _metaTrainingTypes = (Array.isArray(items) ? items : []).filter(
    (item) =>
      String(item.department || "").trim() &&
      String(item.checklist_slug || "").trim() &&
      String(item.checklist_slug || "").trim() !== "criteria",
  );
  return _metaTrainingTypes;
}

async function fetchUploadQuota() {
  const r = await apiFetch(`${API}/api/upload/quota`);
  if (!r.ok) {
    _uploadQuota = null;
    return null;
  }
  _uploadQuota = await r.json().catch(() => null);
  return _uploadQuota;
}

function renderUploadQuota(files) {
  const quotaEl = document.getElementById("upload-wizard-quota");
  const submitBtn = document.getElementById("upload-wizard-submit");
  const errEl = document.getElementById("upload-wizard-error");
  if (!quotaEl || !submitBtn) return;
  const quota = _uploadQuota;
  let message = "Квота не определена.";
  let blockingError = "";
  if (quota && quota.daily_limit != null) {
    message = `Сегодня загружено ${quota.daily_uploaded_count ?? 0} из ${quota.daily_limit}. Осталось: ${quota.daily_remaining ?? 0}.`;
    if ((files || []).length > Number(quota.daily_remaining ?? 0)) {
      blockingError = `Выбрано ${files.length} файл(ов), а по квоте осталось только ${quota.daily_remaining ?? 0}.`;
    }
  } else if (quota && quota.auth_enabled === false) {
    message = "Для текущего режима входа персональная дневная квота не применяется.";
  }
  if (quota?.blocked_reasons?.length) {
    blockingError = quota.blocked_reasons[0];
  }
  quotaEl.textContent = message;
  submitBtn.disabled = Boolean(blockingError);
  if (errEl) {
    if (blockingError) {
      errEl.textContent = blockingError;
      errEl.style.display = "block";
    } else {
      errEl.style.display = "none";
      errEl.textContent = "";
    }
  }
}

async function renderUploadWizardChecklist(trainingTypeSlug) {
  const preview = document.getElementById("upload-wizard-checklist-preview");
  if (!preview) return;
  const item = (_metaTrainingTypes || []).find((row) => row.slug === trainingTypeSlug);
  if (!item || !item.checklist_slug) {
    preview.innerHTML = '<p class="block-missing">Для выбранного типа тренировки чеклист не привязан.</p>';
    return;
  }
  const r = await apiFetch(`${API}/api/criteria/content/${encodeURIComponent(item.checklist_slug)}`);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    preview.innerHTML = `<p class="block-missing">${escapeHtml(data.detail || "Не удалось загрузить чеклист")}</p>`;
    return;
  }
  const rows = Array.isArray(data.criteria) ? data.criteria : [];
  preview.innerHTML = `<ul>${rows.map((row) => `<li>${escapeHtml(row.name || row.id)} <span class="criteria-badge">${escapeHtml(criterionBadgeLabel(row.weight || 1))}</span></li>`).join("")}</ul>`;
}

async function fillUploadWizardTrainingTypes(departmentCode, selectedSlug = "") {
  const typeSel = document.getElementById("upload-wizard-training-type");
  if (!typeSel) return;
  const filtered = trainingTypesForDepartment(departmentCode);
  typeSel.innerHTML = "";
  const empty = document.createElement("option");
  empty.value = "";
  empty.textContent = filtered.length ? "Выберите тип тренировки" : "Для выбранного отдела типы не настроены";
  typeSel.appendChild(empty);
  for (const item of filtered) {
    const opt = document.createElement("option");
    opt.value = item.slug;
    opt.textContent = item.name;
    typeSel.appendChild(opt);
  }
  if (selectedSlug && filtered.some((item) => item.slug === selectedSlug)) {
    typeSel.value = selectedSlug;
  } else if (filtered.length === 1) {
    typeSel.value = filtered[0].slug;
  } else {
    typeSel.value = "";
  }
  await renderUploadWizardChecklist(typeSel.value);
}

async function openUploadWizard(files, options = {}) {
  if (!files || !files.length) return;
  const previewOnly = Boolean(options.previewOnly);
  _pendingUploadFiles = files;
  if (_authState.auth_enabled && _authState.profile_complete === false) {
    openProfileDialog();
    return;
  }
  await Promise.all([fetchTrainingTypesForWizard(), fetchUploadQuota()]);
  const dlg = document.getElementById("upload-wizard-dialog");
  const fileEl = document.getElementById("upload-wizard-files");
  const managerEl = document.getElementById("upload-wizard-manager");
  const locationEl = document.getElementById("upload-wizard-location");
  const departmentSel = document.getElementById("upload-wizard-department");
  const typeSel = document.getElementById("upload-wizard-training-type");
  const speakerCountEl = document.getElementById("upload-wizard-speaker-count");
  const previewNoteEl = document.getElementById("upload-wizard-preview-note");
  const submitBtn = document.getElementById("upload-wizard-submit");
  const errEl = document.getElementById("upload-wizard-error");
  if (!dlg || !fileEl || !managerEl || !locationEl || !departmentSel || !typeSel || !submitBtn) return;
  _uploadWizardPreviewOnly = previewOnly;
  errEl.style.display = "none";
  errEl.textContent = "";
  if (previewNoteEl) {
    previewNoteEl.hidden = !previewOnly;
    previewNoteEl.textContent = previewOnly
      ? "В режиме инструкции окно показано как пример. Реальная загрузка начнётся после выбора файла."
      : "В режиме инструкции окно показано как пример. Реальная загрузка начнётся после выбора файла.";
  }
  fileEl.textContent = formatUploadWizardFiles(files);
  managerEl.textContent = _authState.full_name || _authState.display_name || _authState.user || "Не определён";
  locationEl.textContent = (_authState.location && (_authState.location.crm_name || _authState.location.name)) || "Не выбрана";
  departmentSel.value = currentUserDepartment() || "";
  if (speakerCountEl) speakerCountEl.value = "2";
  departmentSel.onchange = async () => {
    await fillUploadWizardTrainingTypes(departmentSel.value, "");
  };
  typeSel.onchange = () => renderUploadWizardChecklist(typeSel.value);
  await fillUploadWizardTrainingTypes(departmentSel.value, "");
  renderUploadQuota(files);
  submitBtn.textContent = "Загрузить";
  if (previewOnly) {
    submitBtn.disabled = true;
  }
  dlg.showModal();
}

function uploadFile(file, trainingTypeSlug, departmentCode, speakerCount, uploadItem) {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("training_type_slug", trainingTypeSlug || "");
  fd.append("department", departmentCode || "");
  fd.append("speaker_count", String(speakerCount || 2));
  return new Promise((resolve) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API}/api/upload`, true);
    xhr.withCredentials = true;

    xhr.upload.onprogress = (event) => {
      if (!uploadItem) return;
      const progress = event.lengthComputable
        ? Math.max(0, Math.min((event.loaded / event.total) * 100, 99))
        : Math.max(Number(uploadItem.upload_progress) || 0, 5);
      upsertUploadLibraryItem(uploadItem, {
        upload_phase: "uploading",
        upload_progress: progress,
      });
    };

    xhr.onload = async () => {
      let data = {};
      try {
        data = JSON.parse(xhr.responseText || "{}");
      } catch (_err) {
        data = { detail: xhr.responseText || xhr.statusText || "Ошибка загрузки" };
      }
      if (xhr.status < 200 || xhr.status >= 300) {
        const msg = data.detail || data.message || "Ошибка загрузки";
        upsertUploadLibraryItem(uploadItem, {
          upload_phase: "error",
          upload_error: msg,
          upload_progress: 0,
        });
        await showAppAlert(`${file.name}: ${msg}`, { title: "Ошибка загрузки" });
        resolve(null);
        return;
      }
      upsertUploadLibraryItem(uploadItem, {
        upload_phase: "processing",
        upload_progress: 100,
        server_stem: data.stem,
      });
      resolve(data.stem || null);
    };

    xhr.onerror = async () => {
      const msg = "Не удалось передать файл на сервер";
      upsertUploadLibraryItem(uploadItem, {
        upload_phase: "error",
        upload_error: msg,
        upload_progress: 0,
      });
      await showAppAlert(`${file.name}: ${msg}`, { title: "Ошибка загрузки" });
      resolve(null);
    };

    xhr.send(fd);
  });
}

async function uploadFiles(files, trainingTypeSlug, departmentCode, speakerCount) {
  if (!files || !files.length) return;
  const metaEl = document.getElementById("workspace-title-meta");
  const workspaceEl = document.getElementById("workspace");
  const restoreStem = selectedStem;
  const restoreCriteria = lastWorkspaceCriteriaRequested || undefined;
  const cancelBtn = document.getElementById("workspace-job-cancel");
  const afterStop = document.getElementById("workspace-job-after-stop");
  updateWorkspaceJobStreamPanel(null);
  if (metaEl && workspaceEl && workspaceEl.style.display !== "none") {
    renderWorkspaceMetaItems(
      metaEl,
      [{ text: `Загрузка ${files.length} файл(ов)…`, tone: "info" }],
      `Загрузка ${files.length} файл(ов)…`,
    );
  }
  if (cancelBtn) cancelBtn.hidden = true;
  if (afterStop) afterStop.hidden = true;
  syncWorkspaceJobActionsVisibility();
  let lastStem = null;
  for (let i = 0; i < files.length; i++) {
    const uploadItem = createUploadLibraryItem(files[i], trainingTypeSlug, departmentCode);
    upsertUploadLibraryItem(uploadItem);
    if (metaEl && workspaceEl && workspaceEl.style.display !== "none") {
      renderWorkspaceMetaItems(
        metaEl,
        [{ text: `Загрузка ${i + 1} / ${files.length}: ${files[i].name}…`, tone: "info" }],
        `Загрузка ${i + 1} / ${files.length}: ${files[i].name}…`,
      );
    }
    const stem = await uploadFile(files[i], trainingTypeSlug, departmentCode, speakerCount, uploadItem);
    if (!stem) break;
    lastStem = stem;
    uploadItem.server_stem = stem;
  }
  if (cancelBtn) cancelBtn.hidden = true;
  if (lastStem) {
    selectedStem = lastStem;
    await loadLibrary();
    for (const item of Array.from(_activeUploadItems.values())) {
      if (item.server_stem) removeUploadLibraryItem(item);
    }
    startLibraryPoll();
    collapseSidebarOnMobileIfNeeded();
    await loadWorkspace(lastStem);
    document.querySelectorAll(".library-row").forEach((el) => {
      el.classList.toggle("selected", el.dataset.stem === selectedStem);
    });
  } else {
    await loadLibrary();
    if (restoreStem) {
      await loadWorkspace(restoreStem, true, restoreCriteria);
    }
  }
}

function setupUploadWizard() {
  const dlg = document.getElementById("upload-wizard-dialog");
  const closeBtn = document.getElementById("upload-wizard-close");
  const cancelBtn = document.getElementById("upload-wizard-cancel");
  const submitBtn = document.getElementById("upload-wizard-submit");
  const departmentSel = document.getElementById("upload-wizard-department");
  const typeSel = document.getElementById("upload-wizard-training-type");
  const speakerCountEl = document.getElementById("upload-wizard-speaker-count");
  const previewNoteEl = document.getElementById("upload-wizard-preview-note");
  const errEl = document.getElementById("upload-wizard-error");
  if (!dlg || !submitBtn || !departmentSel || !typeSel) return;

  function close() {
    dlg.close();
    _pendingUploadFiles = [];
    _uploadWizardPreviewOnly = false;
    submitBtn.disabled = false;
    submitBtn.textContent = "Загрузить";
    if (previewNoteEl) previewNoteEl.hidden = true;
    if (errEl) {
      errEl.style.display = "none";
      errEl.textContent = "";
    }
  }

  closeBtn?.addEventListener("click", close);
  cancelBtn?.addEventListener("click", close);
  dlg.addEventListener("click", (e) => {
    if (e.target === dlg) close();
  });
  submitBtn.addEventListener("click", async () => {
    if (_uploadWizardPreviewOnly) return;
    const departmentCode = departmentSel.value;
    if (!departmentCode) {
      if (errEl) {
        errEl.textContent = "Выберите отдел.";
        errEl.style.display = "block";
      }
      return;
    }
    const trainingTypeSlug = typeSel.value;
    if (!trainingTypeSlug) {
      if (errEl) {
        errEl.textContent = "Выберите тип тренировки.";
        errEl.style.display = "block";
      }
      return;
    }
    const speakerCount = Math.max(1, Math.min(Number.parseInt(speakerCountEl?.value || "2", 10) || 2, 8));
    if (speakerCountEl) speakerCountEl.value = String(speakerCount);
    const files = [..._pendingUploadFiles];
    renderUploadQuota(files);
    if (submitBtn.disabled) return;
    close();
    await uploadFiles(files, trainingTypeSlug, departmentCode, speakerCount);
  });
}

function setupDropzone() {
  const dz = document.getElementById("dropzone");
  const input = document.getElementById("file-input");

  input.addEventListener("change", () => {
    if (input.files && input.files.length) openUploadWizard(Array.from(input.files));
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
    if (files && files.length) openUploadWizard(Array.from(files));
  });
}

async function ensureAuthLocationsLoaded() {
  if (_authLocations.length) return _authLocations;
  const r = await apiFetch(`${API}/api/auth/locations`);
  if (!r.ok) return [];
  const items = await r.json().catch(() => []);
  _authLocations = Array.isArray(items) ? items : [];
  return _authLocations;
}

async function openProfileDialog() {
  const dlg = document.getElementById("profile-dialog");
  const titleEl = document.getElementById("profile-dialog-title");
  const hintEl = document.getElementById("profile-dialog-hint");
  const closeBtn = document.getElementById("profile-dialog-close");
  const fullName = document.getElementById("profile-full-name");
  const locationSel = document.getElementById("profile-location");
  const departmentSel = document.getElementById("profile-department");
  const errorEl = document.getElementById("profile-error");
  const passwordSection = document.getElementById("profile-password-section");
  const passwordError = document.getElementById("profile-password-error");
  const passwordSuccess = document.getElementById("profile-password-success");
  const currentPassword = document.getElementById("profile-current-password");
  const newPassword = document.getElementById("profile-new-password");
  const repeatPassword = document.getElementById("profile-new-password-repeat");
  if (!dlg || !fullName || !locationSel || !departmentSel) return;
  const locations = await ensureAuthLocationsLoaded();
  locationSel.innerHTML = '<option value="">Выберите локацию</option>';
  for (const row of locations) {
    const opt = document.createElement("option");
    opt.value = row.id;
    opt.textContent = row.crm_name || row.name || row.id;
    locationSel.appendChild(opt);
  }
  fullName.value = _authState.full_name || _authState.display_name || "";
  locationSel.value = (_authState.location && _authState.location.id) || "";
  departmentSel.value = currentUserDepartment() || "";
  const profileRequired = Boolean(_authState.auth_enabled && _authState.profile_complete === false);
  const localAccount = String((_authState && _authState.auth_type) || "").trim().toLowerCase() === "local";
  if (titleEl) {
    titleEl.textContent = profileRequired ? "Заполните профиль" : "Настройки аккаунта";
  }
  if (hintEl) {
    hintEl.textContent = profileRequired
      ? "Для продолжения укажите ФИО, локацию и отдел. Без этого загрузка записей недоступна."
      : "Здесь можно изменить ФИО, локацию, отдел и пароль.";
  }
  if (closeBtn) {
    closeBtn.hidden = profileRequired;
  }
  if (passwordSection) {
    passwordSection.hidden = profileRequired || !localAccount;
  }
  if (errorEl) {
    errorEl.style.display = "none";
    errorEl.textContent = "";
  }
  if (passwordError) {
    passwordError.style.display = "none";
    passwordError.textContent = "";
  }
  if (passwordSuccess) {
    passwordSuccess.style.display = "none";
    passwordSuccess.textContent = "";
  }
  if (currentPassword) currentPassword.value = "";
  if (newPassword) newPassword.value = "";
  if (repeatPassword) repeatPassword.value = "";
  dlg.showModal();
}

function setupProfileDialog() {
  const dlg = document.getElementById("profile-dialog");
  const closeBtn = document.getElementById("profile-dialog-close");
  const saveBtn = document.getElementById("profile-save");
  const fullName = document.getElementById("profile-full-name");
  const locationSel = document.getElementById("profile-location");
  const departmentSel = document.getElementById("profile-department");
  const errorEl = document.getElementById("profile-error");
  const passwordSaveBtn = document.getElementById("profile-password-save");
  const currentPassword = document.getElementById("profile-current-password");
  const newPassword = document.getElementById("profile-new-password");
  const repeatPassword = document.getElementById("profile-new-password-repeat");
  const passwordError = document.getElementById("profile-password-error");
  const passwordSuccess = document.getElementById("profile-password-success");
  if (!dlg || !saveBtn || !fullName || !locationSel || !departmentSel || !errorEl) return;
  function profileCompletionRequired() {
    return Boolean(_authState.auth_enabled && _authState.profile_complete === false);
  }
  dlg.addEventListener("cancel", (e) => {
    if (profileCompletionRequired()) {
      e.preventDefault();
    }
  });
  if (closeBtn) {
    closeBtn.addEventListener("click", () => {
      if (profileCompletionRequired()) return;
      dlg.close();
    });
  }
  saveBtn.addEventListener("click", async () => {
    errorEl.style.display = "none";
    errorEl.textContent = "";
    try {
      const r = await apiFetch(`${API}/api/auth/profile`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          full_name: fullName.value.trim(),
          location_id: locationSel.value,
          department: departmentSel.value,
        }),
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        errorEl.textContent = data.detail || "Не удалось сохранить профиль";
        errorEl.style.display = "block";
        return;
      }
      _authState = { ..._authState, ...(data || {}) };
      dlg.close();
      await initAuthUi();
      if (_pendingUploadFiles.length) {
        const files = [..._pendingUploadFiles];
        _pendingUploadFiles = [];
        await openUploadWizard(files);
      }
      await maybeAutoStartOnboardingTour();
    } catch (e) {
      errorEl.textContent = String(e);
      errorEl.style.display = "block";
    }
  });
  if (
    passwordSaveBtn &&
    currentPassword &&
    newPassword &&
    repeatPassword &&
    passwordError &&
    passwordSuccess
  ) {
    passwordSaveBtn.addEventListener("click", async () => {
      passwordError.style.display = "none";
      passwordError.textContent = "";
      passwordSuccess.style.display = "none";
      passwordSuccess.textContent = "";
      const currentValue = currentPassword.value || "";
      const newValue = newPassword.value || "";
      const repeatValue = repeatPassword.value || "";
      if (!currentValue.trim()) {
        passwordError.textContent = "Введите текущий пароль";
        passwordError.style.display = "block";
        return;
      }
      if (!newValue) {
        passwordError.textContent = "Введите новый пароль";
        passwordError.style.display = "block";
        return;
      }
      if (newValue !== repeatValue) {
        passwordError.textContent = "Новый пароль и повтор не совпадают";
        passwordError.style.display = "block";
        return;
      }
      passwordSaveBtn.disabled = true;
      try {
        const r = await apiFetch(`${API}/api/auth/change-password`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            current_password: currentValue,
            new_password: newValue,
            new_password_repeat: repeatValue,
          }),
        });
        const data = await r.json().catch(() => ({}));
        if (!r.ok) {
          passwordError.textContent = data.detail || "Не удалось изменить пароль";
          passwordError.style.display = "block";
          return;
        }
        currentPassword.value = "";
        newPassword.value = "";
        repeatPassword.value = "";
        passwordSuccess.textContent = "Пароль обновлён";
        passwordSuccess.style.display = "block";
      } catch (e) {
        passwordError.textContent = String(e);
        passwordError.style.display = "block";
      } finally {
        passwordSaveBtn.disabled = false;
      }
    });
  }
}

const SIDEBAR_COLLAPSED_KEY = "fresh-fa-sidebar-collapsed";
/** Совпадает с max-width в app.css (колонка + мобильная сетка). */
const MOBILE_LAYOUT_MAX_PX = 880;

/** @type {((collapsed: boolean, persist?: boolean) => void) | null} */
let applySidebarCollapsed = null;

/** После выбора записи на узком экране — как «Скрыть панель», без записи в localStorage (десктоп-настройка не затирается). */
function collapseSidebarOnMobileIfNeeded() {
  if (!applySidebarCollapsed) return;
  try {
    if (!window.matchMedia(`(max-width: ${MOBILE_LAYOUT_MAX_PX}px)`).matches) return;
  } catch (_) {
    return;
  }
  applySidebarCollapsed(true, false);
}

function setupSidebarToggle() {
  const shell = document.getElementById("app-shell");
  const collapseBtn = document.getElementById("sidebar-collapse");
  const expandBtn = document.getElementById("sidebar-expand");
  const expandMobile = document.getElementById("sidebar-expand-mobile");
  if (!shell || !collapseBtn || !expandBtn) return;

  function apply(collapsed, persist = true) {
    shell.classList.toggle("sidebar-collapsed", collapsed);
    expandBtn.setAttribute("aria-hidden", collapsed ? "false" : "true");
    collapseBtn.setAttribute("aria-hidden", collapsed ? "true" : "false");
    if (expandMobile) {
      expandMobile.setAttribute("aria-hidden", collapsed ? "false" : "true");
    }
    if (persist) {
      try {
        if (collapsed) localStorage.setItem(SIDEBAR_COLLAPSED_KEY, "1");
        else localStorage.removeItem(SIDEBAR_COLLAPSED_KEY);
      } catch (_) {
        /* ignore */
      }
    }
  }

  applySidebarCollapsed = apply;

  let initial = false;
  try {
    initial = localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === "1";
  } catch (_) {
    /* ignore */
  }
  apply(initial);

  collapseBtn.addEventListener("click", () => apply(true));
  expandBtn.addEventListener("click", () => apply(false));
  if (expandMobile) expandMobile.addEventListener("click", () => apply(false));
}

async function initAuthUi() {
  const bar = document.getElementById("auth-user-bar");
  if (!bar) return;
  try {
    const sr = await apiFetch(`${API}/api/auth/status`);
    const st = await sr.json();
    if (!st.auth_enabled) {
      _authState = { ..._authState, ...(st || {}), is_admin: true };
      bar.style.display = "none";
      syncOnboardingReplayUi();
      return;
    }
    const mr = await apiFetch(`${API}/api/auth/me`);
    const me = await mr.json();
    _authState = me || _authState;
    if (me.user) {
      const nameEl = bar.querySelector(".auth-user-name");
      const subEl = bar.querySelector(".auth-user-sub");
      const labelEl = bar.querySelector(".auth-user-label");
      const dn = me.display_name && String(me.display_name).trim();
      const locationName = me.location && (me.location.crm_name || me.location.name);
      const departmentName = me.department_label || departmentLabel(me.department || "");
      if (nameEl) {
        if (dn) {
          nameEl.textContent = dn;
          if (subEl) {
            subEl.textContent = [me.user, locationName, departmentName, me.role === "admin" ? "admin" : ""]
              .filter(Boolean)
              .join(" · ");
            subEl.hidden = false;
          }
          if (labelEl) labelEl.textContent = me.profile_complete ? "Профиль заполнен" : "Нужно заполнить профиль";
        } else {
          nameEl.textContent = me.user;
          if (subEl) {
            subEl.textContent = [locationName, departmentName, me.role === "admin" ? "admin" : ""]
              .filter(Boolean)
              .join(" · ");
            subEl.hidden = !subEl.textContent;
          }
          if (labelEl) labelEl.textContent = me.auth_type === "local" ? "Локальный пользователь" : "Учётная запись";
        }
      }
      bar.style.display = "";
      const settingsBtn = bar.querySelector(".auth-settings");
      if (settingsBtn) {
        settingsBtn.onclick = async () => {
          await openProfileDialog();
        };
      }
      syncLibraryForeignToggle();
      const logoutBtn = bar.querySelector(".auth-logout");
      if (logoutBtn) {
        logoutBtn.onclick = async () => {
          await apiFetch(`${API}/api/auth/logout`, { method: "POST" });
          window.location.href = "/login.html";
        };
      }
      if (me.profile_complete === false) {
        await openProfileDialog();
      }
      syncOnboardingReplayUi();
    } else {
      bar.style.display = "none";
      syncLibraryForeignToggle();
      syncOnboardingReplayUi();
    }
  } catch (_) {
    bar.style.display = "none";
    syncLibraryForeignToggle();
    syncOnboardingReplayUi();
  }
}

function setupWorkspaceJobCancel() {
  document.addEventListener(
    "click",
    async (e) => {
      const raw = e.target;
      const btn = raw && raw.closest && raw.closest("#workspace-job-cancel");
      if (!btn || btn.disabled) return;
      const id =
        btn.getAttribute("data-job-id") ||
        (btn.dataset && btn.dataset.jobId) ||
        "";
      if (!id) {
        await showAppAlert("Нет id задачи — обновите страницу.");
        return;
      }
      e.preventDefault();
      e.stopPropagation();
      btn.disabled = true;
      btn.setAttribute("aria-busy", "true");
      const ac = new AbortController();
      const to = setTimeout(() => ac.abort(), 45000);
      try {
        const r = await apiFetch(`${API}/api/jobs/${encodeURIComponent(id)}/cancel`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: "{}",
          signal: ac.signal,
        });
        clearTimeout(to);
        let data = {};
        const ct = (r.headers.get("content-type") || "").includes("application/json");
        if (ct) data = await r.json().catch(() => ({}));
        else if (!r.ok) {
          const t = await r.text().catch(() => "");
          data = { detail: t || r.statusText };
        }
        if (!r.ok) {
          const d = data.detail;
          await showAppAlert(
            typeof d === "string"
              ? d
              : Array.isArray(d)
                ? d.map((x) => (x && x.msg) || String(x)).join("; ")
                : d
                  ? JSON.stringify(d)
                  : "Не удалось остановить",
          );
          return;
        }
        if (selectedStem) await loadWorkspace(selectedStem, true);
        await loadLibrary();
      } catch (err) {
        clearTimeout(to);
        if (err && err.name === "AbortError") {
          await showAppAlert("Таймаут запроса — сервер не ответил. Проверьте, что веб-сервер запущен.");
        } else {
          await showAppAlert(String(err));
        }
      } finally {
        btn.disabled = false;
        btn.removeAttribute("aria-busy");
      }
    },
    true,
  );
}

function setupWorkspacePipelineRestart() {
  document.addEventListener(
    "click",
    async (e) => {
      const res = e.target && e.target.closest && e.target.closest("#workspace-job-resume");
      const rst = e.target && e.target.closest && e.target.closest("#workspace-job-restart-full");
      const cmp = e.target && e.target.closest && e.target.closest("#workspace-compare-restart");
      if (!res && !rst && !cmp) return;
      e.preventDefault();
      e.stopPropagation();
      const stem = selectedStem;
      if (!stem) return;
      if (cmp) {
        cmp.disabled = true;
        cmp.setAttribute("aria-busy", "true");
        try {
          _currentEvalMode = "compare";
          _showWorkspaceJobLog = false;
          syncEvalPanelView();
          await runComparison(stem, { force: true });
          await loadLibrary();
          startLibraryPoll();
          await loadWorkspace(stem, true);
        } catch (err) {
          await showAppAlert(String(err));
        } finally {
          cmp.disabled = false;
          cmp.removeAttribute("aria-busy");
        }
        return;
      }
      const url = res
        ? `${API}/api/workspace/${encodeURIComponent(stem)}/pipeline/resume`
        : `${API}/api/workspace/${encodeURIComponent(stem)}/pipeline/restart`;
      try {
        const r = await apiFetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: "{}",
        });
        const data = await r.json().catch(() => ({}));
        if (!r.ok) {
          await showAppAlert(flattenApiDetail(data.detail) || "Не удалось запустить");
          return;
        }
        await loadLibrary();
        startLibraryPoll();
        await loadWorkspace(stem, true);
      } catch (err) {
        await showAppAlert(String(err));
      }
    },
    true,
  );
}

async function bootstrapApp() {
  await Promise.all([initAuthUi(), loadLibrary()]);
  syncOnboardingReplayUi();
  await maybeAutoStartOnboardingTour();
}

document.addEventListener("DOMContentLoaded", () => {
  setupSystemMessageDialog();
  setupOnboardingTour();
  setupSidebarToggle();
  setupLibraryControls();
  setupEvalToolbar();
  setupEvalModeToggle();
  setupCriteriaDialogs();
  setupMetaPanel();
  setupTrainingTypeDialog();
  setupProfileDialog();
  setupUploadWizard();
  setupDropzone();
  setupWorkspaceJobCancel();
  setupWorkspacePipelineRestart();
  setupWorkspaceJobStreamToggle();
  setupTranscriptDownload();
  void bootstrapApp();
});
