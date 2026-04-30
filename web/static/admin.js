const API = "";

const DEPARTMENT_LABELS = {
  "ОО": "Отдел оценки",
  "ОП": "Отдел продаж",
};

const SECTION_LABELS = {
  overview: "Обзор",
  jobs: "Задачи",
  users: "Пользователи",
  videos: "Видео",
  checklists: "Чеклисты",
  settings: "Настройки",
  audit: "Аудит",
};

const state = {
  me: null,
  reference: null,
  overview: null,
  jobs: [],
  users: [],
  videos: [],
  audit: [],
  settings: null,
  selectedJobId: null,
  selectedUser: null,
  selectedVideo: null,
  selectedChecklist: null,
  selectedTrainingType: null,
  checklistContent: null,
  activeSection: "overview",
  lastUpdatedAt: null,
  autoRefreshTimer: null,
  flashTimer: null,
};

function apiFetch(input, init) {
  const opts = init ? { ...init, credentials: "same-origin" } : { credentials: "same-origin" };
  return globalThis.fetch(input, opts);
}

function flattenApiDetail(detail) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((item) => item?.msg || String(item)).join("; ");
  if (detail && typeof detail === "object") return JSON.stringify(detail);
  return "";
}

async function apiJson(url, init) {
  const response = await apiFetch(url, init);
  const contentType = response.headers.get("content-type") || "";
  let payload = null;
  if (contentType.includes("application/json")) {
    payload = await response.json().catch(() => ({}));
  } else {
    const text = await response.text().catch(() => "");
    payload = text ? { detail: text } : {};
  }
  if (!response.ok) {
    const error = new Error(flattenApiDetail(payload?.detail) || response.statusText || "Ошибка API");
    error.payload = payload;
    throw error;
  }
  return payload;
}

function qs(id) {
  return document.getElementById(id);
}

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = value == null ? "" : String(value);
  return div.innerHTML;
}

function formatDateTime(iso) {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return String(iso);
  return date.toLocaleString("ru-RU", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function departmentLabel(code) {
  return DEPARTMENT_LABELS[String(code || "").trim().toUpperCase()] || String(code || "").trim() || "—";
}

function boolValue(value) {
  return String(value) === "true";
}


function numberInputValueOrNull(id) {
  const node = qs(id);
  if (!node) return null;
  const raw = String(node.value || "").trim();
  if (!raw) return null;
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
}

function auditValuePreview(value) {
  if (value == null || value === "") return "—";
  if (Array.isArray(value)) return value.length ? `${value.length} elem.` : "[]";
  if (typeof value === "object") {
    const keys = Object.keys(value);
    return keys.length ? `${keys.length} fields` : "{}";
  }
  return String(value);
}

function summarizeAuditDetails(details) {
  const entries = Object.entries(details || {}).filter(([, value]) => value != null && value !== "");
  if (!entries.length) return "Без деталей";
  return entries.slice(0, 3).map(([key, value]) => `${key}: ${auditValuePreview(value)}`).join(" · ");
}

function setFlash(message, tone = "info", timeoutMs = tone === "error" ? 0 : 4000) {
  const flash = qs("admin-flash");
  if (!flash) return;
  if (state.flashTimer) {
    clearTimeout(state.flashTimer);
    state.flashTimer = null;
  }
  if (!message) {
    flash.textContent = "";
    flash.dataset.tone = "";
    flash.dataset.visible = "false";
    return;
  }
  flash.textContent = message;
  flash.dataset.tone = tone;
  flash.dataset.visible = "true";
  if (timeoutMs > 0) {
    state.flashTimer = setTimeout(() => {
      flash.textContent = "";
      flash.dataset.tone = "";
      flash.dataset.visible = "false";
    }, timeoutMs);
  }
}

function setBusy(button, busy) {
  if (!button) return;
  button.disabled = busy;
  if (busy) button.setAttribute("aria-busy", "true");
  else button.removeAttribute("aria-busy");
}

function makePill(label, tone = "muted") {
  return `<span class="admin-pill" data-tone="${escapeHtml(tone)}">${escapeHtml(label)}</span>`;
}

function setSection(section) {
  state.activeSection = section;
  document.querySelectorAll(".admin-nav-btn").forEach((button) => {
    button.classList.toggle("admin-nav-btn--active", button.dataset.section === section);
  });
  document.querySelectorAll(".admin-section").forEach((node) => {
    node.classList.toggle("admin-section--active", node.dataset.section === section);
  });
  const label = qs("admin-active-section-label");
  if (label) label.textContent = SECTION_LABELS[section] || "Админка";
}

function touchUpdatedAt() {
  state.lastUpdatedAt = new Date().toISOString();
  const node = qs("admin-last-updated");
  if (node) node.textContent = formatDateTime(state.lastUpdatedAt);
}

function selectOptions(select, items, valueKey, labelKey, emptyLabel = "—") {
  if (!select) return;
  const current = select.value;
  select.innerHTML = "";
  const empty = document.createElement("option");
  empty.value = "";
  empty.textContent = emptyLabel;
  select.appendChild(empty);
  for (const item of items || []) {
    const option = document.createElement("option");
    option.value = item[valueKey] ?? "";
    option.textContent = item[labelKey] ?? item[valueKey] ?? "";
    select.appendChild(option);
  }
  if ([...select.options].some((option) => option.value === current)) {
    select.value = current;
  }
}

function renderSidebarUser() {
  const me = state.me || {};
  qs("admin-user-name").textContent = me.display_name || me.full_name || me.user || "—";
  const locationName = me.location && (me.location.crm_name || me.location.name);
  qs("admin-user-meta").textContent = [me.user, locationName, me.department_label, me.role].filter(Boolean).join(" · ") || "—";
}

function renderReferenceControls() {
  const reference = state.reference || {};
  selectOptions(qs("user-location"), reference.locations || [], "id", "crm_name", "—");
  selectOptions(qs("video-location"), reference.locations || [], "id", "crm_name", "—");
  selectOptions(qs("video-manager"), reference.managers || [], "id", "name", "—");
  selectOptions(qs("video-training-type"), reference.training_types || [], "slug", "name", "—");
  selectOptions(qs("active-checklist-select"), (reference.checklists || {}).files || [], "name", "display_name", "—");
  selectOptions(qs("new-checklist-copy"), (reference.checklists || {}).files || [], "name", "display_name", "Пустой шаблон");
  selectOptions(qs("training-type-checklist-admin"), (reference.checklists || {}).files || [], "name", "display_name", "—");
}

function currentRefreshSeconds() {
  const settings = state.settings || {};
  const runtime = settings.runtime || {};
  return Math.max(3, Number(runtime.admin_auto_refresh_seconds || 10));
}

function restartAutoRefresh() {
  if (state.autoRefreshTimer) {
    clearInterval(state.autoRefreshTimer);
    state.autoRefreshTimer = null;
  }
  state.autoRefreshTimer = setInterval(() => {
    if (document.hidden) return;
    if (state.activeSection === "overview") {
      void loadOverview().catch(() => {});
      return;
    }
    if (state.activeSection === "jobs") {
      void loadJobs().catch(() => {});
      return;
    }
    if (state.activeSection === "audit") {
      void loadAudit().catch(() => {});
      return;
    }
    if (state.activeSection === "users") {
      void loadUsers().catch(() => {});
    }
  }, currentRefreshSeconds() * 1000);
}

function renderOverview() {
  const data = state.overview;
  if (!data) return;
  const counts = data.counts || {};
  const videos = counts.videos || {};
  const jobs = counts.jobs || {};
  const users = counts.users || {};
  qs("overview-kpis").innerHTML = `
    <div class="admin-card"><div class="admin-kpi-label">Видео</div><div class="admin-kpi-value">${videos.total ?? 0}</div><div class="admin-kpi-sub">processing: ${videos.processing ?? 0}</div></div>
    <div class="admin-card"><div class="admin-kpi-label">Jobs</div><div class="admin-kpi-value">${jobs.total ?? 0}</div><div class="admin-kpi-sub">running: ${jobs.running ?? 0}, error: ${jobs.error ?? 0}</div></div>
    <div class="admin-card"><div class="admin-kpi-label">Пользователи</div><div class="admin-kpi-value">${users.total ?? 0}</div><div class="admin-kpi-sub">admin: ${users.admins ?? 0}, active: ${users.active ?? 0}</div></div>
    <div class="admin-card"><div class="admin-kpi-label">Чеклисты</div><div class="admin-kpi-value">${counts.checklists ?? 0}</div><div class="admin-kpi-sub">training types: ${counts.training_types ?? 0}</div></div>
    <div class="admin-card"><div class="admin-kpi-label">Локации</div><div class="admin-kpi-value">${counts.locations ?? 0}</div><div class="admin-kpi-sub">managers: ${counts.managers ?? 0}</div></div>
    <div class="admin-card"><div class="admin-kpi-label">Оценки</div><div class="admin-kpi-value">${data.analytics?.total_evaluations ?? 0}</div><div class="admin-kpi-sub">avg: ${data.analytics?.average_score ?? "—"}</div></div>
  `;

  const runtime = data.runtime || {};
  qs("overview-runtime").innerHTML = `
    <div class="admin-meta__row"><span class="admin-meta__key">Worker-ы</span><strong>${runtime.max_workers?.current ?? "—"} / настроено ${runtime.max_workers?.configured ?? "—"}</strong></div>
    <div class="admin-meta__row"><span class="admin-meta__key">Автообновление</span><strong>${runtime.admin_auto_refresh_seconds ?? "—"} сек</strong></div>
    <div class="admin-meta__row"><span class="admin-meta__key">Local registration</span><strong>${runtime.local_registration_enabled ? "вкл" : "выкл"}</strong></div>
    <div class="admin-meta__row"><span class="admin-meta__key">Собрано</span><strong>${formatDateTime(data.generated_at)}</strong></div>
  `;

  const analytics = data.analytics || {};
  qs("overview-analytics").innerHTML = `
    <div class="admin-meta__row"><span class="admin-meta__key">Средний балл</span><strong>${analytics.average_score ?? "—"}</strong></div>
    <div class="admin-meta__row"><span class="admin-meta__key">Топ менеджер</span><strong>${analytics.by_manager?.[0]?.name || "—"}</strong></div>
    <div class="admin-meta__row"><span class="admin-meta__key">Топ локация</span><strong>${analytics.by_location?.[0]?.name || "—"}</strong></div>
    <div class="admin-meta__row"><span class="admin-meta__key">Слабый критерий</span><strong>${analytics.by_criteria?.slice(-1)?.[0]?.name || "—"}</strong></div>
  `;

  const jobsBody = qs("overview-jobs-body");
  jobsBody.innerHTML = "";
  for (const item of data.recent_jobs || []) {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td class="admin-code">${escapeHtml(item.id || "—")}</td>
      <td class="admin-code">${escapeHtml(item.stem || "—")}</td>
      <td>${escapeHtml(item.kind || "—")}</td>
      <td>${makePill(item.status || "—", item.status === "error" ? "danger" : item.status === "running" ? "ok" : "muted")}</td>
      <td>${escapeHtml(item.stage || "—")}</td>
      <td>${escapeHtml(formatDateTime(item.updated_at || item.created_at))}</td>
    `;
    jobsBody.appendChild(row);
  }
  if (!jobsBody.children.length) jobsBody.innerHTML = '<tr><td colspan="6" class="admin-empty">Нет задач.</td></tr>';

  const auditBody = qs("overview-audit-body");
  auditBody.innerHTML = "";
  for (const item of data.recent_audit || []) {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${escapeHtml(formatDateTime(item.created_at))}</td>
      <td>${escapeHtml(item.actor || "system")}</td>
      <td class="admin-code">${escapeHtml(item.action || "—")}</td>
      <td>${escapeHtml([item.target_type, item.target_id].filter(Boolean).join(":"))}</td>
    `;
    auditBody.appendChild(row);
  }
  if (!auditBody.children.length) auditBody.innerHTML = '<tr><td colspan="4" class="admin-empty">Пока нет событий.</td></tr>';
}

function renderJobs() {
  const tbody = qs("jobs-body");
  const detail = qs("job-detail");
  const summary = qs("job-detail-summary");
  const cancelButton = qs("job-cancel-btn");
  const retryButton = qs("job-retry-btn");
  tbody.innerHTML = "";
  for (const item of state.jobs || []) {
    const row = document.createElement("tr");
    row.classList.toggle("is-selected", item.id === state.selectedJobId);
    row.dataset.jobId = item.id;
    row.innerHTML = `
      <td class="admin-code">${escapeHtml(item.id || "—")}</td>
      <td class="admin-code">${escapeHtml(item.stem || "—")}</td>
      <td>${escapeHtml(item.kind || "—")}</td>
      <td>${makePill(item.status || "—", item.status === "error" ? "danger" : item.status === "running" ? "ok" : item.status === "queued" ? "warn" : "muted")}</td>
      <td>${escapeHtml(item.stage || "—")}</td>
      <td>${escapeHtml(formatDateTime(item.updated_at || item.created_at))}</td>
    `;
    tbody.appendChild(row);
  }
  if (!tbody.children.length) tbody.innerHTML = '<tr><td colspan="6" class="admin-empty">Нет задач по выбранным фильтрам.</td></tr>';

  const selected = (state.jobs || []).find((item) => item.id === state.selectedJobId) || null;
  if (!selected) {
    if (summary) summary.textContent = "Выберите задачу слева, чтобы увидеть контекст и доступные действия.";
    if (cancelButton) cancelButton.disabled = true;
    if (retryButton) retryButton.disabled = true;
    detail.innerHTML = '<div class="admin-empty">Выберите задачу из списка.</div>';
    return;
  }
  if (summary) summary.textContent = `Статус: ${selected.status || "—"} · обновлено ${formatDateTime(selected.updated_at || selected.created_at)}`;
  if (cancelButton) cancelButton.disabled = !["queued", "running"].includes(String(selected.status || ""));
  if (retryButton) retryButton.disabled = ["queued", "running"].includes(String(selected.status || ""));
  detail.innerHTML = `
    <div class="admin-meta__row"><span class="admin-meta__key">ID</span><strong class="admin-code">${escapeHtml(selected.id)}</strong></div>
    <div class="admin-meta__row"><span class="admin-meta__key">Stem</span><strong class="admin-code">${escapeHtml(selected.stem || "—")}</strong></div>
    <div class="admin-meta__row"><span class="admin-meta__key">Тип</span><strong>${escapeHtml(selected.kind || "—")}</strong></div>
    <div class="admin-meta__row"><span class="admin-meta__key">Статус</span><strong>${makePill(selected.status || "—", selected.status === "error" ? "danger" : selected.status === "running" ? "ok" : selected.status === "queued" ? "warn" : "muted")}</strong></div>
    <div class="admin-meta__row"><span class="admin-meta__key">Stage</span><strong>${escapeHtml(selected.stage || "—")}</strong></div>
    <div class="admin-meta__row"><span class="admin-meta__key">Видео</span><strong>${escapeHtml(selected.video_file || selected.display_title || "—")}</strong></div>
    <div class="admin-meta__row"><span class="admin-meta__key">Пользователь</span><strong>${escapeHtml(selected.uploaded_by_name || selected.uploaded_by || "—")}</strong></div>
    <div class="admin-meta__row"><span class="admin-meta__key">Ошибка</span><strong>${escapeHtml(selected.error || "—")}</strong></div>
    <div class="admin-meta__row"><span class="admin-meta__key">Лог</span><strong class="admin-code">${escapeHtml((selected.stream_log || "").slice(-240) || "—")}</strong></div>
  `;
}

function renderUsers() {
  const tbody = qs("users-body");
  tbody.innerHTML = "";
  for (const item of state.users || []) {
    const row = document.createElement("tr");
    row.dataset.username = item.username;
    row.classList.toggle("is-selected", state.selectedUser === item.username);
    row.innerHTML = `
      <td class="admin-code">${escapeHtml(item.username)}</td>
      <td>${escapeHtml(item.full_name || item.display_name || "—")}</td>
      <td>${escapeHtml(item.auth_source || "—")}</td>
      <td>${escapeHtml(item.location_name || item.location_id || "—")}</td>
      <td>${escapeHtml(item.department_label || "—")}</td>
      <td>${makePill(item.effective_role || "user", item.is_admin ? "ok" : "muted")}</td>
      <td>${makePill(item.is_active ? "active" : "disabled", item.is_active ? "ok" : "danger")}</td>
    `;
    tbody.appendChild(row);
  }
  if (!tbody.children.length) tbody.innerHTML = '<tr><td colspan="7" class="admin-empty">Нет пользователей.</td></tr>';
}

function renderUserQuotaSummary(item) {
  const node = qs("user-quota-summary");
  if (!node) return;
  if (!item) {
    node.innerHTML = '<div class="admin-meta__row"><span class="admin-meta__key">Квоты</span><strong>Выберите пользователя, чтобы увидеть usage и effective limits.</strong></div>';
    return;
  }
  node.innerHTML = `
    <div class="admin-meta__row"><span class="admin-meta__key">Загружено сегодня</span><strong>${item.daily_uploaded_count ?? 0} / ${item.effective_daily_upload_limit ?? "—"}</strong></div>
    <div class="admin-meta__row"><span class="admin-meta__key">Осталось сегодня</span><strong>${item.daily_remaining ?? "—"}</strong></div>
    <div class="admin-meta__row"><span class="admin-meta__key">Очередь пользователя</span><strong>queued ${item.queued_count ?? 0} / ${item.effective_max_queued_jobs ?? "—"}</strong></div>
    <div class="admin-meta__row"><span class="admin-meta__key">Одновременная обработка</span><strong>running ${item.running_count ?? 0} / ${item.effective_max_running_jobs ?? "—"}</strong></div>
    <div class="admin-meta__row"><span class="admin-meta__key">Обновлено</span><strong>${formatDateTime(item.quotas_updated_at || item.updated_at)}</strong></div>
  `;
}

function resetUserForm() {
  state.selectedUser = null;
  qs("user-form-title").textContent = "Новый пользователь";
  qs("user-form-meta").textContent = "Создание локальной или AD-учётной записи и управление доступом.";
  qs("user-save-btn").textContent = "Создать пользователя";
  qs("user-username").disabled = false;
  qs("user-auth-source").disabled = false;
  qs("user-username").value = "";
  qs("user-auth-source").value = "local";
  qs("user-password").value = "";
  qs("user-full-name").value = "";
  qs("user-role").value = "user";
  qs("user-location").value = "";
  qs("user-department").value = "";
  qs("user-is-active").value = "true";
  qs("user-daily-upload-limit").value = "";
  qs("user-max-queued-jobs").value = "";
  qs("user-max-running-jobs").value = "";
  qs("user-activate-btn").disabled = true;
  qs("user-deactivate-btn").disabled = true;
  qs("user-reset-password-btn").disabled = true;
  renderUserQuotaSummary(null);
  renderUsers();
}

function fillUserForm(username) {
  const item = (state.users || []).find((row) => row.username === username);
  if (!item) return;
  state.selectedUser = username;
  qs("user-form-title").textContent = `Пользователь ${username}`;
  qs("user-form-meta").textContent = [
    item.auth_source ? `Источник: ${item.auth_source}` : null,
    item.location_name || item.location_id,
    item.department_label,
  ].filter(Boolean).join(" · ") || "Редактирование профиля и прав доступа.";
  qs("user-save-btn").textContent = "Сохранить изменения";
  qs("user-username").disabled = true;
  qs("user-auth-source").disabled = true;
  qs("user-username").value = item.username || "";
  qs("user-auth-source").value = item.auth_source || "local";
  qs("user-password").value = "";
  qs("user-full-name").value = item.full_name || "";
  qs("user-role").value = item.stored_role || "user";
  qs("user-location").value = item.location_id || "";
  qs("user-department").value = item.department || "";
  qs("user-is-active").value = String(Boolean(item.is_active));
  qs("user-daily-upload-limit").value = item.daily_upload_limit ?? "";
  qs("user-max-queued-jobs").value = item.max_queued_jobs ?? "";
  qs("user-max-running-jobs").value = item.max_running_jobs ?? "";
  qs("user-activate-btn").disabled = Boolean(item.is_active);
  qs("user-deactivate-btn").disabled = !Boolean(item.is_active);
  qs("user-reset-password-btn").disabled = item.auth_source !== "local";
  renderUserQuotaSummary(item);
  renderUsers();
}

function renderVideos() {
  const tbody = qs("videos-body");
  tbody.innerHTML = "";
  for (const item of state.videos || []) {
    const row = document.createElement("tr");
    row.dataset.stem = item.stem;
    row.classList.toggle("is-selected", state.selectedVideo === item.stem);
    const pipelineStatus = item.job?.status || item.status || "—";
    row.innerHTML = `
      <td class="admin-code">${escapeHtml(item.stem)}</td>
      <td>${escapeHtml(item.display_title || item.video_file || "—")}</td>
      <td>${escapeHtml(item.manager_name || item.manager_id || "—")}</td>
      <td>${escapeHtml(item.location_name || item.location_id || "—")}</td>
      <td>${makePill(pipelineStatus, pipelineStatus === "error" ? "danger" : pipelineStatus === "running" ? "ok" : pipelineStatus === "queued" ? "warn" : "muted")}</td>
      <td>${escapeHtml(item.checklist_slug_snapshot || "—")}</td>
    `;
    tbody.appendChild(row);
  }
  if (!tbody.children.length) tbody.innerHTML = '<tr><td colspan="6" class="admin-empty">Нет записей.</td></tr>';
}

function resetVideoForm() {
  state.selectedVideo = null;
  qs("video-form-title").textContent = "Карточка записи";
  qs("video-form-meta").textContent = "Выберите запись в таблице слева, чтобы отредактировать метаданные и действия обработки.";
  qs("video-form-stem").textContent = "—";
  qs("video-form-status").textContent = "Не выбрано";
  qs("video-form-status").dataset.tone = "muted";
  qs("video-display-title").value = "";
  qs("video-manager").value = "";
  qs("video-location").value = "";
  qs("video-training-type").value = "";
  qs("video-date").value = "";
  qs("video-tags").value = "";
  qs("video-save-btn").disabled = true;
  qs("video-resume-btn").disabled = true;
  qs("video-restart-btn").disabled = true;
  qs("video-re-eval-btn").disabled = true;
  qs("video-restore-btn").disabled = true;
  qs("video-delete-btn").disabled = true;
  renderVideos();
}

function fillVideoForm(stem) {
  const item = (state.videos || []).find((row) => row.stem === stem);
  if (!item) return;
  state.selectedVideo = stem;
  const pipelineStatus = item.job?.status || item.status || "—";
  qs("video-form-title").textContent = item.display_title || item.video_file || stem;
  qs("video-form-meta").textContent = [
    item.manager_name || item.manager_id,
    item.location_name || item.location_id,
    item.training_type_name || item.training_type_slug,
  ].filter(Boolean).join(" · ") || "Редактирование карточки записи.";
  qs("video-form-stem").textContent = stem;
  qs("video-form-status").textContent = pipelineStatus;
  qs("video-form-status").dataset.tone =
    pipelineStatus === "error" ? "danger" : pipelineStatus === "running" ? "ok" : pipelineStatus === "queued" ? "warn" : "muted";
  qs("video-display-title").value = item.display_title || "";
  qs("video-manager").value = item.manager_id || "";
  qs("video-location").value = item.location_id || "";
  qs("video-training-type").value = item.training_type_slug || "";
  qs("video-date").value = item.interaction_date || "";
  qs("video-tags").value = (item.tags || []).join(", ");
  qs("video-save-btn").disabled = false;
  qs("video-resume-btn").disabled = false;
  qs("video-restart-btn").disabled = false;
  qs("video-re-eval-btn").disabled = false;
  qs("video-restore-btn").disabled = !Boolean(item.delete_requested_at || item.deleted_at || item.status === "deleted");
  qs("video-delete-btn").disabled = false;
  renderVideos();
}

function renderChecklists() {
  const files = (state.reference?.checklists || {}).files || [];
  const active = (state.reference?.checklists || {}).active || "";
  const tbody = qs("checklists-body");
  tbody.innerHTML = "";
  for (const item of files) {
    const row = document.createElement("tr");
    row.dataset.checklist = item.name;
    row.classList.toggle("is-selected", item.name === state.selectedChecklist);
    row.innerHTML = `
      <td class="admin-code">${escapeHtml(item.name)}</td>
      <td>${escapeHtml(item.display_name || item.name)}</td>
      <td>${escapeHtml(departmentLabel(item.department))}</td>
      <td>${escapeHtml(item.version || "1")}${item.name === active ? " · active" : ""}</td>
    `;
    tbody.appendChild(row);
  }
  if (!tbody.children.length) tbody.innerHTML = '<tr><td colspan="4" class="admin-empty">Нет чеклистов.</td></tr>';
  qs("active-checklist-select").value = active;
}

function clearChecklistEditor() {
  state.selectedChecklist = null;
  state.checklistContent = null;
  qs("checklist-editor-title").textContent = "Редактор чеклиста";
  qs("checklist-editor-display-name").value = "";
  qs("checklist-editor-version").value = "1";
  qs("checklist-editor-department").value = "";
  qs("checklist-criteria-list").innerHTML = "";
  renderChecklists();
}

function appendChecklistCriterionRow(data = {}) {
  const list = qs("checklist-criteria-list");
  const row = document.createElement("div");
  row.className = "admin-criteria-row";
  row.innerHTML = `
    <div class="admin-criteria-row-head">
      <input class="admin-input admin-code" data-field="id" type="text" placeholder="criterion_id" value="${escapeHtml(data.id || "")}" />
      <input class="admin-input" data-field="name" type="text" placeholder="Название" value="${escapeHtml(data.name || "")}" />
      <input class="admin-input" data-field="weight" type="number" min="1" step="1" value="${escapeHtml(data.weight != null ? data.weight : 1)}" />
      <button type="button" class="admin-inline-btn admin-inline-btn--danger admin-criteria-remove">×</button>
    </div>
    <textarea class="admin-textarea" data-field="description" placeholder="Описание для модели">${escapeHtml(data.description || "")}</textarea>
  `;
  row.querySelector(".admin-criteria-remove").addEventListener("click", () => row.remove());
  list.appendChild(row);
}

function fillChecklistEditor(content) {
  state.selectedChecklist = content.filename;
  state.checklistContent = content;
  qs("checklist-editor-title").textContent = content.display_name || content.filename;
  qs("checklist-editor-display-name").value = content.display_name || "";
  qs("checklist-editor-version").value = content.version || "1";
  qs("checklist-editor-department").value = content.department || "";
  qs("checklist-criteria-list").innerHTML = "";
  for (const row of content.criteria || []) appendChecklistCriterionRow(row);
  renderChecklists();
}

function renderManagers() {
  const tbody = qs("managers-body");
  tbody.innerHTML = "";
  for (const item of state.reference?.managers || []) {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td class="admin-code">${escapeHtml(item.id)}</td>
      <td>${escapeHtml(item.name)}</td>
      <td><button type="button" class="admin-inline-btn admin-inline-btn--danger" data-manager-delete="${escapeHtml(item.id)}">Удалить</button></td>
    `;
    tbody.appendChild(row);
  }
  if (!tbody.children.length) tbody.innerHTML = '<tr><td colspan="3" class="admin-empty">Нет менеджеров.</td></tr>';
}

function renderLocations() {
  const tbody = qs("locations-body");
  tbody.innerHTML = "";
  for (const item of state.reference?.locations || []) {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td class="admin-code">${escapeHtml(item.id)}</td>
      <td>${escapeHtml(item.crm_name || item.name)}</td>
      <td><button type="button" class="admin-inline-btn admin-inline-btn--danger" data-location-delete="${escapeHtml(item.id)}">Удалить</button></td>
    `;
    tbody.appendChild(row);
  }
  if (!tbody.children.length) tbody.innerHTML = '<tr><td colspan="3" class="admin-empty">Нет локаций.</td></tr>';
}

function renderTrainingTypes() {
  const tbody = qs("training-types-body");
  tbody.innerHTML = "";
  for (const item of state.reference?.training_types || []) {
    const row = document.createElement("tr");
    row.dataset.trainingSlug = item.slug;
    row.classList.toggle("is-selected", state.selectedTrainingType === item.slug);
    row.innerHTML = `
      <td class="admin-code">${escapeHtml(item.slug)}</td>
      <td>${escapeHtml(item.name)}</td>
      <td>${escapeHtml(item.checklist_display_name || item.checklist_slug || "—")}</td>
    `;
    tbody.appendChild(row);
  }
  if (!tbody.children.length) tbody.innerHTML = '<tr><td colspan="3" class="admin-empty">Нет типов тренировки.</td></tr>';
}

function resetTrainingTypeForm() {
  state.selectedTrainingType = null;
  qs("training-type-slug").disabled = false;
  qs("training-type-slug").value = "";
  qs("training-type-name-admin").value = "";
  qs("training-type-department-admin").value = "";
  qs("training-type-checklist-admin").value = "";
  qs("training-type-sort-order-admin").value = "";
  renderTrainingTypes();
}

function fillTrainingTypeForm(slug) {
  const item = (state.reference?.training_types || []).find((row) => row.slug === slug);
  if (!item) return;
  state.selectedTrainingType = slug;
  qs("training-type-slug").disabled = true;
  qs("training-type-slug").value = item.slug || "";
  qs("training-type-name-admin").value = item.name || "";
  qs("training-type-department-admin").value = item.department || "";
  qs("training-type-checklist-admin").value = item.checklist_slug || "";
  qs("training-type-sort-order-admin").value = item.sort_order ?? "";
  renderTrainingTypes();
}

function renderSettings() {
  const payload = state.settings || {};
  const runtime = payload.runtime || {};
  qs("settings-max-workers").value = runtime.max_workers?.configured ?? runtime.max_workers?.current ?? 2;
  qs("settings-admin-refresh").value = runtime.admin_auto_refresh_seconds ?? 10;
  qs("settings-local-registration").value = String(Boolean(runtime.local_registration_enabled));
  qs("settings-max-queue-depth").value = runtime.max_queue_depth ?? 100;
  qs("settings-default-daily-upload-limit").value = runtime.default_daily_upload_limit ?? 20;
  qs("settings-default-max-queued-jobs").value = runtime.default_max_queued_jobs ?? 5;
  qs("settings-default-max-running-jobs").value = runtime.default_max_running_jobs ?? 1;
  qs("settings-max-workers-help").textContent = `Сейчас активно ${runtime.max_workers?.current ?? "—"} worker-ов; настроено ${runtime.max_workers?.configured ?? "—"}.`;
  qs("settings-runtime").innerHTML = `
    <div class="admin-meta__row"><span class="admin-meta__key">Текущие worker-ы</span><strong>${runtime.max_workers?.current ?? "—"}</strong></div>
    <div class="admin-meta__row"><span class="admin-meta__key">Настроенные worker-ы</span><strong>${runtime.max_workers?.configured ?? "—"}</strong></div>
    <div class="admin-meta__row"><span class="admin-meta__key">Применение</span><strong>${runtime.max_workers?.applied ? "да" : "ожидает применения"}</strong></div>
    <div class="admin-meta__row"><span class="admin-meta__key">Автообновление админки</span><strong>${runtime.admin_auto_refresh_seconds ?? "—"} сек</strong></div>
    <div class="admin-meta__row"><span class="admin-meta__key">Local registration</span><strong>${runtime.local_registration_enabled ? "вкл" : "выкл"}</strong></div>
    <div class="admin-meta__row"><span class="admin-meta__key">Глобальная очередь</span><strong>${runtime.max_queue_depth ?? "—"} задач</strong></div>
    <div class="admin-meta__row"><span class="admin-meta__key">Дневная квота по умолчанию</span><strong>${runtime.default_daily_upload_limit ?? "—"} видео</strong></div>
    <div class="admin-meta__row"><span class="admin-meta__key">Queued jobs default</span><strong>${runtime.default_max_queued_jobs ?? "—"}</strong></div>
    <div class="admin-meta__row"><span class="admin-meta__key">Running jobs default</span><strong>${runtime.default_max_running_jobs ?? "—"}</strong></div>
  `;
}

function renderAudit() {
  const tbody = qs("audit-body");
  tbody.innerHTML = "";
  for (const item of state.audit || []) {
    const details = item.details || {};
    const summary = summarizeAuditDetails(details);
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${escapeHtml(formatDateTime(item.created_at))}</td>
      <td>${escapeHtml(item.actor || "system")}</td>
      <td class="admin-code">${escapeHtml(item.action || "—")}</td>
      <td>${escapeHtml([item.target_type, item.target_id].filter(Boolean).join(":"))}</td>
      <td>${makePill(item.status || "ok", item.status === "ok" ? "ok" : "danger")}</td>
      <td>
        <details class="admin-audit-details">
          <summary>${escapeHtml(summary)}</summary>
          <pre class="admin-audit-json admin-code">${escapeHtml(JSON.stringify(details, null, 2))}</pre>
        </details>
      </td>
    `;
    tbody.appendChild(row);
  }
  if (!tbody.children.length) tbody.innerHTML = '<tr><td colspan="6" class="admin-empty">Аудит пуст.</td></tr>';
}

async function loadBootstrap() {
  const data = await apiJson(`${API}/api/admin/bootstrap`);
  state.me = data.me;
  state.reference = data.reference;
  state.settings = data.settings;
  renderSidebarUser();
  renderReferenceControls();
  renderChecklists();
  renderManagers();
  renderLocations();
  renderTrainingTypes();
  renderSettings();
  restartAutoRefresh();
  touchUpdatedAt();
}

async function refreshReferenceData() {
  state.reference = await apiJson(`${API}/api/admin/reference-data`);
  renderReferenceControls();
  renderChecklists();
  renderManagers();
  renderLocations();
  renderTrainingTypes();
  touchUpdatedAt();
}

async function loadOverview() {
  state.overview = await apiJson(`${API}/api/admin/overview`);
  renderOverview();
  touchUpdatedAt();
}

async function loadJobs() {
  const params = new URLSearchParams();
  if (qs("jobs-status-filter").value) params.set("status", qs("jobs-status-filter").value);
  if (qs("jobs-kind-filter").value) params.set("kind", qs("jobs-kind-filter").value);
  if (qs("jobs-query").value.trim()) params.set("q", qs("jobs-query").value.trim());
  params.set("limit", "250");
  const data = await apiJson(`${API}/api/admin/jobs?${params.toString()}`);
  state.jobs = data.items || [];
  if (state.selectedJobId && !state.jobs.some((item) => item.id === state.selectedJobId)) {
    state.selectedJobId = null;
  }
  renderJobs();
  touchUpdatedAt();
}

async function loadUsers() {
  const data = await apiJson(`${API}/api/admin/users`);
  state.users = data.items || [];
  if (state.selectedUser && !state.users.some((item) => item.username === state.selectedUser)) {
    resetUserForm();
  } else {
    if (state.selectedUser) fillUserForm(state.selectedUser);
    else renderUsers();
  }
  touchUpdatedAt();
}

async function loadVideos() {
  const params = new URLSearchParams();
  if (qs("videos-query").value.trim()) params.set("q", qs("videos-query").value.trim());
  if (qs("videos-status-filter").value) params.set("status", qs("videos-status-filter").value);
  const data = await apiJson(`${API}/api/admin/videos?${params.toString()}`);
  state.videos = data.items || [];
  if (state.selectedVideo && !state.videos.some((item) => item.stem === state.selectedVideo)) {
    resetVideoForm();
  } else {
    renderVideos();
  }
  touchUpdatedAt();
}

async function loadChecklists() {
  await refreshReferenceData();
  if (state.selectedChecklist) {
    try {
      const content = await apiJson(`${API}/api/admin/checklists/${encodeURIComponent(state.selectedChecklist)}`);
      fillChecklistEditor(content);
    } catch (_) {
      clearChecklistEditor();
    }
  }
  touchUpdatedAt();
}

async function loadSettings() {
  state.settings = await apiJson(`${API}/api/admin/settings`);
  renderSettings();
  restartAutoRefresh();
  touchUpdatedAt();
}

async function loadAudit() {
  const data = await apiJson(`${API}/api/admin/audit?limit=200`);
  state.audit = data.items || [];
  renderAudit();
  touchUpdatedAt();
}

async function loadAll() {
  await loadBootstrap();
  await Promise.all([loadOverview(), loadJobs(), loadUsers(), loadVideos(), loadChecklists(), loadAudit()]);
}

async function handleRefreshAll() {
  const button = qs("admin-refresh-all");
  setBusy(button, true);
  try {
    await loadAll();
    setFlash("Данные синхронизированы.", "info", 2200);
  } catch (error) {
    setFlash(error.message || String(error), "error", 0);
  } finally {
    setBusy(button, false);
  }
}

async function handleUserSubmit(event) {
  event.preventDefault();
  const button = qs("user-save-btn");
  setBusy(button, true);
  try {
    const body = {
      username: qs("user-username").value.trim(),
      auth_source: qs("user-auth-source").value,
      password: qs("user-password").value,
      full_name: qs("user-full-name").value.trim(),
      location_id: qs("user-location").value,
      department: qs("user-department").value,
      role: qs("user-role").value,
      daily_upload_limit: numberInputValueOrNull("user-daily-upload-limit"),
      max_queued_jobs: numberInputValueOrNull("user-max-queued-jobs"),
      max_running_jobs: numberInputValueOrNull("user-max-running-jobs"),
      is_active: boolValue(qs("user-is-active").value),
    };
    if (state.selectedUser) {
      await apiJson(`${API}/api/admin/users/${encodeURIComponent(state.selectedUser)}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          full_name: body.full_name,
          location_id: body.location_id,
          department: body.department,
          role: body.role,
          daily_upload_limit: body.daily_upload_limit,
          max_queued_jobs: body.max_queued_jobs,
          max_running_jobs: body.max_running_jobs,
          is_active: body.is_active,
        }),
      });
      setFlash("Пользователь обновлён.", "ok");
    } else {
      await apiJson(`${API}/api/admin/users`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      setFlash("Пользователь создан.", "ok");
      resetUserForm();
    }
    await Promise.all([loadUsers(), refreshReferenceData(), loadOverview()]);
  } catch (error) {
    setFlash(error.message || String(error), "error", 0);
  } finally {
    setBusy(button, false);
  }
}

async function handleUserLifecycle(action) {
  if (!state.selectedUser) {
    setFlash("Сначала выберите пользователя.", "error", 0);
    return;
  }
  try {
    if (action === "reset-password") {
      const newPassword = qs("user-password").value;
      if (!newPassword) {
        setFlash("Введите новый пароль в поле пароля.", "error", 0);
        return;
      }
      await apiJson(`${API}/api/admin/users/${encodeURIComponent(state.selectedUser)}/reset-password`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ new_password: newPassword }),
      });
      qs("user-password").value = "";
      setFlash("Пароль обновлён.", "ok");
    } else {
      const path = action === "activate" ? "activate" : "deactivate";
      await apiJson(`${API}/api/admin/users/${encodeURIComponent(state.selectedUser)}/${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
      });
      setFlash(action === "activate" ? "Пользователь активирован." : "Пользователь отключён.", "ok");
    }
    await Promise.all([loadUsers(), loadOverview()]);
  } catch (error) {
    setFlash(error.message || String(error), "error", 0);
  }
}

async function handleVideoSubmit(event) {
  event.preventDefault();
  if (!state.selectedVideo) {
    setFlash("Сначала выберите запись.", "error", 0);
    return;
  }
  const button = qs("video-save-btn");
  setBusy(button, true);
  try {
    const managerSelect = qs("video-manager");
    const locationSelect = qs("video-location");
    const payload = {
      display_title: qs("video-display-title").value.trim() || null,
      manager_id: managerSelect.value || null,
      manager_name: managerSelect.value ? managerSelect.selectedOptions[0]?.textContent || null : null,
      location_id: locationSelect.value || null,
      location_name: locationSelect.value ? locationSelect.selectedOptions[0]?.textContent || null : null,
      interaction_date: qs("video-date").value || null,
      training_type_slug: qs("video-training-type").value || null,
      tags: qs("video-tags")
        .value.split(",")
        .map((item) => item.trim())
        .filter(Boolean),
    };
    await apiJson(`${API}/api/admin/videos/${encodeURIComponent(state.selectedVideo)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    setFlash("Карточка записи обновлена.", "ok");
    await Promise.all([loadVideos(), loadOverview()]);
  } catch (error) {
    setFlash(error.message || String(error), "error", 0);
  } finally {
    setBusy(button, false);
  }
}

async function handleVideoAction(stem, action) {
  if (!stem) {
    setFlash("Сначала выберите запись.", "error", 0);
    return;
  }
  try {
    if (action === "select") {
      fillVideoForm(stem);
      return;
    }
    if (action === "delete") {
      await apiJson(`${API}/api/admin/videos/${encodeURIComponent(stem)}`, { method: "DELETE" });
    } else if (action === "restore") {
      await apiJson(`${API}/api/admin/videos/${encodeURIComponent(stem)}/restore`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
      });
    } else {
      const mode = action === "resume" ? "resume" : action === "restart" ? "restart" : "re-evaluate";
      await apiJson(`${API}/api/admin/videos/${encodeURIComponent(stem)}/reprocess`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode }),
      });
    }
    setFlash("Операция выполнена.", "ok");
    await Promise.all([loadVideos(), loadJobs(), loadOverview(), loadAudit()]);
  } catch (error) {
    setFlash(error.message || String(error), "error", 0);
  }
}

async function handleChecklistCreate() {
  const filename = qs("new-checklist-name").value.trim();
  const copyFrom = qs("new-checklist-copy").value || null;
  if (!filename) {
    setFlash("Укажите slug нового чеклиста.", "error", 0);
    return;
  }
  try {
    const data = await apiJson(`${API}/api/admin/checklists`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename, copy_from: copyFrom }),
    });
    qs("new-checklist-name").value = "";
    setFlash("Чеклист создан.", "ok");
    await loadChecklists();
    const content = await apiJson(`${API}/api/admin/checklists/${encodeURIComponent(data.filename)}`);
    fillChecklistEditor(content);
    await Promise.all([loadOverview(), loadAudit()]);
  } catch (error) {
    setFlash(error.message || String(error), "error", 0);
  }
}

async function handleChecklistSave(event) {
  event.preventDefault();
  if (!state.selectedChecklist) {
    setFlash("Сначала выберите чеклист.", "error", 0);
    return;
  }
  const criteria = [...qs("checklist-criteria-list").querySelectorAll(".admin-criteria-row")].map((row) => ({
    id: row.querySelector('[data-field="id"]').value.trim(),
    name: row.querySelector('[data-field="name"]').value.trim(),
    description: row.querySelector('[data-field="description"]').value.trim(),
    weight: Number(row.querySelector('[data-field="weight"]').value || 1),
  }));
  try {
    await apiJson(`${API}/api/admin/checklists/${encodeURIComponent(state.selectedChecklist)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        display_name: qs("checklist-editor-display-name").value.trim(),
        version: qs("checklist-editor-version").value.trim() || "1",
        department: qs("checklist-editor-department").value,
        criteria,
      }),
    });
    setFlash("Чеклист сохранён.", "ok");
    await loadChecklists();
    const content = await apiJson(`${API}/api/admin/checklists/${encodeURIComponent(state.selectedChecklist)}`);
    fillChecklistEditor(content);
    await Promise.all([loadOverview(), loadAudit()]);
  } catch (error) {
    setFlash(error.message || String(error), "error", 0);
  }
}

async function handleChecklistDelete() {
  if (!state.selectedChecklist) {
    setFlash("Сначала выберите чеклист.", "error", 0);
    return;
  }
  try {
    await apiJson(`${API}/api/admin/checklists/${encodeURIComponent(state.selectedChecklist)}`, {
      method: "DELETE",
    });
    setFlash("Чеклист удалён.", "ok");
    clearChecklistEditor();
    await Promise.all([loadChecklists(), loadOverview(), loadAudit()]);
  } catch (error) {
    setFlash(error.message || String(error), "error", 0);
  }
}

async function handleSetActiveChecklist() {
  const selected = qs("active-checklist-select").value;
  if (!selected) {
    setFlash("Выберите активный чеклист.", "error", 0);
    return;
  }
  try {
    await apiJson(`${API}/api/admin/checklists/active`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ file: selected }),
    });
    setFlash("Активный чеклист обновлён.", "ok");
    await Promise.all([loadChecklists(), loadOverview(), loadAudit()]);
  } catch (error) {
    setFlash(error.message || String(error), "error", 0);
  }
}

async function handleManagerSubmit(event) {
  event.preventDefault();
  try {
    await apiJson(`${API}/api/managers`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id: qs("manager-id").value.trim(),
        name: qs("manager-name").value.trim(),
      }),
    });
    qs("manager-id").value = "";
    qs("manager-name").value = "";
    setFlash("Менеджер сохранён.", "ok");
    await Promise.all([refreshReferenceData(), loadAudit(), loadOverview()]);
  } catch (error) {
    setFlash(error.message || String(error), "error", 0);
  }
}

async function handleLocationSubmit(event) {
  event.preventDefault();
  try {
    await apiJson(`${API}/api/locations`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id: qs("location-id").value.trim(),
        name: qs("location-name").value.trim(),
        crm_id: qs("location-crm-id").value.trim() || null,
      }),
    });
    qs("location-id").value = "";
    qs("location-name").value = "";
    qs("location-crm-id").value = "";
    setFlash("Локация сохранена.", "ok");
    await Promise.all([refreshReferenceData(), loadUsers(), loadVideos(), loadAudit(), loadOverview()]);
  } catch (error) {
    setFlash(error.message || String(error), "error", 0);
  }
}

async function handleTrainingTypeSubmit(event) {
  event.preventDefault();
  const slug = qs("training-type-slug").value.trim();
  if (!slug) {
    setFlash("Укажите slug типа тренировки.", "error", 0);
    return;
  }
  try {
    const body = {
      slug,
      name: qs("training-type-name-admin").value.trim(),
      department: qs("training-type-department-admin").value || null,
      checklist_slug: qs("training-type-checklist-admin").value || null,
      sort_order: Number(qs("training-type-sort-order-admin").value || 0),
    };
    const url = state.selectedTrainingType
      ? `${API}/api/training-types/${encodeURIComponent(state.selectedTrainingType)}`
      : `${API}/api/training-types`;
    await apiJson(url, {
      method: state.selectedTrainingType ? "PUT" : "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    setFlash("Тип тренировки сохранён.", "ok");
    await Promise.all([refreshReferenceData(), loadAudit(), loadOverview()]);
    if (!state.selectedTrainingType) resetTrainingTypeForm();
  } catch (error) {
    setFlash(error.message || String(error), "error", 0);
  }
}

async function handleTrainingTypeDelete() {
  if (!state.selectedTrainingType) {
    setFlash("Сначала выберите тип тренировки.", "error", 0);
    return;
  }
  try {
    await apiJson(`${API}/api/training-types/${encodeURIComponent(state.selectedTrainingType)}`, {
      method: "DELETE",
    });
    setFlash("Тип тренировки удалён.", "ok");
    resetTrainingTypeForm();
    await Promise.all([refreshReferenceData(), loadAudit(), loadOverview()]);
  } catch (error) {
    setFlash(error.message || String(error), "error", 0);
  }
}

async function handleSettingsSubmit(event) {
  event.preventDefault();
  try {
    const payload = await apiJson(`${API}/api/admin/settings`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        max_workers: Number(qs("settings-max-workers").value || 2),
        admin_auto_refresh_seconds: Number(qs("settings-admin-refresh").value || 10),
        local_registration_enabled: boolValue(qs("settings-local-registration").value),
        max_queue_depth: Number(qs("settings-max-queue-depth").value || 100),
        default_daily_upload_limit: Number(qs("settings-default-daily-upload-limit").value || 20),
        default_max_queued_jobs: Number(qs("settings-default-max-queued-jobs").value || 5),
        default_max_running_jobs: Number(qs("settings-default-max-running-jobs").value || 1),
      }),
    });
    state.settings = payload;
    renderSettings();
    restartAutoRefresh();
    const note = payload.runtime_apply?.applied === false
      ? "Настройка сохранена, но применение worker-ов отложено до освобождения очереди или рестарта."
      : "Настройки сохранены.";
    setFlash(note, payload.runtime_apply?.applied === false ? "info" : "ok");
    await Promise.all([loadOverview(), loadAudit()]);
  } catch (error) {
    setFlash(error.message || String(error), "error", 0);
  }
}

async function handleJobAction(jobId, action) {
  if (!jobId) {
    setFlash("Не удалось определить задачу.", "error", 0);
    return;
  }
  try {
    if (action === "select") {
      state.selectedJobId = jobId;
      renderJobs();
      return;
    }
    if (action === "cancel") {
      await apiJson(`${API}/api/admin/jobs/${encodeURIComponent(jobId)}/cancel`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
      });
    } else {
      await apiJson(`${API}/api/admin/jobs/${encodeURIComponent(jobId)}/retry`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: "resume" }),
      });
    }
    setFlash("Операция по задаче выполнена.", "ok");
    await Promise.all([loadJobs(), loadOverview(), loadVideos(), loadAudit()]);
  } catch (error) {
    setFlash(error.message || String(error), "error", 0);
  }
}

function bindEvents() {
  document.querySelectorAll(".admin-nav-btn").forEach((button) => {
    button.addEventListener("click", () => setSection(button.dataset.section));
  });

  qs("admin-refresh-all").addEventListener("click", () => void handleRefreshAll());

  qs("jobs-status-filter").addEventListener("change", () => void loadJobs());
  qs("jobs-kind-filter").addEventListener("change", () => void loadJobs());
  qs("jobs-query").addEventListener("input", () => void loadJobs());
  qs("videos-query").addEventListener("input", () => void loadVideos());
  qs("videos-status-filter").addEventListener("change", () => void loadVideos());
  qs("job-cancel-btn").addEventListener("click", () => void handleJobAction(state.selectedJobId, "cancel"));
  qs("job-retry-btn").addEventListener("click", () => void handleJobAction(state.selectedJobId, "retry"));

  qs("user-form").addEventListener("submit", handleUserSubmit);
  qs("user-form-reset").addEventListener("click", resetUserForm);
  qs("user-activate-btn").addEventListener("click", () => void handleUserLifecycle("activate"));
  qs("user-deactivate-btn").addEventListener("click", () => void handleUserLifecycle("deactivate"));
  qs("user-reset-password-btn").addEventListener("click", () => void handleUserLifecycle("reset-password"));

  qs("video-form").addEventListener("submit", handleVideoSubmit);
  qs("video-resume-btn").addEventListener("click", () => void handleVideoAction(state.selectedVideo, "resume"));
  qs("video-restart-btn").addEventListener("click", () => void handleVideoAction(state.selectedVideo, "restart"));
  qs("video-re-eval-btn").addEventListener("click", () => void handleVideoAction(state.selectedVideo, "re-evaluate"));
  qs("video-restore-btn").addEventListener("click", () => void handleVideoAction(state.selectedVideo, "restore"));
  qs("video-delete-btn").addEventListener("click", () => void handleVideoAction(state.selectedVideo, "delete"));

  qs("create-checklist-btn").addEventListener("click", () => void handleChecklistCreate());
  qs("active-checklist-save").addEventListener("click", () => void handleSetActiveChecklist());
  qs("checklist-editor-form").addEventListener("submit", handleChecklistSave);
  qs("checklist-delete-btn").addEventListener("click", () => void handleChecklistDelete());
  qs("checklist-add-row").addEventListener("click", () => appendChecklistCriterionRow());

  qs("manager-form").addEventListener("submit", handleManagerSubmit);
  qs("location-form").addEventListener("submit", handleLocationSubmit);
  qs("training-type-form").addEventListener("submit", handleTrainingTypeSubmit);
  qs("training-form-reset").addEventListener("click", resetTrainingTypeForm);
  qs("training-type-delete-btn").addEventListener("click", () => void handleTrainingTypeDelete());

  qs("settings-form").addEventListener("submit", handleSettingsSubmit);

  qs("admin-logout").addEventListener("click", async () => {
    await apiFetch(`${API}/api/auth/logout`, { method: "POST" });
    window.location.href = "/login.html";
  });

  document.addEventListener("click", (event) => {
    const target = event.target;
    const userRow = target.closest && target.closest("#users-body tr[data-username]");
    if (userRow) {
      fillUserForm(userRow.dataset.username);
      return;
    }

    const videoAction = target.closest && target.closest("[data-video-action]");
    if (videoAction) {
      const stem = videoAction.dataset.stem || state.selectedVideo;
      if (stem) void handleVideoAction(stem, videoAction.dataset.videoAction);
      return;
    }
    const videoRow = target.closest && target.closest("#videos-body tr[data-stem]");
    if (videoRow) {
      fillVideoForm(videoRow.dataset.stem);
      return;
    }

    const jobAction = target.closest && target.closest("[data-job-action]");
    if (jobAction) {
      const jobId = jobAction.dataset.jobId;
      if (jobId) void handleJobAction(jobId, jobAction.dataset.jobAction);
      return;
    }
    const jobRow = target.closest && target.closest("#jobs-body tr[data-job-id]");
    if (jobRow) {
      state.selectedJobId = jobRow.dataset.jobId;
      renderJobs();
      return;
    }

    const checklistRow = target.closest && target.closest("#checklists-body tr[data-checklist]");
    if (checklistRow) {
      void apiJson(`${API}/api/admin/checklists/${encodeURIComponent(checklistRow.dataset.checklist)}`)
        .then((content) => fillChecklistEditor(content))
        .catch((error) => setFlash(error.message || String(error), "error", 0));
      return;
    }

    const managerDelete = target.closest && target.closest("[data-manager-delete]");
    if (managerDelete) {
      void apiJson(`${API}/api/managers/${encodeURIComponent(managerDelete.dataset.managerDelete)}`, { method: "DELETE" })
        .then(async () => {
          setFlash("Менеджер удалён.", "ok");
          await Promise.all([refreshReferenceData(), loadAudit(), loadOverview()]);
        })
        .catch((error) => setFlash(error.message || String(error), "error", 0));
      return;
    }

    const locationDelete = target.closest && target.closest("[data-location-delete]");
    if (locationDelete) {
      void apiJson(`${API}/api/locations/${encodeURIComponent(locationDelete.dataset.locationDelete)}`, { method: "DELETE" })
        .then(async () => {
          setFlash("Локация удалена.", "ok");
          await Promise.all([refreshReferenceData(), loadAudit(), loadOverview()]);
        })
        .catch((error) => setFlash(error.message || String(error), "error", 0));
      return;
    }

    const trainingRow = target.closest && target.closest("#training-types-body tr[data-training-slug]");
    if (trainingRow) {
      fillTrainingTypeForm(trainingRow.dataset.trainingSlug);
    }
  });
}

async function init() {
  bindEvents();
  setSection(state.activeSection);
  try {
    await loadAll();
    resetUserForm();
    resetVideoForm();
    clearChecklistEditor();
    resetTrainingTypeForm();
  } catch (error) {
    if (String(error.message || "").includes("Требуется вход")) {
      window.location.href = "/login.html?next=/admin";
      return;
    }
    setFlash(error.message || String(error), "error", 0);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  void init();
});
