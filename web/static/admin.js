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
  glossary: "Глоссарий",
  feedback: "Обратная связь",
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
  glossary: [],
  glossaryCategories: [],
  feedback: [],
  feedbackTypes: [],
  feedbackStatuses: [],
  feedbackCounts: {},
  audit: [],
  apiKeys: [],
  settings: null,
  selectedJobId: null,
  selectedUser: null,
  selectedVideo: null,
  selectedChecklist: null,
  selectedTrainingType: null,
  selectedGlossaryId: null,
  selectedFeedbackId: null,
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

function corporateEmailForUser(item) {
  const explicit = String((item && item.corporate_email) || "").trim();
  if (explicit) return explicit;
  const username = String((item && item.username) || "").trim();
  return username && !username.includes("@") ? `${username}@freshauto.ru` : "";
}

function userAccessStatus(item) {
  const raw = String((item && item.access_status) || "").trim().toLowerCase();
  if (raw) return raw;
  if (item && item.is_pending_approval) return "pending";
  return item && item.is_active ? "active" : "inactive";
}

function userStatusPill(item) {
  const status = userAccessStatus(item);
  if (status === "pending") return makePill("на подтверждении", "warn");
  if (status === "rejected") return makePill("отказано", "danger");
  if (status === "active") return makePill("active", "ok");
  return makePill("disabled", "danger");
}

function feedbackTypeLabel(value) {
  const item = (state.feedbackTypes || []).find((row) => row.value === value);
  return item ? item.label : value || "—";
}

function feedbackRatingLabel(item) {
  const rating = Number((item && item.rating) || 0);
  return rating > 0 ? "★".repeat(rating) : "—";
}

function feedbackStatusLabel(item) {
  if (item && item.status_label) return item.status_label;
  const value = String((item && item.status) || "");
  const found = (state.feedbackStatuses || []).find((row) => row.value === value);
  return found ? found.label : value || "—";
}

function feedbackStatusTone(status) {
  if (status === "done") return "ok";
  if (status === "rejected") return "danger";
  if (status === "in_progress" || status === "returned") return "warn";
  if (status === "review") return "muted";
  return "muted";
}

function updateUsersPendingBadge(count) {
  const badge = qs("users-pending-badge");
  if (!badge) return;
  badge.textContent = String(count || 0);
  badge.hidden = !count;
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
      return;
    }
    if (state.activeSection === "glossary") {
      void loadGlossary().catch(() => {});
      return;
    }
    if (state.activeSection === "feedback") {
      void loadFeedback().catch(() => {});
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
    <div class="admin-card"><div class="admin-kpi-label">Пользователи</div><div class="admin-kpi-value">${users.total ?? 0}</div><div class="admin-kpi-sub">pending: ${users.pending_approval ?? 0}, active: ${users.active ?? 0}</div></div>
    <div class="admin-card"><div class="admin-kpi-label">Чеклисты</div><div class="admin-kpi-value">${counts.checklists ?? 0}</div><div class="admin-kpi-sub">training types: ${counts.training_types ?? 0}</div></div>
    <div class="admin-card"><div class="admin-kpi-label">Локации</div><div class="admin-kpi-value">${counts.locations ?? 0}</div><div class="admin-kpi-sub">managers: ${counts.managers ?? 0}</div></div>
    <div class="admin-card"><div class="admin-kpi-label">Оценки</div><div class="admin-kpi-value">${data.analytics?.total_evaluations ?? 0}</div><div class="admin-kpi-sub">avg: ${data.analytics?.average_score ?? "—"}</div></div>
  `;
  updateUsersPendingBadge(Number(users.pending_approval || 0));

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

function filteredUsers() {
  const query = String(qs("users-query")?.value || "").trim().toLowerCase();
  const status = String(qs("users-status-filter")?.value || "").trim();
  const source = String(qs("users-source-filter")?.value || "").trim();
  const role = String(qs("users-role-filter")?.value || "").trim();
  return [...(state.users || [])]
    .filter((item) => {
      if (status && userAccessStatus(item) !== status) return false;
      if (source && String(item.auth_source || "") !== source) return false;
      if (role && String(item.effective_role || item.stored_role || "") !== role) return false;
      if (!query) return true;
      const haystack = [
        item.username,
        corporateEmailForUser(item),
        item.full_name,
        item.display_name,
        item.location_name,
        item.department_label,
      ].join(" ").toLowerCase();
      return haystack.includes(query);
    })
    .sort((a, b) => {
      const ap = userAccessStatus(a) === "pending" ? 0 : 1;
      const bp = userAccessStatus(b) === "pending" ? 0 : 1;
      if (ap !== bp) return ap - bp;
      return String(a.username || "").localeCompare(String(b.username || ""), "ru");
    });
}

function renderPendingUsers() {
  const pending = (state.users || []).filter((item) => userAccessStatus(item) === "pending");
  const list = qs("users-pending-list");
  const count = qs("users-pending-count");
  if (count) count.textContent = `${pending.length} ${pending.length === 1 ? "заявка" : "заявок"}`;
  updateUsersPendingBadge(pending.length);
  if (!list) return;
  if (!pending.length) {
    list.innerHTML = '<div class="admin-empty">Нет пользователей на подтверждении.</div>';
    return;
  }
  list.innerHTML = pending.map((item) => `
    <div class="admin-pending-user">
      <div>
        <strong>${escapeHtml(item.full_name || item.display_name || item.username || "—")}</strong>
        <div class="admin-note">${escapeHtml(item.username || "—")} · ${escapeHtml(corporateEmailForUser(item) || "—")} · ${escapeHtml(item.location_name || item.location_id || "—")} · ${escapeHtml(item.department_label || "—")}</div>
      </div>
      <div class="admin-table-actions">
        <button type="button" class="admin-inline-btn admin-inline-btn--brand" data-user-action="approve" data-username="${escapeHtml(item.username)}">Подтвердить</button>
        <button type="button" class="admin-inline-btn admin-inline-btn--danger" data-user-action="reject" data-username="${escapeHtml(item.username)}">Отказать</button>
      </div>
    </div>
  `).join("");
}

function renderUsers() {
  const tbody = qs("users-body");
  const usersTable = tbody?.closest("table");
  usersTable?.querySelectorAll("thead th").forEach((th) => {
    if (String(th.textContent || "").trim().toLowerCase() === "действия") th.remove();
  });
  tbody.innerHTML = "";
  renderPendingUsers();
  const items = filteredUsers();
  for (const item of items) {
    const row = document.createElement("tr");
    row.dataset.username = item.username;
    row.classList.toggle("is-selected", state.selectedUser === item.username);
    row.innerHTML = `
      <td class="admin-code">${escapeHtml(item.username)}</td>
      <td>${escapeHtml(corporateEmailForUser(item) || "—")}</td>
      <td>${escapeHtml(item.full_name || item.display_name || "—")}</td>
      <td>${escapeHtml(item.auth_source || "—")}</td>
      <td>${escapeHtml(item.location_name || item.location_id || "—")}</td>
      <td>${escapeHtml(item.department_label || "—")}</td>
      <td>${makePill(item.effective_role || "user", item.is_admin ? "ok" : "muted")}</td>
      <td>${userStatusPill(item)}</td>
    `;
    tbody.appendChild(row);
  }
  if (!tbody.children.length) tbody.innerHTML = '<tr><td colspan="8" class="admin-empty">Нет пользователей по выбранным фильтрам.</td></tr>';
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

function updateUserCorporateEmailPreview() {
  const username = String(qs("user-username")?.value || "").trim().toLowerCase();
  const node = qs("user-corporate-email");
  if (!node) return;
  node.value = username && !username.includes("@") ? `${username}@freshauto.ru` : "";
}

function resetUserForm() {
  state.selectedUser = null;
  qs("user-form-title").textContent = "Новый пользователь";
  qs("user-form-meta").textContent = "Создание локальной или AD-учётной записи и управление доступом.";
  qs("user-save-btn").textContent = "Создать пользователя";
  qs("user-username").disabled = false;
  qs("user-auth-source").disabled = false;
  qs("user-username").value = "";
  qs("user-corporate-email").value = "";
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
  qs("user-approve-btn").disabled = true;
  qs("user-activate-btn").disabled = true;
  qs("user-deactivate-btn").disabled = true;
  qs("user-reset-password-btn").disabled = true;
  qs("user-delete-btn").disabled = true;
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
    corporateEmailForUser(item),
    item.location_name || item.location_id,
    item.department_label,
    item.approved_at ? `Подтверждён: ${formatDateTime(item.approved_at)}` : null,
    item.rejection_reason ? `Отказ: ${item.rejection_reason}` : null,
  ].filter(Boolean).join(" · ") || "Редактирование профиля и прав доступа.";
  qs("user-save-btn").textContent = "Сохранить изменения";
  qs("user-username").disabled = true;
  qs("user-auth-source").disabled = true;
  qs("user-username").value = item.username || "";
  qs("user-corporate-email").value = corporateEmailForUser(item) || "";
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
  const pending = userAccessStatus(item) === "pending";
  qs("user-approve-btn").disabled = !pending;
  qs("user-activate-btn").disabled = Boolean(item.is_active) || pending;
  qs("user-deactivate-btn").disabled = !Boolean(item.is_active) && !pending;
  qs("user-reset-password-btn").disabled = item.auth_source !== "local";
  qs("user-delete-btn").disabled =
    Boolean(item.forced_admin) ||
    String(item.username || "").toLowerCase() === String(state.me?.user || "").toLowerCase();
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
      <td>${escapeHtml(item.checklist_display_name_snapshot || item.checklist_slug_snapshot || "—")}</td>
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
      <td>${escapeHtml(item.version || "1")}</td>
    `;
    tbody.appendChild(row);
  }
  if (!tbody.children.length) tbody.innerHTML = '<tr><td colspan="4" class="admin-empty">Нет чеклистов.</td></tr>';
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

function glossaryVariantsFromText(value) {
  return String(value || "")
    .split(/[\n,]+/)
    .map((part) => part.trim())
    .filter(Boolean);
}

function filteredGlossaryItems() {
  const query = String(qs("glossary-query")?.value || "").trim().toLowerCase();
  const category = String(qs("glossary-category-filter")?.value || "").trim();
  const activeFilter = String(qs("glossary-active-filter")?.value || "").trim();
  return (state.glossary || []).filter((item) => {
    if (category && item.category !== category) return false;
    if (activeFilter === "active" && !item.is_active) return false;
    if (activeFilter === "inactive" && item.is_active) return false;
    if (!query) return true;
    const haystack = [
      item.id,
      item.category,
      item.term,
      ...(item.variants || []),
      item.definition,
      item.whisper_hint,
      item.llm_hint,
    ].join(" ").toLowerCase();
    return haystack.includes(query);
  });
}

function renderGlossaryCategoryFilter() {
  const select = qs("glossary-category-filter");
  if (!select) return;
  const current = select.value;
  select.innerHTML = '<option value="">Все категории</option>';
  for (const category of state.glossaryCategories || []) {
    const option = document.createElement("option");
    option.value = category;
    option.textContent = category;
    select.appendChild(option);
  }
  if ([...select.options].some((option) => option.value === current)) select.value = current;
}

function renderGlossary() {
  renderGlossaryCategoryFilter();
  const tbody = qs("glossary-body");
  if (!tbody) return;
  tbody.innerHTML = "";
  for (const item of filteredGlossaryItems()) {
    const row = document.createElement("tr");
    row.dataset.glossaryId = item.id;
    row.classList.toggle("is-selected", item.id === state.selectedGlossaryId);
    const usage = [
      item.use_for_whisper ? "Whisper" : null,
      item.use_for_llm ? "LLM" : null,
    ].filter(Boolean).join(" + ") || "—";
    row.innerHTML = `
      <td><strong>${escapeHtml(item.term || "—")}</strong><div class="admin-note admin-code">${escapeHtml(item.id || "")}</div></td>
      <td>${escapeHtml(item.category || "—")}</td>
      <td>${escapeHtml(usage)}</td>
      <td>${makePill(item.is_active ? "active" : "disabled", item.is_active ? "ok" : "danger")}</td>
    `;
    tbody.appendChild(row);
  }
  if (!tbody.children.length) tbody.innerHTML = '<tr><td colspan="4" class="admin-empty">Термины не найдены.</td></tr>';
}

function resetGlossaryForm() {
  state.selectedGlossaryId = null;
  qs("glossary-editor-title").textContent = "Новый термин";
  qs("glossary-editor-meta").textContent = "Заполните основное написание, варианты и отдельные подсказки для ASR/LLM.";
  qs("glossary-id").disabled = false;
  qs("glossary-id").value = "";
  qs("glossary-category").value = "";
  qs("glossary-term").value = "";
  qs("glossary-sort-order").value = "";
  qs("glossary-variants").value = "";
  qs("glossary-definition").value = "";
  qs("glossary-whisper-hint").value = "";
  qs("glossary-llm-hint").value = "";
  qs("glossary-use-whisper").checked = true;
  qs("glossary-use-llm").checked = true;
  qs("glossary-is-active").checked = true;
  qs("glossary-disable-btn").disabled = true;
  const preview = qs("glossary-preview");
  if (preview) preview.hidden = true;
  renderGlossary();
}

function fillGlossaryForm(entryId) {
  const item = (state.glossary || []).find((row) => row.id === entryId);
  if (!item) return;
  state.selectedGlossaryId = item.id;
  qs("glossary-editor-title").textContent = `Термин: ${item.term || item.id}`;
  qs("glossary-editor-meta").textContent = `Обновлено ${formatDateTime(item.updated_at)} · ${item.updated_by || "system"}`;
  qs("glossary-id").disabled = true;
  qs("glossary-id").value = item.id || "";
  qs("glossary-category").value = item.category || "";
  qs("glossary-term").value = item.term || "";
  qs("glossary-sort-order").value = item.sort_order ?? "";
  qs("glossary-variants").value = (item.variants || []).join("\n");
  qs("glossary-definition").value = item.definition || "";
  qs("glossary-whisper-hint").value = item.whisper_hint || "";
  qs("glossary-llm-hint").value = item.llm_hint || "";
  qs("glossary-use-whisper").checked = Boolean(item.use_for_whisper);
  qs("glossary-use-llm").checked = Boolean(item.use_for_llm);
  qs("glossary-is-active").checked = Boolean(item.is_active);
  qs("glossary-disable-btn").disabled = !Boolean(item.is_active);
  renderGlossary();
}

function glossaryPayloadFromForm() {
  return {
    id: qs("glossary-id").value.trim() || null,
    category: qs("glossary-category").value.trim(),
    term: qs("glossary-term").value.trim(),
    variants: glossaryVariantsFromText(qs("glossary-variants").value),
    definition: qs("glossary-definition").value.trim(),
    whisper_hint: qs("glossary-whisper-hint").value.trim(),
    llm_hint: qs("glossary-llm-hint").value.trim(),
    use_for_whisper: qs("glossary-use-whisper").checked,
    use_for_llm: qs("glossary-use-llm").checked,
    is_active: qs("glossary-is-active").checked,
    sort_order: numberInputValueOrNull("glossary-sort-order"),
  };
}

function renderFeedbackFilterOptions() {
  const statusSelect = qs("feedback-status-filter");
  const typeSelect = qs("feedback-type-filter");
  if (statusSelect) {
    const current = statusSelect.value;
    statusSelect.innerHTML = '<option value="">Все</option>';
    for (const item of state.feedbackStatuses || []) {
      const count = state.feedbackCounts?.[item.value];
      const option = document.createElement("option");
      option.value = item.value;
      option.textContent = count == null ? item.label : `${item.label} (${count})`;
      statusSelect.appendChild(option);
    }
    if ([...statusSelect.options].some((option) => option.value === current)) statusSelect.value = current;
  }
  if (typeSelect) {
    const current = typeSelect.value;
    typeSelect.innerHTML = '<option value="">Все</option>';
    for (const item of state.feedbackTypes || []) {
      const option = document.createElement("option");
      option.value = item.value;
      option.textContent = item.label;
      typeSelect.appendChild(option);
    }
    if ([...typeSelect.options].some((option) => option.value === current)) typeSelect.value = current;
  }
}

function feedbackEventsHtml(item) {
  const events = item?.events || [];
  if (!events.length) return '<p class="admin-note">История пока пустая.</p>';
  return `
    <div class="admin-feedback-events">
      ${events
        .map(
          (event) => `
            <div class="admin-feedback-event">
              <div><strong>${escapeHtml(event.action || "—")}</strong> · ${escapeHtml(formatDateTime(event.created_at))}</div>
              <div class="admin-note">${escapeHtml(event.actor || "system")} · ${escapeHtml(event.status_from || "—")} → ${escapeHtml(event.status_to || "—")}</div>
              ${event.comment ? `<p>${escapeHtml(event.comment)}</p>` : ""}
            </div>
          `,
        )
        .join("")}
    </div>
  `;
}

function renderFeedback() {
  renderFeedbackFilterOptions();
  const tbody = qs("feedback-body");
  const detail = qs("feedback-detail");
  const summary = qs("feedback-detail-summary");
  const title = qs("feedback-detail-title");
  const comment = qs("feedback-admin-comment");
  if (!tbody || !detail) return;
  tbody.innerHTML = "";
  for (const item of state.feedback || []) {
    const row = document.createElement("tr");
    row.dataset.feedbackId = item.id;
    row.classList.toggle("is-selected", item.id === state.selectedFeedbackId);
    row.innerHTML = `
      <td class="admin-code">${escapeHtml(item.id || "—")}</td>
      <td>${escapeHtml(item.user_display_name || item.user_username || "—")}</td>
      <td>${escapeHtml(formatDateTime(item.created_at))}</td>
      <td>${escapeHtml(feedbackTypeLabel(item.feedback_type))}</td>
      <td>${escapeHtml(item.target_label || item.target_id || "—")}</td>
      <td>${escapeHtml(feedbackRatingLabel(item))}</td>
      <td>${makePill(feedbackStatusLabel(item), feedbackStatusTone(item.status))}</td>
    `;
    tbody.appendChild(row);
  }
  if (!tbody.children.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="admin-empty">Заявки не найдены.</td></tr>';
  }

  const selected = (state.feedback || []).find((item) => item.id === state.selectedFeedbackId) || null;
  const buttons = {
    start: qs("feedback-start-btn"),
    review: qs("feedback-review-btn"),
    complete: qs("feedback-complete-btn"),
    reject: qs("feedback-reject-btn"),
  };
  if (!selected) {
    if (title) title.textContent = "Карточка заявки";
    if (summary) summary.textContent = "Выберите тикет слева, чтобы увидеть описание, историю и действия.";
    detail.innerHTML = '<div class="admin-meta__row"><span class="admin-meta__key">Заявка</span><strong>Не выбрана</strong></div>';
    if (comment) {
      comment.value = "";
      comment.disabled = true;
    }
    Object.values(buttons).forEach((button) => {
      if (button) button.disabled = true;
    });
    return;
  }

  if (title) title.textContent = selected.id || "Карточка заявки";
  if (summary) {
    summary.textContent = `${feedbackStatusLabel(selected)} · ${feedbackTypeLabel(selected.feedback_type)} · ${selected.user_display_name || selected.user_username || "—"}`;
  }
  if (comment) comment.disabled = false;
  detail.innerHTML = `
    <div class="admin-meta__row"><span class="admin-meta__key">Пользователь</span><strong>${escapeHtml(selected.user_display_name || selected.user_username || "—")}</strong></div>
    <div class="admin-meta__row"><span class="admin-meta__key">Создано</span><strong>${escapeHtml(formatDateTime(selected.created_at))}</strong></div>
    <div class="admin-meta__row"><span class="admin-meta__key">Тип</span><strong>${escapeHtml(feedbackTypeLabel(selected.feedback_type))}</strong></div>
    <div class="admin-meta__row"><span class="admin-meta__key">Объект</span><strong>${escapeHtml(selected.target_label || selected.target_id || "—")}</strong></div>
    <div class="admin-meta__row"><span class="admin-meta__key">Оценка</span><strong>${escapeHtml(feedbackRatingLabel(selected))}</strong></div>
    <div class="admin-meta__row"><span class="admin-meta__key">Статус</span><strong>${feedbackStatusLabel(selected)}</strong></div>
    <div class="admin-feedback-detail-block"><strong>Описание</strong><p>${escapeHtml(selected.description || "—")}</p></div>
    ${
      selected.admin_comment
        ? `<div class="admin-feedback-detail-block"><strong>Комментарий администратора</strong><p>${escapeHtml(selected.admin_comment)}</p></div>`
        : ""
    }
    ${
      selected.user_review_comment
        ? `<div class="admin-feedback-detail-block"><strong>Комментарий пользователя</strong><p>${escapeHtml(selected.user_review_comment)}</p></div>`
        : ""
    }
    <div class="admin-feedback-detail-block"><strong>История</strong>${feedbackEventsHtml(selected)}</div>
  `;
  if (buttons.start) buttons.start.disabled = !selected.can_admin_start;
  if (buttons.review) buttons.review.disabled = !selected.can_admin_send_for_review;
  if (buttons.complete) buttons.complete.disabled = !selected.can_admin_complete;
  if (buttons.reject) buttons.reject.disabled = !selected.can_admin_reject;
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
  renderApiKeys();
}

function renderApiKeys() {
  const tbody = qs("api-keys-body");
  if (!tbody) return;
  tbody.innerHTML = "";
  for (const item of state.apiKeys || []) {
    const active = Boolean(item.is_active) && !item.revoked_at;
    const row = document.createElement("tr");
    row.dataset.apiKeyId = item.id || "";
    row.innerHTML = `
      <td>
        <strong>${escapeHtml(item.name || "—")}</strong>
        <div class="admin-note">${escapeHtml(item.created_by || "system")}</div>
      </td>
      <td class="admin-code">${escapeHtml(item.key_prefix || "—")}…</td>
      <td>${makePill(active ? "active" : "revoked", active ? "ok" : "muted")}</td>
      <td>${escapeHtml(formatDateTime(item.created_at))}</td>
      <td>${escapeHtml(item.last_used_at ? formatDateTime(item.last_used_at) : "—")}</td>
      <td>
        <button type="button" class="admin-inline-btn admin-inline-btn--danger" data-api-key-revoke="${escapeHtml(item.id || "")}" ${active ? "" : "disabled"}>Отозвать</button>
      </td>
    `;
    tbody.appendChild(row);
  }
  if (!tbody.children.length) {
    tbody.innerHTML = '<tr><td colspan="6" class="admin-empty">API-ключей пока нет.</td></tr>';
  }
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
  const [settings, apiKeys] = await Promise.all([
    apiJson(`${API}/api/admin/settings`),
    apiJson(`${API}/api/admin/api-keys`),
  ]);
  state.settings = settings;
  state.apiKeys = apiKeys.items || [];
  renderSettings();
  restartAutoRefresh();
  touchUpdatedAt();
}

async function loadGlossary() {
  const data = await apiJson(`${API}/api/admin/glossary`);
  state.glossary = data.items || [];
  state.glossaryCategories = data.categories || [];
  if (state.selectedGlossaryId && !state.glossary.some((item) => item.id === state.selectedGlossaryId)) {
    resetGlossaryForm();
  } else if (state.selectedGlossaryId) {
    fillGlossaryForm(state.selectedGlossaryId);
  } else {
    renderGlossary();
  }
  touchUpdatedAt();
}

async function loadFeedback() {
  const params = new URLSearchParams();
  if (qs("feedback-query").value.trim()) params.set("q", qs("feedback-query").value.trim());
  if (qs("feedback-status-filter").value) params.set("status", qs("feedback-status-filter").value);
  if (qs("feedback-type-filter").value) params.set("feedback_type", qs("feedback-type-filter").value);
  const data = await apiJson(`${API}/api/admin/feedback?${params.toString()}`);
  state.feedback = data.items || [];
  state.feedbackTypes = data.types || [];
  state.feedbackStatuses = data.statuses || [];
  state.feedbackCounts = data.counts || {};
  if (state.selectedFeedbackId && !state.feedback.some((item) => item.id === state.selectedFeedbackId)) {
    state.selectedFeedbackId = null;
    const comment = qs("feedback-admin-comment");
    if (comment) comment.value = "";
  }
  renderFeedback();
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
  await Promise.all([loadOverview(), loadJobs(), loadUsers(), loadVideos(), loadChecklists(), loadSettings(), loadGlossary(), loadFeedback(), loadAudit()]);
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
    if (body.auth_source === "local" && (body.username.includes("@") || /\s/u.test(body.username))) {
      setFlash("Укажите логин без @: первую часть корпоративной почты.", "error", 0);
      return;
    }
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

function openUserRejectDialog(username) {
  const dialog = qs("user-reject-dialog");
  const form = qs("user-reject-form");
  const reasonInput = qs("user-reject-reason");
  const error = qs("user-reject-error");
  const subtitle = qs("user-reject-subtitle");
  const cancel = qs("user-reject-cancel");
  const close = qs("user-reject-cancel-x");
  if (!dialog || !form || !reasonInput || !error) {
    return Promise.resolve("");
  }

  const user = (state.users || []).find((item) => item.username === username) || {};
  const label = user.full_name || user.display_name || username || "пользователю";
  if (subtitle) {
    subtitle.textContent = `Укажите причину отказа для ${label}. Пользователь увидит это сообщение при попытке входа.`;
  }
  reasonInput.value = "";
  error.hidden = true;

  return new Promise((resolve) => {
    let settled = false;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      form.removeEventListener("submit", onSubmit);
      cancel?.removeEventListener("click", onCancel);
      close?.removeEventListener("click", onCancel);
      dialog.removeEventListener("cancel", onCancel);
      dialog.removeEventListener("close", onClose);
      if (dialog.open) dialog.close();
      resolve(value);
    };
    const onSubmit = (event) => {
      event.preventDefault();
      const reason = String(reasonInput.value || "").trim();
      if (!reason) {
        error.hidden = false;
        reasonInput.focus();
        return;
      }
      finish(reason);
    };
    const onCancel = (event) => {
      event.preventDefault();
      finish("");
    };
    const onClose = () => {
      finish("");
    };

    form.addEventListener("submit", onSubmit);
    cancel?.addEventListener("click", onCancel);
    close?.addEventListener("click", onCancel);
    dialog.addEventListener("cancel", onCancel);
    dialog.addEventListener("close", onClose);
    if (typeof dialog.showModal === "function") dialog.showModal();
    else dialog.setAttribute("open", "");
    requestAnimationFrame(() => reasonInput.focus());
  });
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
    } else if (action === "delete") {
      const item = (state.users || []).find((row) => row.username === state.selectedUser) || {};
      const label = item.full_name || item.display_name || state.selectedUser;
      if (!window.confirm(`Удалить пользователя «${label}»? Это действие нельзя отменить.`)) return;
      await apiJson(`${API}/api/admin/users/${encodeURIComponent(state.selectedUser)}`, {
        method: "DELETE",
      });
      resetUserForm();
      setFlash("Пользователь удалён.", "ok");
    } else if (action === "approve") {
      await apiJson(`${API}/api/admin/users/${encodeURIComponent(state.selectedUser)}/approve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
      });
      setFlash("Регистрация подтверждена. Пользователь может войти.", "ok");
    } else if (action === "reject") {
      const trimmedReason = await openUserRejectDialog(state.selectedUser);
      if (!trimmedReason) return;
      await apiJson(`${API}/api/admin/users/${encodeURIComponent(state.selectedUser)}/reject`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reason: trimmedReason }),
      });
      setFlash("В регистрации отказано. Причина будет показана пользователю при входе.", "ok");
    } else {
      const path = action === "activate" ? "activate" : "deactivate";
      await apiJson(`${API}/api/admin/users/${encodeURIComponent(state.selectedUser)}/${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
      });
      setFlash(action === "activate" ? "Пользователь активирован." : "Пользователь отключён.", "ok");
    }
    await Promise.all([loadUsers(), loadOverview(), refreshReferenceData(), loadAudit()]);
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
      const item = (state.videos || []).find((row) => row.stem === stem) || {};
      const label = item.display_title || item.video_file || stem;
      const ok = window.confirm(
        `Видео «${label}» будет удалено безвозвратно.\n\nБудут удалены файл, транскрипт, тон, ручные и ИИ-оценки, сравнение и история задач. Продолжить?`,
      );
      if (!ok) return;
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

async function handleApiKeyCreate(event) {
  event.preventDefault();
  const nameEl = qs("api-key-name");
  const button = qs("api-key-create-btn");
  const name = String(nameEl?.value || "").trim();
  if (!name) {
    setFlash("Укажите название API-ключа.", "error", 0);
    nameEl?.focus();
    return;
  }
  setBusy(button, true);
  try {
    const data = await apiJson(`${API}/api/admin/api-keys`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, role: "admin" }),
    });
    if (nameEl) nameEl.value = "";
    const secretBox = qs("api-key-secret-box");
    const secretInput = qs("api-key-secret");
    if (secretBox && secretInput) {
      secretInput.value = data.key?.api_key || "";
      secretBox.hidden = false;
    }
    setFlash("API-ключ создан. Скопируйте его сейчас.", "ok", 5000);
    await Promise.all([loadSettings(), loadAudit()]);
  } catch (error) {
    setFlash(error.message || String(error), "error", 0);
  } finally {
    setBusy(button, false);
  }
}

async function copyApiKeySecret() {
  const input = qs("api-key-secret");
  const value = String(input?.value || "");
  if (!value) return;
  try {
    await navigator.clipboard.writeText(value);
    setFlash("API-ключ скопирован.", "ok", 2200);
  } catch (_) {
    input?.select();
    setFlash("Не удалось скопировать автоматически. Ключ выделен в поле.", "info", 3500);
  }
}

async function revokeApiKey(keyId) {
  if (!keyId) return;
  const item = (state.apiKeys || []).find((row) => row.id === keyId) || {};
  const label = item.name || item.key_prefix || keyId;
  if (!window.confirm(`Отозвать API-ключ «${label}»? Qlik больше не сможет использовать этот ключ.`)) return;
  try {
    await apiJson(`${API}/api/admin/api-keys/${encodeURIComponent(keyId)}`, { method: "DELETE" });
    setFlash("API-ключ отозван.", "ok");
    await Promise.all([loadSettings(), loadAudit()]);
  } catch (error) {
    setFlash(error.message || String(error), "error", 0);
  }
}

async function handleGlossarySubmit(event) {
  event.preventDefault();
  const button = qs("glossary-save-btn");
  setBusy(button, true);
  try {
    const payload = glossaryPayloadFromForm();
    if (!payload.term) {
      setFlash("Укажите основной термин.", "error", 0);
      return;
    }
    const url = state.selectedGlossaryId
      ? `${API}/api/admin/glossary/${encodeURIComponent(state.selectedGlossaryId)}`
      : `${API}/api/admin/glossary`;
    const data = await apiJson(url, {
      method: state.selectedGlossaryId ? "PUT" : "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    setFlash("Термин глоссария сохранён.", "ok");
    state.selectedGlossaryId = data.entry?.id || state.selectedGlossaryId;
    await Promise.all([loadGlossary(), loadAudit()]);
  } catch (error) {
    setFlash(error.message || String(error), "error", 0);
  } finally {
    setBusy(button, false);
  }
}

async function handleGlossaryDisable() {
  if (!state.selectedGlossaryId) {
    setFlash("Сначала выберите термин.", "error", 0);
    return;
  }
  try {
    await apiJson(`${API}/api/admin/glossary/${encodeURIComponent(state.selectedGlossaryId)}`, {
      method: "DELETE",
    });
    setFlash("Термин отключён. Его можно снова включить через флаг «Активен».", "ok");
    await Promise.all([loadGlossary(), loadAudit()]);
  } catch (error) {
    setFlash(error.message || String(error), "error", 0);
  }
}

async function handleGlossaryPreview() {
  const preview = qs("glossary-preview");
  if (!preview) return;
  const button = qs("glossary-preview-btn");
  setBusy(button, true);
  try {
    const data = await apiJson(`${API}/api/admin/glossary/preview`);
    preview.hidden = false;
    preview.innerHTML = `
      <h4>Whisper initial_prompt <span class="admin-note">${data.whisper_chars ?? 0} символов</span></h4>
      <pre class="admin-code">${escapeHtml(data.whisper_prompt || "Пусто")}</pre>
      <h4>LLM glossary block <span class="admin-note">${data.eval_chars ?? 0} символов</span></h4>
      <pre class="admin-code">${escapeHtml(data.eval_prompt || "Пусто")}</pre>
    `;
  } catch (error) {
    setFlash(error.message || String(error), "error", 0);
  } finally {
    setBusy(button, false);
  }
}

async function handleFeedbackAction(action) {
  if (!state.selectedFeedbackId) {
    setFlash("Сначала выберите заявку обратной связи.", "error", 0);
    return;
  }
  const commentEl = qs("feedback-admin-comment");
  const comment = String((commentEl && commentEl.value) || "").trim();
  if (!comment) {
    setFlash("Укажите комментарий для смены статуса.", "error", 0);
    commentEl?.focus();
    return;
  }
  const endpoint = {
    start: "start",
    review: "send-for-review",
    complete: "complete",
    reject: "reject",
  }[action];
  if (!endpoint) return;
  try {
    await apiJson(`${API}/api/admin/feedback/${encodeURIComponent(state.selectedFeedbackId)}/${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ comment }),
    });
    if (commentEl) commentEl.value = "";
    setFlash("Статус заявки обновлён.", "ok");
    await Promise.all([loadFeedback(), loadAudit(), loadOverview()]);
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
  qs("user-new-btn").addEventListener("click", resetUserForm);
  qs("user-username").addEventListener("input", updateUserCorporateEmailPreview);
  qs("users-query").addEventListener("input", renderUsers);
  qs("users-status-filter").addEventListener("change", renderUsers);
  qs("users-source-filter").addEventListener("change", renderUsers);
  qs("users-role-filter").addEventListener("change", renderUsers);
  qs("user-approve-btn").addEventListener("click", () => void handleUserLifecycle("approve"));
  qs("user-activate-btn").addEventListener("click", () => void handleUserLifecycle("activate"));
  qs("user-deactivate-btn").addEventListener("click", () => void handleUserLifecycle("deactivate"));
  qs("user-reset-password-btn").addEventListener("click", () => void handleUserLifecycle("reset-password"));
  qs("user-delete-btn").addEventListener("click", () => void handleUserLifecycle("delete"));

  qs("video-form").addEventListener("submit", handleVideoSubmit);
  qs("video-resume-btn").addEventListener("click", () => void handleVideoAction(state.selectedVideo, "resume"));
  qs("video-restart-btn").addEventListener("click", () => void handleVideoAction(state.selectedVideo, "restart"));
  qs("video-re-eval-btn").addEventListener("click", () => void handleVideoAction(state.selectedVideo, "re-evaluate"));
  qs("video-restore-btn").addEventListener("click", () => void handleVideoAction(state.selectedVideo, "restore"));
  qs("video-delete-btn").addEventListener("click", () => void handleVideoAction(state.selectedVideo, "delete"));

  qs("create-checklist-btn").addEventListener("click", () => void handleChecklistCreate());
  qs("checklist-editor-form").addEventListener("submit", handleChecklistSave);
  qs("checklist-delete-btn").addEventListener("click", () => void handleChecklistDelete());
  qs("checklist-add-row").addEventListener("click", () => appendChecklistCriterionRow());

  qs("manager-form").addEventListener("submit", handleManagerSubmit);
  qs("location-form").addEventListener("submit", handleLocationSubmit);
  qs("training-type-form").addEventListener("submit", handleTrainingTypeSubmit);
  qs("training-form-reset").addEventListener("click", resetTrainingTypeForm);
  qs("training-type-delete-btn").addEventListener("click", () => void handleTrainingTypeDelete());

  qs("glossary-query").addEventListener("input", renderGlossary);
  qs("glossary-category-filter").addEventListener("change", renderGlossary);
  qs("glossary-active-filter").addEventListener("change", renderGlossary);
  qs("glossary-new-btn").addEventListener("click", resetGlossaryForm);
  qs("glossary-reset-btn").addEventListener("click", resetGlossaryForm);
  qs("glossary-form").addEventListener("submit", handleGlossarySubmit);
  qs("glossary-disable-btn").addEventListener("click", () => void handleGlossaryDisable());
  qs("glossary-preview-btn").addEventListener("click", () => void handleGlossaryPreview());

  qs("feedback-query").addEventListener("input", () => void loadFeedback());
  qs("feedback-status-filter").addEventListener("change", () => void loadFeedback());
  qs("feedback-type-filter").addEventListener("change", () => void loadFeedback());
  qs("feedback-start-btn").addEventListener("click", () => void handleFeedbackAction("start"));
  qs("feedback-review-btn").addEventListener("click", () => void handleFeedbackAction("review"));
  qs("feedback-complete-btn").addEventListener("click", () => void handleFeedbackAction("complete"));
  qs("feedback-reject-btn").addEventListener("click", () => void handleFeedbackAction("reject"));

  qs("settings-form").addEventListener("submit", handleSettingsSubmit);
  qs("api-key-form").addEventListener("submit", handleApiKeyCreate);
  qs("api-key-copy-btn").addEventListener("click", () => void copyApiKeySecret());

  qs("admin-logout").addEventListener("click", async () => {
    await apiFetch(`${API}/api/auth/logout`, { method: "POST" });
    window.location.href = "/login.html";
  });

  document.addEventListener("click", (event) => {
    const target = event.target;
    const userAction = target.closest && target.closest("[data-user-action][data-username]");
    if (userAction) {
      const username = userAction.dataset.username;
      if (userAction.dataset.userAction === "approve" || userAction.dataset.userAction === "reject") {
        state.selectedUser = username;
        void handleUserLifecycle(userAction.dataset.userAction);
      } else if (username) {
        fillUserForm(username);
      }
      return;
    }
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

    const glossaryRow = target.closest && target.closest("#glossary-body tr[data-glossary-id]");
    if (glossaryRow) {
      fillGlossaryForm(glossaryRow.dataset.glossaryId);
      return;
    }

    const feedbackRow = target.closest && target.closest("#feedback-body tr[data-feedback-id]");
    if (feedbackRow) {
      state.selectedFeedbackId = feedbackRow.dataset.feedbackId;
      const comment = qs("feedback-admin-comment");
      if (comment) comment.value = "";
      renderFeedback();
      return;
    }

    const apiKeyRevoke = target.closest && target.closest("[data-api-key-revoke]");
    if (apiKeyRevoke) {
      void revokeApiKey(apiKeyRevoke.dataset.apiKeyRevoke);
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
    resetGlossaryForm();
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
