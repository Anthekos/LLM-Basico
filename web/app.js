const STORAGE_KEY = "llm-basico-web-v1";
const DEFAULT_STATE = {
  name: "Usuario",
  chats: [],
  activeChatId: null,
  modelSettings: { temperature: 0.8, topK: 8, contextWindow: 32 },
  onlineFineTuning: false,
};

const elements = {
  chatList: document.querySelector("#chat-list"),
  exampleList: document.querySelector("#example-list"),
  corpusList: document.querySelector("#corpus-list"),
  corpusCount: document.querySelector("#corpus-count"),
  conversation: document.querySelector("#conversation"),
  welcome: document.querySelector("#welcome"),
  messageList: document.querySelector("#message-list"),
  form: document.querySelector("#chat-form"),
  input: document.querySelector("#message-input"),
  send: document.querySelector("#send-button"),
  pdfInput: document.querySelector("#pdf-input"),
  toast: document.querySelector("#toast"),
  dialog: document.querySelector("#name-dialog"),
  nameForm: document.querySelector("#name-form"),
  nameInput: document.querySelector("#name-input"),
  sidebar: document.querySelector("#sidebar"),
  controlPanel: document.querySelector("#control-panel"),
  controlBackdrop: document.querySelector("#control-backdrop"),
  corpusContent: document.querySelector("#corpus-content"),
  modelContent: document.querySelector("#model-content"),
  previewDialog: document.querySelector("#preview-dialog"),
  pdfFrame: document.querySelector("#pdf-frame"),
  textPreview: document.querySelector("#text-preview"),
  onlineFineTuning: document.querySelector("#online-finetuning"),
  retry: document.querySelector("#retry-last"),
  corpusNoteForm: document.querySelector("#corpus-note-form"),
  corpusRemoveForm: document.querySelector("#corpus-remove-form"),
};

function loadState() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
    if (saved && typeof saved.name === "string" && Array.isArray(saved.chats)) {
      const chats = saved.chats.filter(chat =>
        chat && typeof chat.id === "string" && Array.isArray(chat.messages)
      );
      const activeChatId = chats.some(chat => chat.id === saved.activeChatId)
        ? saved.activeChatId
        : chats[0]?.id || null;
      const temperature = Number(saved.modelSettings?.temperature);
      const topK = Number(saved.modelSettings?.topK);
      const contextWindow = Number(saved.modelSettings?.contextWindow);
      return {
        name: saved.name.slice(0, 32) || "Usuario",
        chats,
        activeChatId,
        onlineFineTuning: saved.onlineFineTuning === true,
        modelSettings: {
          temperature: Number.isFinite(temperature) ? temperature : 0.8,
          topK: Number.isFinite(topK) ? Math.round(topK) : 8,
          contextWindow: Number.isFinite(contextWindow) ? Math.round(contextWindow) : 32,
        },
      };
    }
  } catch (error) {
    console.warn("No se pudo recuperar el estado guardado del navegador.", error);
  }
  return { ...DEFAULT_STATE, chats: [], modelSettings: { ...DEFAULT_STATE.modelSettings } };
}

const state = loadState();
let toastTimer;
let adminToken = null;
let modelConfig = {
  temperature_min: 0.1,
  temperature_max: 3,
  top_k_min: 1,
  top_k_max: 8,
  temperature_default: 0.8,
  top_k_default: 8,
  context_window_min: 4,
  context_window_max: 128,
  context_window_default: 32,
  context_window: 32,
};

function persist() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch (error) {
    showToast("No se pudo guardar este cambio en el navegador.", true);
    console.warn("No se pudo guardar el estado local.", error);
  }
}

function currentChat() {
  return state.chats.find(chat => chat.id === state.activeChatId) || null;
}

function createChat() {
  const chat = {
    id: crypto.randomUUID(),
    title: "Nueva conversación",
    messages: [],
    updatedAt: Date.now(),
  };
  state.chats.unshift(chat);
  state.activeChatId = chat.id;
  persist();
  render();
  elements.input.focus();
  closeSidebar();
}

function renderChatList() {
  elements.chatList.replaceChildren();
  if (state.chats.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-chats";
    empty.textContent = "Tus conversaciones aparecerán aquí.";
    elements.chatList.append(empty);
    return;
  }
  for (const chat of [...state.chats].sort((a, b) => b.updatedAt - a.updatedAt)) {
    const row = document.createElement("div");
    row.className = `chat-list-item${chat.id === state.activeChatId ? " active" : ""}`;
    row.tabIndex = 0;
    row.setAttribute("role", "button");
    row.setAttribute("aria-label", `Abrir ${chat.title}`);
    const title = document.createElement("span");
    title.className = "chat-item-title";
    title.textContent = chat.title;
    const remove = document.createElement("button");
    remove.className = "delete-chat";
    remove.type = "button";
    remove.setAttribute("aria-label", `Eliminar ${chat.title}`);
    remove.textContent = "×";
    remove.addEventListener("click", event => {
      event.stopPropagation();
      state.chats = state.chats.filter(item => item.id !== chat.id);
      if (state.activeChatId === chat.id) state.activeChatId = state.chats[0]?.id || null;
      persist();
      render();
    });
    const open = () => {
      state.activeChatId = chat.id;
      persist();
      render();
      closeSidebar();
    };
    row.addEventListener("click", open);
    row.addEventListener("keydown", event => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        open();
      }
    });
    row.append(title, remove);
    elements.chatList.append(row);
  }
}

function renderMessages() {
  const chat = currentChat();
  elements.messageList.replaceChildren();
  const hasMessages = Boolean(chat?.messages.length);
  elements.welcome.hidden = hasMessages;
  elements.messageList.classList.toggle("visible", hasMessages);
  if (!hasMessages) {
    elements.retry.hidden = true;
    return;
  }
  for (const [messageIndex, message] of chat.messages.entries()) {
    const row = document.createElement("article");
    row.className = `message ${message.role}`;
    const avatar = document.createElement("span");
    avatar.className = "message-avatar";
    avatar.textContent = message.role === "user" ? initials(state.name) : "✳";
    const body = document.createElement("div");
    body.className = "message-body";
    if (message.role === "assistant") {
      const role = document.createElement("div");
      role.className = "message-role";
      role.textContent = "LLM Básico";
      body.append(role);
    }
    if (message.role === "assistant" && message.thought) {
      const thought = document.createElement("details");
      thought.className = "message-thought";
      const summary = document.createElement("summary");
      summary.textContent = "Ver pensamiento generado";
      const thoughtText = document.createElement("div");
      thoughtText.className = "message-thought-text";
      thoughtText.textContent = message.thought;
      thought.append(summary, thoughtText);
      body.append(thought);
    }
    const text = document.createElement("div");
    text.textContent = message.content;
    body.append(text);
    if (message.role === "assistant") {
      const actions = document.createElement("div");
      actions.className = "message-actions";
      const prompt = lastUserBefore(chat, messageIndex + 1);
      if (message.memoryPdf?.download_url) {
        const download = document.createElement("a");
        download.className = "message-action memory-download";
        download.href = message.memoryPdf.download_url;
        download.download = message.memoryPdf.filename || "nota-memoria.pdf";
        download.textContent = "Descargar nota PDF";
        actions.append(download);
      } else {
        const saveMemory = document.createElement("button");
        saveMemory.className = "message-action";
        saveMemory.type = "button";
        saveMemory.textContent = "Guardar como nota de memoria";
        saveMemory.disabled = !prompt || message.memorySaving === true;
        saveMemory.addEventListener("click", () => saveAsMemory(message, prompt?.content, saveMemory));
        actions.append(saveMemory);
      }
      for (const [reward, label] of [[1, "Útil"], [-1, "No útil"]]) {
        const feedback = document.createElement("button");
        feedback.className = `message-action${message.reward === reward ? " selected" : ""}`;
        feedback.type = "button";
        feedback.textContent = reward === 1 ? `👍 ${label}` : `👎 ${label}`;
        feedback.disabled = !prompt || Number.isInteger(message.reward);
        feedback.addEventListener("click", () =>
          sendFeedback(message, prompt?.content, reward, feedback)
        );
        actions.append(feedback);
      }
      body.append(actions);
    }
    if (message.role === "user") row.append(body, avatar);
    else row.append(avatar, body);
    elements.messageList.append(row);
  }
  let ultimaPregunta = -1;
  let ultimaRespuesta = -1;
  chat.messages.forEach((item, index) => {
    if (item.role === "user") ultimaPregunta = index;
    if (item.role === "assistant") ultimaRespuesta = index;
  });
  elements.retry.hidden = ultimaPregunta < 0 || ultimaPregunta < ultimaRespuesta;
  elements.conversation.scrollTop = elements.conversation.scrollHeight;
}

function render() {
  renderChatList();
  renderMessages();
  document.querySelector("#profile-name").textContent = state.name;
  document.querySelector("#avatar").textContent = initials(state.name);
  document.querySelector("#top-avatar").textContent = initials(state.name);
  document.querySelector("#welcome-title").textContent = `Hola, ${state.name}. ¿Qué exploramos?`;
  elements.onlineFineTuning.checked = state.onlineFineTuning;
  document.querySelector("#online-state").textContent = state.onlineFineTuning
    ? "Activado · ajuste por turno"
    : "Desactivado · predeterminado";
}

function setModelSettings(temperature, topK) {
  state.modelSettings = {
    temperature: Math.min(
      modelConfig.temperature_max,
      Math.max(modelConfig.temperature_min, Number(temperature)),
    ),
    topK: Math.min(
      modelConfig.top_k_max,
      Math.max(modelConfig.top_k_min, Math.round(Number(topK))),
    ),
    contextWindow: state.modelSettings.contextWindow ?? modelConfig.context_window,
  };
  document.querySelector("#temperature-slider").value = String(state.modelSettings.temperature);
  document.querySelector("#top-k-slider").value = String(state.modelSettings.topK);
  document.querySelector("#temperature-value").value = state.modelSettings.temperature.toFixed(1);
  document.querySelector("#top-k-value").value = String(state.modelSettings.topK);
  persist();
}

function initials(name) {
  return [...(name.trim() || "U")][0].toLocaleUpperCase("es");
}

function showToast(message, isError = false) {
  clearTimeout(toastTimer);
  elements.toast.textContent = message;
  elements.toast.classList.toggle("error", isError);
  elements.toast.classList.add("show");
  toastTimer = setTimeout(() => elements.toast.classList.remove("show"), 3800);
}

function closeSidebar() {
  elements.sidebar.classList.remove("open");
}

function addMessage(role, content, thought = "") {
  const chat = currentChat();
  if (!chat) return;
  const message = { role, content };
  if (role === "assistant" && typeof thought === "string" && thought.trim()) {
    message.thought = thought.trim();
  }
  chat.messages.push(message);
  chat.updatedAt = Date.now();
  if (role === "user" && chat.title === "Nueva conversación") {
    chat.title = content.trim().replace(/\s+/g, " ").slice(0, 34);
    if (content.trim().length > 34) chat.title += "…";
  }
  persist();
  render();
}

async function sendMessage(rawText, { retry = false } = {}) {
  const message = rawText.trim();
  if (!message || elements.send.disabled) return;
  if (!currentChat()) createChat();
  if (!retry) addMessage("user", message);
  const chat = currentChat();
  const history = chat.messages
    .slice(0, -1)
    .filter(message =>
      (message.role === "user" || message.role === "assistant")
      && typeof message.content === "string"
    )
    .slice(-200);
  elements.send.disabled = true;
  elements.input.disabled = true;
  const pending = document.createElement("article");
  pending.className = "message assistant";
  pending.innerHTML = '<span class="message-avatar" aria-hidden="true">✳</span><div class="message-body"><div class="message-role">LLM Básico</div><span class="typing-dots" aria-label="Generando respuesta"><i></i><i></i><i></i></span></div>';
  elements.messageList.append(pending);
  elements.conversation.scrollTop = elements.conversation.scrollHeight;
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        history,
        temperature: state.modelSettings.temperature,
        top_k: state.modelSettings.topK,
        online_fine_tuning: state.onlineFineTuning,
      }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "No se pudo generar la respuesta.");
    pending.remove();
    addMessage("assistant", result.reply, result.thought || "");
    if (result.online_fine_tuned) {
      showToast("Ajuste online aplicado al turno. Los cambios viven durante esta sesión.");
    }
  } catch (error) {
    pending.remove();
    showToast(error.message || "No se pudo conectar con el modelo local.", true);
  } finally {
    elements.send.disabled = false;
    elements.input.disabled = false;
    elements.input.focus();
  }
}

function lastUserBefore(chat, index) {
  for (let cursor = index - 1; cursor >= 0; cursor -= 1) {
    if (chat.messages[cursor].role === "user") return chat.messages[cursor];
  }
  return null;
}

async function saveAsMemory(message, prompt, button) {
  if (!prompt || message.memorySaving) return;
  message.memorySaving = true;
  button.disabled = true;
  button.textContent = "Guardando…";
  try {
    const response = await fetch("/api/memory", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title: `Memoria - ${prompt.slice(0, 48)}`,
        prompt,
        response: message.content,
      }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "No se pudo guardar la nota.");
    message.memoryPdf = result.document;
    message.memoryPdf.download_url = result.download_url;
    delete message.memorySaving;
    persist();
    render();
    showToast(result.message);
    if (elements.controlPanel.classList.contains("open") && !elements.corpusContent.hidden) {
      await loadCorpus();
    }
  } catch (error) {
    delete message.memorySaving;
    render();
    showToast(error.message || "No se pudo guardar la memoria.", true);
  }
}

async function sendFeedback(message, prompt, reward, button) {
  if (!prompt || message.feedbackSubmitting || Number.isInteger(message.reward)) return;
  message.feedbackSubmitting = true;
  button.disabled = true;
  try {
    const response = await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        response: message.content,
        reward,
      }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "No se pudo registrar el feedback.");
    message.reward = reward;
    delete message.feedbackSubmitting;
    persist();
    render();
    showToast(result.message);
  } catch (error) {
    delete message.feedbackSubmitting;
    render();
    showToast(error.message || "No se pudo registrar el feedback.", true);
  }
}

async function retryLatestResponse() {
  const chat = currentChat();
  if (!chat || elements.send.disabled) return;
  let promptIndex = -1;
  let responseIndex = -1;
  for (let index = 0; index < chat.messages.length; index += 1) {
    if (chat.messages[index].role === "user") promptIndex = index;
    if (chat.messages[index].role === "assistant") responseIndex = index;
  }
  if (promptIndex < 0 || promptIndex < responseIndex) return;
  const prompt = chat.messages[promptIndex].content;
  await sendMessage(prompt, { retry: true });
}

async function mutateCorpus(endpoint, texto, fieldName, form) {
  const boton = form.querySelector("button[type=submit]");
  boton.disabled = true;
  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ [fieldName]: texto }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "No se pudo actualizar el corpus.");
    form.reset();
    showToast(result.message);
    await loadCorpus();
  } catch (error) {
    showToast(error.message || "No se pudo actualizar el corpus local.", true);
  } finally {
    boton.disabled = false;
  }
}

function clearAdminSession() {
  adminToken = null;
  document.querySelector("#admin-login-form").hidden = false;
  document.querySelector("#admin-password-form").hidden = true;
  document.querySelector("#admin-corpus-form").hidden = true;
  document.querySelector("#admin-corpus-status").textContent = "";
}

async function loadAdminCorpus() {
  const status = document.querySelector("#admin-corpus-status");
  try {
    const response = await fetch("/api/admin/corpus-maximo", {
      headers: { Authorization: `Bearer ${adminToken}` },
    });
    const result = await response.json();
    if (!response.ok) {
      if (response.status === 401) clearAdminSession();
      throw new Error(result.error || "No se pudo cargar el corpus máximo.");
    }
    document.querySelector("#admin-core-foundation").value =
      result.core_fundacional;
    document.querySelector("#admin-core-personality").value =
      result.core_personalidad;
    status.classList.remove("error");
    document.querySelector("#admin-login-form").hidden = true;
    document.querySelector("#admin-password-form").hidden = true;
    document.querySelector("#admin-corpus-form").hidden = false;
    status.textContent = "Sesión administrativa activa.";
  } catch (error) {
    status.textContent = error.message || "No se pudo cargar el corpus máximo.";
    status.classList.add("error");
  }
}

document.querySelector("#admin-login-form").addEventListener("submit", async event => {
  event.preventDefault();
  const form = event.currentTarget;
  const status = document.querySelector("#admin-corpus-status");
  const boton = form.querySelector("button[type=submit]");
  boton.disabled = true;
  status.classList.remove("error");
  status.textContent = "Verificando credenciales…";
  try {
    const response = await fetch("/api/admin/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        username: document.querySelector("#admin-username").value,
        password: document.querySelector("#admin-password").value,
      }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "No se pudo iniciar sesión.");
    adminToken = result.token;
    form.reset();
    form.querySelector("#admin-username").value = "admin";
    form.hidden = true;
    if (result.must_change_password) {
      document.querySelector("#admin-password-form").hidden = false;
      status.textContent = "Acceso correcto. Cambia la contraseña provisional para continuar.";
    } else {
      await loadAdminCorpus();
    }
  } catch (error) {
    status.classList.add("error");
    status.textContent = error.message || "No se pudo iniciar sesión.";
  } finally {
    boton.disabled = false;
  }
});

document.querySelector("#admin-password-form").addEventListener("submit", async event => {
  event.preventDefault();
  const form = event.currentTarget;
  const status = document.querySelector("#admin-corpus-status");
  const boton = form.querySelector("button[type=submit]");
  boton.disabled = true;
  status.classList.remove("error");
  status.textContent = "Actualizando contraseña…";
  try {
    const response = await fetch("/api/admin/password", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${adminToken}`,
      },
      body: JSON.stringify({
        new_password: document.querySelector("#admin-new-password").value,
      }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "No se pudo cambiar la contraseña.");
    form.reset();
    await loadAdminCorpus();
    status.textContent = `${result.message} ${status.textContent}`;
  } catch (error) {
    status.classList.add("error");
    status.textContent = error.message || "No se pudo cambiar la contraseña.";
  } finally {
    boton.disabled = false;
  }
});

document.querySelector("#admin-corpus-form").addEventListener("submit", async event => {
  event.preventDefault();
  const form = event.currentTarget;
  const status = document.querySelector("#admin-corpus-status");
  const boton = form.querySelector("button[type=submit]");
  boton.disabled = true;
  status.classList.remove("error");
  status.textContent = "Guardando corpus máximo y reentrenando…";
  try {
    const response = await fetch("/api/admin/corpus-maximo", {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${adminToken}`,
      },
      body: JSON.stringify({
        core_fundacional: document.querySelector("#admin-core-foundation").value,
        core_personalidad: document.querySelector("#admin-core-personality").value,
      }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "No se pudo guardar el corpus máximo.");
    status.textContent = result.message;
    await loadCorpus();
  } catch (error) {
    status.classList.add("error");
    status.textContent = error.message || "No se pudo guardar el corpus máximo.";
  } finally {
    boton.disabled = false;
  }
});

document.querySelector("#admin-logout").addEventListener("click", async () => {
  if (adminToken) {
    try {
      const response = await fetch("/api/admin/logout", {
        method: "POST",
        headers: { Authorization: `Bearer ${adminToken}` },
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "No se pudo cerrar sesión.");
      showToast(result.message);
    } catch (error) {
      showToast(error.message || "No se pudo cerrar sesión administrativa.", true);
    } finally {
      clearAdminSession();
    }
  } else {
    clearAdminSession();
  }
});

async function loadExamples() {
  try {
    const response = await fetch("/api/examples");
    if (!response.ok) throw new Error("No se pudieron cargar los PDFs de ejemplo.");
    const { examples } = await response.json();
    elements.exampleList.replaceChildren();
    for (const example of examples) {
      const item = document.createElement("div");
      item.className = "example-item";
      const icon = document.createElement("span");
      icon.className = "pdf-icon";
      icon.setAttribute("aria-hidden", "true");
      icon.textContent = "▤";
      const copy = document.createElement("div");
      copy.className = "example-copy";
      const title = document.createElement("span");
      title.className = "example-title";
      title.textContent = example.title;
      const description = document.createElement("span");
      description.className = "example-description";
      description.textContent = example.description;
      copy.append(title, description);
      const add = document.createElement("button");
      add.className = "example-add";
      add.type = "button";
      add.textContent = "Añadir";
      add.addEventListener("click", () => addExample(example.id, add));
      item.append(icon, copy, add);
      elements.exampleList.append(item);
    }
  } catch (error) {
    elements.exampleList.textContent = "No se pudieron cargar los ejemplos.";
    showToast(error.message, true);
  }
}

async function loadCorpus() {
  try {
    const [response, resumenResponse] = await Promise.all([
      fetch("/api/pdfs"),
      fetch("/api/corpus"),
    ]);
    const [result, resumenResult] = await Promise.all([
      response.json(),
      resumenResponse.json(),
    ]);
    if (!response.ok) throw new Error(result.error || "No se pudo leer el corpus local.");
    if (!resumenResponse.ok) {
      throw new Error(resumenResult.error || "No se pudo leer la jerarquía de corpus.");
    }
    document.querySelector("#corpus-maximum-meta").textContent =
      `${resumenResult.corpus.maximum_characters.toLocaleString("es")} caracteres · solo lectura`;
    document.querySelector("#corpus-local-meta").textContent =
      `${resumenResult.corpus.local_characters.toLocaleString("es")} caracteres · ${resumenResult.corpus.pdf_count} PDF`;
    elements.corpusList.replaceChildren();
    const documentos = result.pdfs || [];
    elements.corpusCount.textContent = String(
      documentos.filter(documento => documento.kind === "pdf").length,
    );
    if (documentos.length === 0) {
      const empty = document.createElement("div");
      empty.className = "corpus-empty";
      empty.textContent = "Aún no hay PDF incorporados. Añade uno desde aquí o desde la galería de ejemplos.";
      elements.corpusList.append(empty);
      return;
    }
    for (const documento of documentos) {
      const card = document.createElement("article");
      card.className = "corpus-card";
      const icon = document.createElement("span");
      icon.className = `corpus-card-icon${documento.kind === "text" ? " text-icon" : ""}`;
      icon.setAttribute("aria-hidden", "true");
      icon.textContent = documento.kind === "text" ? "¶" : "▤";
      const copy = document.createElement("div");
      copy.className = "corpus-card-copy";
      const title = document.createElement("span");
      title.className = "corpus-card-title";
      title.textContent = documento.filename;
      const metadata = document.createElement("span");
      metadata.className = "corpus-card-meta";
      metadata.textContent = documento.kind === "pdf"
        ? `${documento.pages} páginas · ${documento.characters.toLocaleString("es")} caracteres · ${new Date(documento.added_at).toLocaleDateString("es")}`
        : `${documento.characters.toLocaleString("es")} caracteres · texto guardado sin el PDF original`;
      const preview = document.createElement("button");
      preview.className = "corpus-preview-button";
      preview.type = "button";
      preview.textContent = "Vista previa";
      preview.addEventListener("click", () => openDocumentPreview(documento));
      card.append(icon, copy, preview);
      if (documento.kind === "pdf") {
        const remove = document.createElement("button");
        remove.className = "corpus-delete-button";
        remove.type = "button";
        remove.textContent = "Eliminar";
        remove.setAttribute("aria-label", `Eliminar ${documento.filename} del corpus local`);
        remove.addEventListener("click", () => deleteCorpusPdf(documento));
        card.append(remove);
      }
      copy.append(title, metadata);
      elements.corpusList.append(card);
    }
  } catch (error) {
    elements.corpusList.textContent = "No se pudo cargar la lista del corpus.";
    showToast(error.message || "No se pudo leer el corpus local.", true);
  }
}

async function deleteCorpusPdf(documento) {
  if (!window.confirm(`¿Eliminar «${documento.filename}» y su texto del corpus local?`)) return;
  try {
    const response = await fetch(`/api/pdfs/${encodeURIComponent(documento.id)}`, {
      method: "DELETE",
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "No se pudo eliminar el documento.");
    showToast(`${documento.filename} se eliminó y el modelo fue reentrenado.`);
    await loadCorpus();
  } catch (error) {
    showToast(error.message || "No se pudo eliminar el documento.", true);
  }
}

async function openDocumentPreview(documento) {
  try {
    const response = await fetch(`/api/pdfs/${encodeURIComponent(documento.id)}/preview`);
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "No se pudo abrir la vista previa.");
    document.querySelector("#preview-title").textContent = result.filename;
    document.querySelector("#preview-meta").textContent = result.kind === "pdf"
      ? `${result.pages} páginas · ${result.characters.toLocaleString("es")} caracteres`
      : `${result.text.length.toLocaleString("es")} caracteres · vista del texto del corpus`;
    document.querySelector("#extracted-text").textContent = result.text;
    document.querySelector(".extracted-text").hidden = result.kind !== "pdf";
    elements.textPreview.hidden = result.kind !== "text";
    elements.textPreview.textContent = result.kind === "text" ? result.text : "";
    elements.pdfFrame.hidden = result.kind !== "pdf";
    elements.pdfFrame.src = result.kind === "pdf"
      ? `/api/pdfs/${encodeURIComponent(documento.id)}.pdf`
      : "about:blank";
    elements.previewDialog.showModal();
  } catch (error) {
    showToast(error.message || "No se pudo abrir la vista previa.", true);
  }
}

function openControlPanel(panel) {
  const showingCorpus = panel === "corpus";
  elements.corpusContent.hidden = !showingCorpus;
  elements.modelContent.hidden = showingCorpus;
  document.querySelector("#control-title").textContent = showingCorpus
    ? "Corpus local"
    : "Configuración modelo";
  document.querySelector("#control-eyebrow").textContent = showingCorpus
    ? "PANEL DE CONTROL"
    : "MUESTREO DE TEXTO";
  elements.controlPanel.classList.add("open");
  elements.controlPanel.setAttribute("aria-hidden", "false");
  elements.controlBackdrop.classList.add("open");
  closeSidebar();
  if (showingCorpus) loadCorpus();
}

function closeControlPanel() {
  elements.controlPanel.classList.remove("open");
  elements.controlPanel.setAttribute("aria-hidden", "true");
  elements.controlBackdrop.classList.remove("open");
}

async function loadModelConfig() {
  try {
    const response = await fetch("/api/model-config");
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "No se pudo cargar la configuración.");
    modelConfig = result;
    const unidadContexto = modelConfig.tokenizer_type === "palabras"
      ? "tokens"
      : "caracteres";
    const tipoModelo = modelConfig.tokenizer_type === "palabras"
      ? "palabras"
      : "caracteres";
    document.querySelector("#model-runtime-label").textContent =
      `Modelo de ${tipoModelo} · local${modelConfig.quantized ? " · int8" : ""}`;
    const topKSlider = document.querySelector("#top-k-slider");
    topKSlider.max = String(modelConfig.top_k_max);
    document.querySelector("#top-k-range-max").textContent =
      `${modelConfig.top_k_max} candidatos`;
    const contextSlider = document.querySelector("#context-window-slider");
    contextSlider.min = String(modelConfig.context_window_min);
    contextSlider.max = String(modelConfig.context_window_max);
    contextSlider.step = "4";
    document.querySelector("#context-window-range-max").textContent =
      `${modelConfig.context_window_max} ${unidadContexto}`;
    document.querySelector("#context-window-unit-min").textContent =
      `${modelConfig.context_window_min} ${unidadContexto}`;
    document.querySelector("#context-window-description").textContent =
      `${unidadContexto[0].toLocaleUpperCase("es") + unidadContexto.slice(1)} previos que el modelo puede consultar.`;
    state.modelSettings.contextWindow = modelConfig.context_window;
    contextSlider.value = String(modelConfig.context_window);
    document.querySelector("#context-window-value").value =
      String(modelConfig.context_window);
    const salienceSlider = document.querySelector("#salience-threshold-slider");
    salienceSlider.min = String(modelConfig.salience_threshold_min);
    salienceSlider.max = String(modelConfig.salience_threshold_max);
    salienceSlider.value = String(modelConfig.salience_threshold);
    document.querySelector("#salience-threshold-value").value =
      Number(modelConfig.salience_threshold).toFixed(2);
    setModelSettings(state.modelSettings.temperature, state.modelSettings.topK);
  } catch (error) {
    showToast(error.message || "No se pudo cargar la configuración del modelo.", true);
  }
}

async function addPdf(file) {
  if (!file) return;
  if (file.size > 12 * 1024 * 1024) {
    showToast("El PDF supera el límite de 12 MB.", true);
    return;
  }
  showToast("Leyendo el PDF y actualizando el modelo…");
  try {
    const response = await fetch("/api/pdf", {
      method: "POST",
      headers: {
        "Content-Type": "application/pdf",
        "X-File-Name": encodeURIComponent(file.name),
      },
      body: file,
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "No se pudo leer el PDF.");
    showToast(result.message);
    elements.corpusCount.textContent = String(Number(elements.corpusCount.textContent) + 1);
    if (elements.controlPanel.classList.contains("open") && !elements.corpusContent.hidden) {
      await loadCorpus();
    }
  } catch (error) {
    showToast(error.message || "No se pudo añadir el PDF.", true);
  }
}

async function addExample(id, button) {
  button.disabled = true;
  button.textContent = "Añadiendo…";
  try {
    const response = await fetch(`/api/examples/${encodeURIComponent(id)}.pdf`);
    if (!response.ok) {
      const result = await response.json();
      throw new Error(result.error || "No se pudo descargar el ejemplo.");
    }
    const blob = await response.blob();
    await addPdf(new File([blob], `${id}.pdf`, { type: "application/pdf" }));
  } catch (error) {
    showToast(error.message || "No se pudo añadir el ejemplo.", true);
  } finally {
    button.disabled = false;
    button.textContent = "Añadir";
  }
}

elements.form.addEventListener("submit", event => {
  event.preventDefault();
  const text = elements.input.value;
  if (!text.trim()) return;
  elements.input.value = "";
  elements.input.style.height = "auto";
  sendMessage(text);
});
elements.retry.addEventListener("click", retryLatestResponse);
elements.corpusNoteForm.addEventListener("submit", event => {
  event.preventDefault();
  const texto = document.querySelector("#corpus-note").value;
  if (texto.trim()) mutateCorpus("/corpus/agregar", texto, "text", elements.corpusNoteForm);
});
elements.corpusRemoveForm.addEventListener("submit", event => {
  event.preventDefault();
  const texto = document.querySelector("#corpus-remove-text").value;
  if (texto.trim() && window.confirm("¿Quitar del corpus local este fragmento exacto?")) {
    mutateCorpus("/corpus/quitar", texto, "text", elements.corpusRemoveForm);
  }
});

elements.input.addEventListener("input", () => {
  elements.input.style.height = "auto";
  elements.input.style.height = `${Math.min(elements.input.scrollHeight, 170)}px`;
});

elements.input.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.form.requestSubmit();
  }
});

document.querySelector("#new-chat").addEventListener("click", createChat);
document.querySelector("#open-sidebar").addEventListener("click", () => elements.sidebar.classList.add("open"));
document.querySelector("#close-sidebar").addEventListener("click", closeSidebar);
elements.pdfInput.addEventListener("change", () => {
  addPdf(elements.pdfInput.files[0]);
  elements.pdfInput.value = "";
});
elements.onlineFineTuning.addEventListener("change", () => {
  state.onlineFineTuning = elements.onlineFineTuning.checked;
  persist();
  render();
  showToast(state.onlineFineTuning
    ? "Ajuste online activado para los próximos turnos."
    : "Ajuste online desactivado.");
});
document.querySelector("#open-corpus-panel").addEventListener("click", () => openControlPanel("corpus"));
document.querySelector("#open-model-panel").addEventListener("click", () => openControlPanel("model"));
document.querySelector("#top-model-settings").addEventListener("click", () => openControlPanel("model"));
document.querySelector("#composer-settings").addEventListener("click", () => openControlPanel("model"));
document.querySelector("#close-control-panel").addEventListener("click", closeControlPanel);
elements.controlBackdrop.addEventListener("click", closeControlPanel);
document.querySelector("#temperature-slider").addEventListener("input", event => {
  setModelSettings(event.currentTarget.value, state.modelSettings.topK);
});
document.querySelector("#top-k-slider").addEventListener("input", event => {
  setModelSettings(state.modelSettings.temperature, event.currentTarget.value);
});
document.querySelector("#context-window-slider").addEventListener("input", event => {
  const ventana = Number(event.currentTarget.value);
  state.modelSettings.contextWindow = ventana;
  document.querySelector("#context-window-value").value = String(ventana);
  const status = document.querySelector("#context-status");
  status.classList.remove("error");
  status.textContent = ventana === modelConfig.context_window
    ? "Esta ventana ya está aplicada."
    : "Pulsa «Aplicar y reentrenar» para reconstruir el modelo con este contexto.";
  persist();
});
document.querySelector("#apply-context-window").addEventListener("click", async event => {
  const boton = event.currentTarget;
  const status = document.querySelector("#context-status");
  const ventana = Number(document.querySelector("#context-window-slider").value);
  if (ventana === modelConfig.context_window) {
    status.classList.remove("error");
    status.textContent = "Esta ventana ya está aplicada.";
    return;
  }
  boton.disabled = true;
  status.classList.remove("error");
  status.textContent = "Reconstruyendo y reentrenando el modelo…";
  try {
    const response = await fetch("/api/model-config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ context_window: ventana }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "No se pudo cambiar la ventana.");
    modelConfig.context_window = result.context_window;
    state.modelSettings.contextWindow = result.context_window;
    persist();
    status.textContent = result.message;
  } catch (error) {
    status.classList.add("error");
    status.textContent = error.message || "No se pudo reentrenar el modelo.";
  } finally {
    boton.disabled = false;
  }
});
document.querySelector("#salience-threshold-slider").addEventListener("input", event => {
  document.querySelector("#salience-threshold-value").value =
    Number(event.currentTarget.value).toFixed(2);
  const status = document.querySelector("#salience-status");
  status.classList.remove("error");
  status.textContent = "Pulsa «Guardar umbral» para aplicar el nuevo criterio.";
});
document.querySelector("#apply-salience-threshold").addEventListener("click", async event => {
  const boton = event.currentTarget;
  const status = document.querySelector("#salience-status");
  const umbral = Number(document.querySelector("#salience-threshold-slider").value);
  boton.disabled = true;
  status.classList.remove("error");
  status.textContent = "Guardando umbral de saliencia…";
  try {
    const response = await fetch("/api/model-config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ salience_threshold: umbral }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "No se pudo guardar el umbral.");
    modelConfig.salience_threshold = result.salience_threshold;
    status.textContent =
      `Umbral ${Number(result.salience_threshold).toFixed(2)} guardado.`;
  } catch (error) {
    status.classList.add("error");
    status.textContent = error.message || "No se pudo guardar el umbral.";
  } finally {
    boton.disabled = false;
  }
});
document.querySelector("#reset-settings").addEventListener("click", () => {
  setModelSettings(modelConfig.temperature_default, modelConfig.top_k_default);
});
document.querySelector("#close-preview").addEventListener("click", () => elements.previewDialog.close());
elements.previewDialog.addEventListener("close", () => {
  elements.pdfFrame.src = "about:blank";
});
document.addEventListener("keydown", event => {
  if (event.key === "Escape") closeControlPanel();
});

function openNameDialog() {
  elements.nameInput.value = state.name === "Usuario" ? "" : state.name;
  elements.dialog.showModal();
  elements.nameInput.focus();
}
document.querySelector("#edit-name").addEventListener("click", openNameDialog);
document.querySelector("#top-profile").addEventListener("click", openNameDialog);
elements.nameForm.addEventListener("submit", event => {
  if (event.submitter?.value !== "save") return;
  event.preventDefault();
  state.name = elements.nameInput.value.trim().slice(0, 32) || "Usuario";
  persist();
  elements.dialog.close();
  render();
});
elements.nameForm.addEventListener("click", event => {
  if (event.target === elements.nameForm) elements.dialog.close();
});

document.querySelectorAll(".suggestion").forEach(button => {
  button.addEventListener("click", () => {
    elements.input.value = button.dataset.prompt || "";
    elements.form.requestSubmit();
  });
});

render();
loadExamples();
loadCorpus();
loadModelConfig();
if (state.name === "Usuario" && !localStorage.getItem(`${STORAGE_KEY}-name-prompted`)) {
  localStorage.setItem(`${STORAGE_KEY}-name-prompted`, "1");
  openNameDialog();
}
