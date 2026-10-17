const STORAGE_KEY = "llm-basico-web-v1";
const DEFAULT_STATE = { name: "Usuario", chats: [], activeChatId: null };

const elements = {
  chatList: document.querySelector("#chat-list"),
  exampleList: document.querySelector("#example-list"),
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
      return { name: saved.name.slice(0, 32) || "Usuario", chats, activeChatId };
    }
  } catch (error) {
    console.warn("No se pudo recuperar el estado guardado del navegador.", error);
  }
  return { ...DEFAULT_STATE, chats: [] };
}

const state = loadState();
let toastTimer;

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
  if (!hasMessages) return;
  for (const message of chat.messages) {
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
    const text = document.createElement("div");
    text.textContent = message.content;
    body.append(text);
    if (message.role === "user") row.append(body, avatar);
    else row.append(avatar, body);
    elements.messageList.append(row);
  }
  elements.conversation.scrollTop = elements.conversation.scrollHeight;
}

function render() {
  renderChatList();
  renderMessages();
  document.querySelector("#profile-name").textContent = state.name;
  document.querySelector("#avatar").textContent = initials(state.name);
  document.querySelector("#top-avatar").textContent = initials(state.name);
  document.querySelector("#welcome-title").textContent = `Hola, ${state.name}. ¿Qué exploramos?`;
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

function addMessage(role, content) {
  const chat = currentChat();
  if (!chat) return;
  chat.messages.push({ role, content });
  chat.updatedAt = Date.now();
  if (role === "user" && chat.title === "Nueva conversación") {
    chat.title = content.trim().replace(/\s+/g, " ").slice(0, 34);
    if (content.trim().length > 34) chat.title += "…";
  }
  persist();
  render();
}

async function sendMessage(rawText) {
  const message = rawText.trim();
  if (!message || elements.send.disabled) return;
  if (!currentChat()) createChat();
  addMessage("user", message);
  const chat = currentChat();
  const history = chat.messages.slice(0, -1).slice(-12);
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
        temperature: Number(document.querySelector("#temperature").value),
        top_k: Number(document.querySelector("#top-k").value),
      }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "No se pudo generar la respuesta.");
    pending.remove();
    addMessage("assistant", result.reply);
  } catch (error) {
    pending.remove();
    showToast(error.message || "No se pudo conectar con el modelo local.", true);
  } finally {
    elements.send.disabled = false;
    elements.input.disabled = false;
    elements.input.focus();
  }
}

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
      headers: { "Content-Type": "application/pdf" },
      body: file,
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "No se pudo leer el PDF.");
    showToast(result.message);
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
if (state.name === "Usuario" && !localStorage.getItem(`${STORAGE_KEY}-name-prompted`)) {
  localStorage.setItem(`${STORAGE_KEY}-name-prompted`, "1");
  openNameDialog();
}
