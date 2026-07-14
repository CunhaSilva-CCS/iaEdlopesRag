const form = document.getElementById("chat-form");
const messagesEl = document.getElementById("messages");
const container = document.getElementById("chat-container");
const textarea = document.getElementById("pergunta");
const sendBtn = document.getElementById("send-btn");
const btnClear = document.getElementById("btn-clear");
const suggestions = document.getElementById("suggestions");
let preparingHits = 0;

if (window.marked && typeof window.marked.setOptions === "function") {
  window.marked.setOptions({ breaks: true });
}

function submitForm() {
  if (typeof form.requestSubmit === "function") {
    form.requestSubmit();
    return;
  }
  form.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
}

// ── Auto-resize textarea ──────────────────────────────
textarea.addEventListener("input", () => {
  textarea.style.height = "auto";
  textarea.style.height = Math.min(textarea.scrollHeight, 150) + "px";
});

// ── Send on Enter, newline on Shift+Enter ─────────────
textarea.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    submitForm();
  }
});

// ── Suggestion chips ──────────────────────────────────
document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    textarea.value = chip.textContent.trim();
    textarea.dispatchEvent(new Event("input"));
    submitForm();
  });
});

// ── Clear conversation ────────────────────────────────
btnClear.addEventListener("click", () => {
  messagesEl.innerHTML = `
    <div class="message ai">
      <div class="avatar">ED</div>
      <div class="bubble">
        <p>Conversa reiniciada. Estou disponível para apoiar consultas operacionais com base na documentação oficial da Edlopes Transportes.</p>
      </div>
    </div>`;
  suggestions.style.display = "";
  textarea.focus();
});

// ── Helpers ───────────────────────────────────────────
function scrollToBottom() {
  container.scrollTo({ top: container.scrollHeight, behavior: "smooth" });
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function addMessage(content, role) {
  const wrap = document.createElement("div");
  wrap.className = `message ${role}`;

  const avatarLabel = role === "user" ? "Você" : "ED";
  const bubbleHtml =
    role === "ai"
      ? renderAssistantMessage(content)
      : `<p>${escapeHtml(content)}</p>`;

  wrap.innerHTML = `
    <div class="avatar">${avatarLabel}</div>
    <div class="bubble">${bubbleHtml}</div>
  `;

  messagesEl.appendChild(wrap);
  scrollToBottom();
  return wrap;
}

function renderAssistantMessage(content) {
  if (window.marked && typeof window.marked.parse === "function") {
    try {
      return window.marked.parse(content);
    } catch (error) {
      console.error("Falha ao renderizar markdown:", error);
    }
  }
  return `<p>${escapeHtml(content).replace(/\n/g, "<br>")}</p>`;
}

function showTyping() {
  const wrap = document.createElement("div");
  wrap.className = "message ai";
  wrap.id = "typing";
  wrap.innerHTML = `
    <div class="avatar">ED</div>
    <div class="bubble">
      <div class="typing"><span></span><span></span><span></span></div>
    </div>
  `;
  messagesEl.appendChild(wrap);
  scrollToBottom();
}

function hideTyping() {
  document.getElementById("typing")?.remove();
}

// ── Submit ────────────────────────────────────────────
form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const pergunta = textarea.value.trim();
  if (!pergunta) return;

  suggestions.style.display = "none";

  textarea.value = "";
  textarea.style.height = "auto";
  sendBtn.disabled = true;

  addMessage(pergunta, "user");
  showTyping();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pergunta }),
    });

    const rawBody = await res.text();
    let data = {};
    try {
      data = rawBody ? JSON.parse(rawBody) : {};
    } catch {
      data = { erro: rawBody || "Resposta inválida do servidor." };
    }
    hideTyping();

    if (res.ok) {
      if (res.status === 202 || data.status === "preparando") {
        preparingHits += 1;
        if (preparingHits >= 5) {
          addMessage(
            "A base ainda esta em preparacao ha algum tempo. Tente novamente em 1-2 minutos. Se persistir, reinicie o servico no Render.",
            "ai",
          );
          return;
        }
        addMessage(
          data.erro ||
            "A base esta sendo preparada. Tente novamente em instantes.",
          "ai",
        );
        return;
      }
      preparingHits = 0;
      addMessage(data.resposta || "Não foi possível gerar a resposta.", "ai");
    } else {
      preparingHits = 0;
      addMessage(`⚠️ ${data.erro || "Erro ao obter resposta."}`, "ai");
    }
  } catch (error) {
    console.error("Falha na chamada do chat:", error);
    hideTyping();
    addMessage(
      "⚠️ Falha de conexão. Verifique se o servidor está rodando.",
      "ai",
    );
  } finally {
    sendBtn.disabled = false;
    textarea.focus();
  }
});
