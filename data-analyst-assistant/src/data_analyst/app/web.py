"""A minimal, dependency-free browser chat UI for manually trying out the
assistant - one self-contained HTML page (inline CSS/JS, no build step, no
static file serving) mounted at `GET /` in api.py. Talks to the same
`POST /chat` JSON endpoint any other client would use.
"""
from __future__ import annotations

CHAT_PAGE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>Data Analyst Assistant</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; height: 100dvh; display: flex; flex-direction: column;
    font: 16px/1.4 system-ui, -apple-system, sans-serif;
    background: #f5f5f7; color: #1a1a1a;
  }
  header {
    padding: 0.9rem 1rem; background: #202124; color: #fff;
    font-weight: 600; flex: none;
  }
  header small { display: block; font-weight: 400; opacity: 0.7; font-size: 0.75rem; margin-top: 2px; }
  #log {
    flex: 1; overflow-y: auto; padding: 1rem;
    display: flex; flex-direction: column; gap: 0.6rem;
  }
  .msg { max-width: 85%; padding: 0.6rem 0.85rem; border-radius: 1rem; white-space: pre-wrap; word-wrap: break-word; }
  .user { align-self: flex-end; background: #2563eb; color: #fff; border-bottom-right-radius: 0.25rem; }
  .assistant { align-self: flex-start; background: #fff; border: 1px solid #e2e2e2; border-bottom-left-radius: 0.25rem; }
  .assistant.clarifying { border-color: #d97706; background: #fffbeb; }
  .assistant.error { border-color: #dc2626; background: #fef2f2; }
  .meta { align-self: flex-start; font-size: 0.75rem; opacity: 0.55; padding: 0 0.2rem; }
  form {
    flex: none; display: flex; gap: 0.5rem; padding: 0.75rem;
    background: #fff; border-top: 1px solid #e2e2e2;
    padding-bottom: max(0.75rem, env(safe-area-inset-bottom));
  }
  input {
    flex: 1; padding: 0.75rem 0.9rem; border-radius: 1.5rem;
    border: 1px solid #d0d0d0; font-size: 16px; outline: none;
  }
  input:focus { border-color: #2563eb; }
  button {
    padding: 0 1.25rem; border-radius: 1.5rem; border: none;
    background: #2563eb; color: #fff; font-size: 1rem; font-weight: 600;
  }
  button:disabled { background: #9ab4ee; }
</style>
</head>
<body>
<header>
  Data Analyst Assistant
  <small id="thread-label">new conversation</small>
</header>
<div id="log"></div>
<form id="form">
  <input id="input" type="text" placeholder="Ask about the data..." autocomplete="off" autofocus>
  <button id="send" type="submit">Send</button>
</form>
<script>
  let threadId = null;
  const log = document.getElementById("log");
  const form = document.getElementById("form");
  const input = document.getElementById("input");
  const sendBtn = document.getElementById("send");
  const threadLabel = document.getElementById("thread-label");

  function addMessage(text, cls) {
    const el = document.createElement("div");
    el.className = "msg " + cls;
    el.textContent = text;
    log.appendChild(el);
    log.scrollTop = log.scrollHeight;
    return el;
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const message = input.value.trim();
    if (!message) return;

    addMessage(message, "user");
    input.value = "";
    input.disabled = true;
    sendBtn.disabled = true;
    const pending = addMessage("...", "assistant");

    try {
      const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, thread_id: threadId }),
      });
      if (!response.ok) {
        throw new Error("HTTP " + response.status);
      }
      const data = await response.json();
      threadId = data.thread_id;
      threadLabel.textContent = "thread " + threadId.slice(0, 8);
      pending.textContent = data.reply;
      pending.className = "msg assistant" + (data.status === "clarification_needed" ? " clarifying" : "");
    } catch (err) {
      pending.textContent = "Error: " + err.message;
      pending.className = "msg assistant error";
    } finally {
      input.disabled = false;
      sendBtn.disabled = false;
      input.focus();
    }
  });
</script>
</body>
</html>
"""
