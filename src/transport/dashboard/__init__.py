import asyncio, json, logging
from collections import deque
from datetime import datetime
from pathlib import Path

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from contextvars import ContextVar
from agent import Skill, bypass
from src.transport.web import WebTransport

log = logging.getLogger(__name__)

_HTML = (Path(__file__).parent / "web" / "index.html").read_text(encoding="utf-8")
_BUFFER_SIZE = 500

agent_context: ContextVar[str] = ContextVar("agent_id", default="main")


class _LogHandler(logging.Handler):
    _instances: dict[str, "DashboardTransport"] = {}

    def _category(self, name: str) -> str:
        if name.startswith("src.memory") or name.startswith("memory"):
            return "memory"
        if name.startswith("aiogram") or name.startswith("src.transport") or name.startswith("httpx") or name.startswith("uvicorn") or name.startswith("google_genai") or name.startswith("sentence_transformers") or name.startswith("huggingface_hub"):
            return "transport"
        return "agent"

    def emit(self, record: logging.LogRecord) -> None:
        try:
            agent_id = agent_context.get()
            transport = self._instances.get(agent_id)
            if not transport:
                return
            category = self._category(record.name)
            record.agent_id = agent_id
            text = self.format(record)
            transport._emit({"type": "log", "category": category, "level": record.levelname, "text": text})
        except Exception:
            self.handleError(record)


class DashboardSkill(Skill):
    def __init__(self, transport: "DashboardTransport"):
        self.transport = transport
        super().__init__()

    @bypass("dashboard", "Ссылка на веб-дашборд", standalone=True)
    def dashboard_command(self, args: str) -> str:
        t = self.transport
        path = f"/{t.agent.id}/"
        lines = [f"🖥 http://localhost:{t._port}{path}"]
        if t._tunnel_url:
            lines.append(f"🌐 {t._tunnel_url}{path}")
        return "\n".join(lines)


class DashboardTransport(WebTransport):
    _log_handler: _LogHandler | None = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._clients: set[WebSocket] = set()
        self._buffer: deque = deque(maxlen=_BUFFER_SIZE)
        self._queue: asyncio.Queue | None = None
        self._skill = DashboardSkill(self)

    def get_skills(self):
        return [self._skill]

    def set_agent(self, agent):
        super().set_agent(agent)
        self.register_route("get", "/", self._page)
        self.register_websocket("/ws", self._ws)

        if DashboardTransport._log_handler is None:
            handler = _LogHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s [%(agent_id)s] %(name)s - %(levelname)s - %(message)s"))
            logging.getLogger().addHandler(handler)
            DashboardTransport._log_handler = handler

        _LogHandler._instances[agent.id] = self
        agent_context.set(agent.id)

    async def _page(self):
        return HTMLResponse(_HTML.replace("__AGENT_ID__", self.agent.id))

    async def _ws(self, websocket: WebSocket):
        await websocket.accept()
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=2000)
            asyncio.create_task(self._broadcaster())
        for event in self._buffer:
            await websocket.send_text(json.dumps(event, ensure_ascii=False))
        self._clients.add(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                try:
                    msg = json.loads(data)
                    if msg.get("type") == "user_message" and msg.get("text", "").strip():
                        text = msg["text"].strip()
                        self._emit({"type": "chat", "role": "user", "text": text})
                        await self.process_message(content_parts=[{"type": "text", "text": text}])
                except Exception:
                    pass
        except WebSocketDisconnect:
            pass
        finally:
            self._clients.discard(websocket)


    def _emit(self, event: dict):
        event["ts"] = datetime.now().strftime("%H:%M:%S")
        self._buffer.append(event)
        if self._queue:
            try:
                self._queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

    async def _broadcaster(self):
        while True:
            event = await self._queue.get()
            if not self._clients:
                continue
            data = json.dumps(event, ensure_ascii=False)
            dead = set()
            for client in list(self._clients):
                try:
                    await client.send_text(data)
                except Exception:
                    dead.add(client)
            self._clients -= dead

    # --- BaseTransport interface ---

    async def send_message(self, text: str, stream_id=None, final: bool = True):
        self._emit({"type": "chat", "role": "assistant", "text": text, "chat_id": stream_id})

    async def send_thinking(self, text: str, stream_id=None, final: bool = False):
        self._emit({"type": "collapsible", "title": "[think]", "text": text, "collapsible_id": stream_id})

    async def send_system_prompt(self, text: str):
        label = text.split("\n")[0].strip("[] ")[:40]
        self._emit({"type": "collapsible", "title": f"[sys] {label}", "text": text})

    async def on_tool_call(self, name: str, args: dict):
        lines = "\n".join(f"  {k}: {v}" for k, v in args.items())
        text = f"[{name}]\n{lines}" if lines else f"[{name}]"
        self._emit({"type": "collapsible", "title": f"[>] {name}", "text": text})

    async def on_tool_result(self, name: str, result):
        if isinstance(result, dict):
            parts = [
                f"<binary {len(v)} bytes>" if isinstance(v, (bytes, bytearray)) else f"[{k}]\n{v}"
                for k, v in result.items() if v not in (None, "", [], {})
            ]
            text = "\n".join(parts) if parts else "(пусто)"
        elif isinstance(result, (bytes, bytearray)):
            text = f"<binary {len(result)} bytes>"
        else:
            text = str(result)
        self._emit({"type": "collapsible", "title": f"[<] {name}", "text": text})

    async def inject_message(self, text: str):
        self._emit({"type": "chat", "role": "user", "text": f"[→] {text}"})
