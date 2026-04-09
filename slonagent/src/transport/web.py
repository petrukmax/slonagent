import asyncio, base64, contextlib, hashlib, logging

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import Response
from src.transport.base import BaseTransport

log = logging.getLogger(__name__)


async def start_tunnel(port: int, subdomain: str, sish_domain: str, sish_port: int, sish_key: str):
    """Start SSH tunnel via sish using asyncssh. Returns (public_url, connection)."""
    import asyncssh
    key = asyncssh.import_private_key(sish_key)
    conn = await asyncssh.connect(
        sish_domain, sish_port, known_hosts=None, client_keys=[key], username="tunnel",
    )
    await conn.forward_remote_port(subdomain, 80, "localhost", port)
    url = f"https://{subdomain}.{sish_domain}:8443"
    log.info("[tunnel] %s -> localhost:%d", url, port)
    return url, conn


class WebTransport(BaseTransport):
    """Base for transports that serve web pages on a shared HTTP server.

    Class-level: lazily starts one uvicorn server per process.
    Instance-level: each instance registers routes under /{agent_id}/.
    """

    _app: FastAPI | None = None
    _server_task: asyncio.Task | None = None
    _port: int = 8765
    _tunnel_conn = None
    _tunnel_url: str | None = None
    _sish_domain: str = ""
    _sish_port: int = 2222
    _sish_key: str = ""
    _password_hash: str = ""
    _cookie_name: str = "slon_auth"

    def __init__(self, port: int = 8765, sish_domain: str = "", sish_port: int = 2222, sish_key: str = "", password_hash: str = ""):
        super().__init__()
        WebTransport._port = port
        if sish_domain:
            WebTransport._sish_domain = sish_domain
            WebTransport._sish_port = sish_port
            WebTransport._sish_key = sish_key
        if password_hash:
            WebTransport._password_hash = password_hash

    @classmethod
    def _ensure_server(cls):
        if cls._app is not None:
            return
        cls._app = FastAPI()

        if cls._password_hash:
            _REALM = "SlonAgent"
            _401 = Response(status_code=401, headers={"WWW-Authenticate": f'Basic realm="{_REALM}"'})

            @cls._app.middleware("http")
            async def auth_middleware(request: Request, call_next):
                auth = request.headers.get("authorization", "")
                if auth.startswith("Basic "):
                    try:
                        decoded = base64.b64decode(auth[6:]).decode()
                        password = decoded.split(":", 1)[1]
                        if hashlib.sha256(password.encode()).hexdigest() == cls._password_hash:
                            return await call_next(request)
                    except Exception:
                        pass
                return _401

        @cls._app.get("/")
        async def root():
            from fastapi.responses import RedirectResponse
            return RedirectResponse("/main/")

        async def _run():
            import uvicorn
            config = uvicorn.Config(
                cls._app,
                host="0.0.0.0",
                port=cls._port,
                ws="wsproto",
                log_config=None,
            )
            server = uvicorn.Server(config)
            server.install_signal_handlers = lambda: None
            server.capture_signals = lambda: contextlib.nullcontext()
            log.info("WebTransport server: http://localhost:%d", cls._port)
            await server.serve()

        logging.getLogger("asyncssh").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        cls._server_task = asyncio.create_task(_run())

        if cls._sish_domain:
            import random, string
            subdomain = "web-" + "".join(random.choices(string.ascii_lowercase, k=4))
            async def _tunnel():
                try:
                    url, cls._tunnel_conn = await start_tunnel(
                        cls._port, subdomain, cls._sish_domain, cls._sish_port, cls._sish_key,
                    )
                    cls._tunnel_url = url
                except Exception as e:
                    log.warning("Tunnel failed: %s", e)
            asyncio.create_task(_tunnel())

    def register_route(self, method: str, path: str, handler):
        self._ensure_server()
        full_path = f"/{self.agent.id}{path}"
        getattr(self._app, method)(full_path)(handler)

    def register_websocket(self, path: str, handler):
        self._ensure_server()
        full_path = f"/{self.agent.id}{path}"
        self._app.websocket(full_path)(handler)
