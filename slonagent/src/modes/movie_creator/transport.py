"""Web transport — mirrors agent events to MovieServer web UI."""
from src.transport.base import BaseTransport
from src.modes.movie_creator.server import MovieServer


class WebTransport(BaseTransport):

    def __init__(self, server: MovieServer):
        super().__init__()
        self.server = server
        self.server.on_chat(lambda text: self.process_message([{"type": "text", "text": text}]))

    async def send_message(self, text: str, stream_id=None, final: bool = True):
        await self.server.send("message", role="assistant", text=text, stream_id=stream_id)

    async def send_thinking(self, text: str, stream_id=None, final: bool = False):
        await self.server.send("message", role="thinking", text=text, stream_id=stream_id, final=final)

    async def on_tool_call(self, name: str, args: dict):
        await self.server.send("tool_call", name=name)

    async def send_processing(self, active: bool):
        await self.server.send("processing" if active else "processing_done")

    async def inject_message(self, text: str):
        await self.server.send("message", role="user", text=text)
