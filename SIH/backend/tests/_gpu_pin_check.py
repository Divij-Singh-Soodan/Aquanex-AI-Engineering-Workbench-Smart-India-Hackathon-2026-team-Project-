"""One-shot check for localhost GPU pinning. Not part of the suite."""
import asyncio

from app.config import EDGE_LAPTOP_CONTEXT_WINDOW, settings
from app.llm.engine import FoundationModelEngine, OLLAMA_GPU_OPTIONS, ollama_request_options


def test_localhost_gpu_pin_contract():
    assert settings.OLLAMA_HOST == "http://127.0.0.1:11434"
    assert settings.API_BASE_URL == "http://127.0.0.1:8000"
    assert settings.MODEL_TIER == "EDGE_LAPTOP_8B"
    assert settings.MAX_MODEL_LEN == 4096
    assert settings.active_model_profile["context_window"] == 4096
    assert EDGE_LAPTOP_CONTEXT_WINDOW == 4096
    opts = ollama_request_options()
    assert opts["num_gpu"] == 99 and opts["num_ctx"] == 4096
    assert OLLAMA_GPU_OPTIONS == {"num_gpu": 99, "num_ctx": 4096}

    class FakeResponse:
        def __init__(self, status, payload, text=""):
            self.status_code = status
            self._payload = payload
            self.text = text

        def json(self):
            return self._payload

    async def _run():
        engine = FoundationModelEngine()
        captured = {}

        async def fake_post(url, json):
            captured.setdefault("calls", []).append((url, json))
            return FakeResponse(200, {"message": {"content": "hello from gpu"}})

        class Client:
            post = staticmethod(fake_post)

        res = await engine._post_chat(
            Client(),
            "http://127.0.0.1:11434/v1",
            "ollama",
            {
                "model": "qwen2.5:3b",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.2,
                "max_tokens": 128,
            },
        )
        url, body = captured["calls"][0]
        assert url == "http://127.0.0.1:11434/api/chat", url
        assert body["options"]["num_gpu"] == 99
        assert body["options"]["num_ctx"] == 4096
        assert body["stream"] is False
        assert engine._content_from_chat_response(res) == "hello from gpu"

        async def missing_native(url, json):
            captured.setdefault("calls2", []).append((url, json))
            if url.endswith("/api/chat"):
                return FakeResponse(404, {}, text="404 page not found")
            return FakeResponse(200, {"choices": [{"message": {"content": "compat"}}]})

        class Client2:
            post = staticmethod(missing_native)

        res2 = await engine._post_chat(
            Client2(),
            "http://127.0.0.1:11434/v1",
            "ollama",
            {
                "model": "qwen2.5:3b",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.1,
                "max_tokens": 64,
            },
        )
        assert captured["calls2"][1][0].endswith("/v1/chat/completions")
        assert captured["calls2"][1][1]["options"] == {"num_gpu": 99, "num_ctx": 4096}
        assert engine._content_from_chat_response(res2) == "compat"

    asyncio.run(_run())
