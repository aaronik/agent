import json
import types
import sys
from types import SimpleNamespace
import pytest

import src.tools as tools


class DummyResponse:
    def __init__(self, json_obj=None, text="", raise_exc=False):
        self._json = json_obj or {}
        self.text = text
        self._raise = raise_exc

    def raise_for_status(self):
        if self._raise:
            raise Exception("http error")

    def json(self):
        return self._json


def test_spawn_ollama_choice_message_content(monkeypatch):
    monkeypatch.setenv("OLLAMA_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "test-model")

    data = {"choices": [{"message": {"content": "hello from ollama"}}]}
    resp = DummyResponse(json_obj=data, text=json.dumps(data))

    def fake_post(url, json, timeout):
        assert url.endswith("/v1/chat/completions")
        return resp

    monkeypatch.setattr(tools.requests, "post", fake_post)

    out = tools.spawn("Do something small")
    assert "hello from ollama" in out


def test_spawn_ollama_choice_content_field(monkeypatch):
    monkeypatch.setenv("OLLAMA_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "test-model")

    data = {"choices": [{"content": "choice-content"}]}
    resp = DummyResponse(json_obj=data, text=json.dumps(data))

    def fake_post(url, json, timeout):
        return resp

    monkeypatch.setattr(tools.requests, "post", fake_post)

    out = tools.spawn("Task")
    assert "choice-content" in out


def test_spawn_ollama_top_level_text(monkeypatch):
    monkeypatch.setenv("OLLAMA_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "test-model")

    data = {"text": "top-level text"}
    resp = DummyResponse(json_obj=data, text=json.dumps(data))

    def fake_post(url, json, timeout):
        return resp

    monkeypatch.setattr(tools.requests, "post", fake_post)

    out = tools.spawn("Another task")
    assert "top-level text" in out


def test_spawn_falls_back_to_aisuite(monkeypatch):
    # Ensure Ollama env vars are not set so default aisuite behavior is used
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    # Fake Message class that matches aisuite.Message API used by spawn
    class FakeMessage:
        def __init__(self, role, content):
            self.role = role
            self.content = content

        def model_dump(self):
            return {"role": self.role, "content": self.content}

    # Fake Client that exposes chat.completions.create
    class FakeClient:
        def __init__(self):
            def create(model, messages, tools, max_turns):
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ai-suite result"))])

            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    fake_aisuite = types.SimpleNamespace(Client=FakeClient, Message=FakeMessage)
    monkeypatch.setitem(sys.modules, 'aisuite', fake_aisuite)

    out = tools.spawn("Fallback task")
    assert "ai-suite result" in out
