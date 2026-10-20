import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

TOKEN = "s3cret-token"

PAYLOAD = dict(
    regulation="dl_150_2015",
    part="article",
    article_num=26,
    diploma="Decreto-Lei n.º 150/2015",
    eli="https://data.dre.pt/eli/dec-lei/150/2015/08/05/p/dre/pt/html",
    header="Artigo 26.º — Decreto-Lei n.º 150/2015",
    text="O operador é responsável por...",
    content_hash="abc123",
)


def parse_sse(text: str) -> list[dict]:
    events = []
    for block in text.replace("\r\n", "\n").strip().split("\n\n"):
        ev = {}
        for line in block.split("\n"):
            key, _, value = line.partition(": ")
            ev[key] = value
        events.append(ev)
    return events


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("API_TOKEN", raising=False)
    import pt_amb_rag.api.main as main

    def fake_ask(question, **kwargs):
        return {
            "answer": f"answer to: {question}",
            "sources": [{"score": 0.9, **PAYLOAD}],  # flat, like the real ask()
        }

    monkeypatch.setattr(main, "ask", fake_ask)
    monkeypatch.setattr(main, "embed", lambda texts: [[0.0] * 3 for _ in texts])
    monkeypatch.setattr(
        main,
        "search",
        lambda emb, **kw: [
            SimpleNamespace(score=0.9, payload=PAYLOAD)
        ],  # like ScoredPoint
    )
    return TestClient(main.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ask_streams_answer_sources_done(client):
    r = client.post("/ask", json={"question": "What does Article 26 say?"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")

    events = parse_sse(r.text)
    assert [e["event"] for e in events] == ["answer", "sources", "done"]
    assert events[0]["data"] == "answer to: What does Article 26 say?"

    sources = json.loads(events[1]["data"])
    assert sources[0]["article_num"] == 26
    assert sources[0]["content_hash"] == "abc123"
    assert "latency_ms" in json.loads(events[2]["data"])


def test_ask_emits_error_event_on_failure(client, monkeypatch):
    import pt_amb_rag.api.main as main

    def boom(question, **kwargs):
        raise RuntimeError("llm down")

    monkeypatch.setattr(main, "ask", boom)
    events = parse_sse(client.post("/ask", json={"question": "q"}).text)
    assert [e["event"] for e in events] == ["error"]
    assert "llm down" in events[0]["data"]


def test_ask_validates_request(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert (
        client.post("/ask", json={"question": "q", "k": 99}).status_code == 422
    )


def test_search_returns_results(client):
    r = client.post("/search", json={"question": "q", "k": 3})
    assert r.status_code == 200
    assert (
        r.json()["results"][0]["header"]
        == "Artigo 26.º — Decreto-Lei n.º 150/2015"
    )


@pytest.mark.parametrize("path", ["/ask", "/search"])
def test_protected_endpoints_require_token(client, monkeypatch, path):
    monkeypatch.setenv("API_TOKEN", TOKEN)
    body = {"question": "q"}
    assert client.post(path, json=body).status_code == 401
    r = client.post(
        path, json=body, headers={"Authorization": f"Bearer {TOKEN}"}
    )
    assert r.status_code == 200


def test_health_is_public(client, monkeypatch):
    monkeypatch.setenv("API_TOKEN", TOKEN)
    assert client.get("/health").status_code == 200
