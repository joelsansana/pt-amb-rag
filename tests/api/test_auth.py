import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from pt_amb_rag.api.auth import require_token

TOKEN = "s3cret-token"


@pytest.fixture
def client():
    app = FastAPI()

    @app.get("/protected", dependencies=[Depends(require_token)])
    async def protected():
        return {"ok": True}

    return TestClient(app)


def test_no_token_configured_allows_access(client, monkeypatch):
    monkeypatch.delenv("API_TOKEN", raising=False)
    assert client.get("/protected").status_code == 200


def test_missing_header_returns_401(client, monkeypatch):
    monkeypatch.setenv("API_TOKEN", TOKEN)
    r = client.get("/protected")
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"


def test_wrong_scheme_returns_401(client, monkeypatch):
    monkeypatch.setenv("API_TOKEN", TOKEN)
    r = client.get("/protected", headers={"Authorization": f"Basic {TOKEN}"})
    assert r.status_code == 401


def test_empty_bearer_returns_401(client, monkeypatch):
    monkeypatch.setenv("API_TOKEN", TOKEN)
    r = client.get("/protected", headers={"Authorization": "Bearer "})
    assert r.status_code == 401


def test_wrong_token_returns_401(client, monkeypatch):
    monkeypatch.setenv("API_TOKEN", TOKEN)
    r = client.get("/protected", headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401
    assert r.json()["detail"] == "Invalid token"


def test_correct_token_returns_200(client, monkeypatch):
    monkeypatch.setenv("API_TOKEN", TOKEN)
    r = client.get("/protected", headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 200


def test_scheme_is_case_insensitive(client, monkeypatch):
    monkeypatch.setenv("API_TOKEN", TOKEN)
    r = client.get("/protected", headers={"Authorization": f"bearer {TOKEN}"})
    assert r.status_code == 200


def test_non_ascii_token_does_not_crash(client, monkeypatch):
    monkeypatch.setenv("API_TOKEN", TOKEN)
    r = client.get(
        "/protected",
        headers={"Authorization": "Bearer tökén".encode("latin-1")},
    )
    assert r.status_code == 401
