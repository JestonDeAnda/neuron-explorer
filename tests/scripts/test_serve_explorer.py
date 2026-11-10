from fastapi.testclient import TestClient

from scripts import serve_explorer


def test_console_port_builds_matching_local_only_app(tmp_path, monkeypatch):
    captured = {}

    def fake_run(app, *, host, port):
        captured.update(app=app, host=host, port=port)

    monkeypatch.setattr("uvicorn.run", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "neuron-explorer",
            "--root",
            str(tmp_path),
            "--port",
            "9123",
        ],
    )

    assert serve_explorer.main() == 0
    assert captured["port"] == 9123
    with TestClient(captured["app"], base_url="http://localhost:9123") as client:
        assert client.get("/api/status").status_code == 200
        assert (
            client.get("/api/status", headers={"host": "localhost:8791"}).status_code
            == 403
        )
