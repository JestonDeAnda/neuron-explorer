from types import SimpleNamespace

import pytest

from neuron_explorer.explorer import build_connections


@pytest.mark.parametrize("cuda_available", [True, False])
def test_builder_selects_device_before_snapshot_download(
    tmp_path, monkeypatch, cuda_available
):
    calls = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: calls.append("cuda-check") or cuda_available
        )
    )

    def stop_before_download(*args, **kwargs):
        calls.append("snapshot")
        raise RuntimeError("snapshot sentinel")

    monkeypatch.setattr(build_connections, "torch", fake_torch)
    monkeypatch.setattr(build_connections, "snapshot_download", stop_before_download)
    options = SimpleNamespace(
        model="sentinel/model",
        k=1,
        out=tmp_path / "connections.npz",
    )

    with pytest.raises(RuntimeError, match="snapshot sentinel"):
        build_connections.run_build_connections(options)

    assert calls == ["cuda-check", "snapshot"]
