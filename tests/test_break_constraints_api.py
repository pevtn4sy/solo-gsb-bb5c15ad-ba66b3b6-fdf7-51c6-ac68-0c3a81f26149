"""POST /api/v1/plans 必翻/禁翻约束的端到端测试。"""

from __future__ import annotations

import importlib
import sys

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """以临时 DATA_DIR 重新加载 app.main，得到隔离的服务实例。"""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    sys.modules.pop("app.main", None)
    main = importlib.import_module("app.main")
    with TestClient(main.app) as c:
        yield c


_BASE = {
    "measure_heights": [1] * 8,
    "silence_seconds": [9, 9, 9, 1, 1, 9, 9, 0],
    "page_height": 3,
    "min_turn_time": 1,
}


def _post(client, extra=None):
    payload = dict(_BASE)
    if extra:
        payload.update(extra)
    return client.post("/api/v1/plans", json=payload)


def test_required_and_forbidden_happy_path(client):
    resp = _post(client, {"required_breaks": [4], "forbidden_breaks": [2]})
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["request"]["required_breaks"] == [4]
    assert body["request"]["forbidden_breaks"] == [2]
    breaks = body["result"]["breaks"]
    assert 4 in breaks
    assert 2 not in breaks
    assert body["result"]["shortest_turn_silence"] == 1.0


def test_constraints_persisted_and_returned_on_get(client):
    resp = _post(client, {"required_breaks": [4, 6], "forbidden_breaks": [2]})
    assert resp.status_code == 201, resp.text
    plan_id = resp.json()["id"]

    got = client.get(f"/api/v1/plans/{plan_id}")
    assert got.status_code == 200
    request = got.json()["request"]
    assert request["required_breaks"] == [4, 6]
    assert request["forbidden_breaks"] == [2]


def test_omitted_constraints_absent_from_echo(client):
    resp = _post(client)
    assert resp.status_code == 201, resp.text
    request = resp.json()["request"]
    assert request["required_breaks"] is None
    assert request["forbidden_breaks"] is None


def test_required_silence_insufficient_returns_422(client):
    # 必翻 b=4 即第 4 小节之后，对应 silence_seconds[3]
    silences = [9, 9, 9, 0, 9, 9, 9, 0]
    resp = _post(client, {"required_breaks": [4], "silence_seconds": silences})
    assert resp.status_code == 422
    assert "必翻位置" in resp.text and "低于最低翻页时间" in resp.text


def test_no_feasible_plan_under_constraints_returns_422(client):
    # 容量 2 的 4 小节，s[1] 不达标；禁掉 b=1、b=3 后 2 小节装不进一页。
    resp = client.post(
        "/api/v1/plans",
        json={
            "measure_heights": [1, 1, 1, 1],
            "silence_seconds": [5, 0, 5, 0],
            "page_height": 2,
            "min_turn_time": 1,
            "forbidden_breaks": [1, 3],
        },
    )
    assert resp.status_code == 422
    assert "无合法翻页方案" in resp.text


@pytest.mark.parametrize(
    "extra",
    [
        {"required_breaks": [0]},                      # 含 0
        {"required_breaks": [8]},                      # 末尾
        {"required_breaks": [9]},                      # 越界
        {"required_breaks": [-1]},                     # 负值越界
        {"required_breaks": [2, 2]},                   # 自身重复
        {"forbidden_breaks": [2, 2]},                  # 自身重复
        {"required_breaks": [2], "forbidden_breaks": [2]},  # 交叉
    ],
)
def test_invalid_constraint_arrays_return_422(client, extra):
    resp = _post(client, extra)
    assert resp.status_code == 422, resp.text


def test_constraint_elements_must_be_integers(client):
    # 带小数部分的数值与 JSON 布尔 true/false 都不得当作断页序号。
    resp = _post(client, {"required_breaks": [2.5]})
    assert resp.status_code == 422
    resp_bool = _post(client, {"required_breaks": [True]})
    assert resp_bool.status_code == 422
    resp_ok = _post(client, {"required_breaks": [2]})
    assert resp_ok.status_code == 201, resp_ok.text


def test_unknown_field_still_rejected(client):
    resp = _post(client, {"required_breaks": [2], "bogus": 1})
    assert resp.status_code == 422


def test_unconstrained_request_unchanged(client):
    resp = _post(client)
    assert resp.status_code == 201, resp.text
    body = resp.json()["result"]
    # 无约束全局最优：[3,6]，最短静默 9
    assert body["breaks"] == [3, 6]
    assert body["page_count"] == 3
    assert body["shortest_turn_silence"] == 9.0
