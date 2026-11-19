from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

from google.genai import types


def _message(text: str):
    return types.Content(role="user", parts=[types.Part.from_text(text=text)])


def test_rate_limiter_enforces_per_user_sliding_window():
    from assignment.rate_limiter import RateLimitPlugin

    limiter = RateLimitPlugin(max_requests=2, window_seconds=10)

    async def request(user_id):
        return await limiter.on_user_message_callback(
            invocation_context=SimpleNamespace(user_id=user_id),
            user_message=_message("What is my account balance?"),
        )

    assert asyncio.run(request("alice")) is None
    assert asyncio.run(request("alice")) is None
    assert asyncio.run(request("alice")) is not None
    limiter.user_windows["alice"].clear()
    limiter.user_windows["alice"].append(time.monotonic() - 11)
    assert asyncio.run(request("alice")) is None
    assert asyncio.run(request("bob")) is None
    assert limiter.blocked_count == 1


def test_audit_log_exports_request_and_decision(tmp_path):
    from assignment.audit_log import AuditLogPlugin

    audit = AuditLogPlugin()
    audit.record_input(user_id="alice", text="What is my balance?", request_id="r-1")
    audit.record_output(
        user_id="alice",
        text="Request blocked.",
        blocked=True,
        layer="input_guardrail",
        request_id="r-1",
    )
    path = tmp_path / "audit.json"
    audit.export_json(str(path))

    [entry] = json.loads(path.read_text(encoding="utf-8"))
    assert entry["input"] == "What is my balance?"
    assert entry["blocked"] is True
    assert entry["layer"] == "input_guardrail"
    assert entry["latency_seconds"] >= 0


def test_monitoring_reports_threshold_breaches():
    from assignment.monitoring import MonitoringAlert

    monitor = MonitoringAlert(
        block_rate_threshold=0.5,
        rate_limit_hit_threshold=1,
        judge_fail_rate_threshold=0.25,
        total_requests=2,
        blocked_requests=2,
        rate_limit_hits=2,
        judge_checks=2,
        judge_fails=1,
    )

    assert {alert.metric for alert in monitor.check_metrics()} == {
        "block_rate",
        "rate_limit_hits",
        "judge_fail_rate",
    }


def test_egress_rejects_malformed_url():
    from assignment.pipeline import is_egress_allowed

    assert is_egress_allowed("https://[::1", "ordinary transfer payload") is False
