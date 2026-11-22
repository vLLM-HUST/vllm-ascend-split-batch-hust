"""诊断脚本跟随登记的本地配置；冻结请求范围与 MOD 合同核对。"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "workloads" / "profile_hotspot_longrun.sh"
SHAPE = ROOT / "docs" / "acceptance-shape.json"

#: 脚本默认值（`NAME=${NAME:-<default>}`）→ JSON 登记键。
SCRIPT_ENV_TO_FIELD = {
    "MAX_MODEL_LEN": "max_model_len",
    "MAX_NUM_SEQS": "max_num_seqs",
    "MAX_BATCHED_TOKENS": "max_num_batched_tokens",
    "BLOCK_SIZE": "block_size",
    "GPU_MEMORY_UTILIZATION": "gpu_memory_utilization",
    "ENABLE_EXPERT_PARALLEL": "expert_parallel",
    "CUDAGRAPH_CAPTURE_SIZES": "cudagraph_capture_sizes",
    "ENABLE_CHUNKED_PREFILL": "chunked_prefill",
    "SCHEDULING_POLICY": "scheduling_policy",
    "DISABLE_CUSTOM_ALL_REDUCE": "disable_custom_all_reduce",
    "SERVER_SEED": "server_seed",
    "COMPILATION_CONFIG": "compilation_config",
    "SERVER_MODE": "server_mode",
    "PREFIX_CACHING": "prefix_caching",
    "PROFILE_REQUESTS": "profile_requests",
    "CONCURRENCY": "concurrency",
}


def script_defaults() -> dict[str, str]:
    text = SCRIPT.read_text(encoding="utf-8")
    pattern = re.compile(r"^([A-Z_]+)=\$\{\1:-(.*)\}$", re.MULTILINE)
    return {name: value.strip('"').strip("'") for name, value in pattern.findall(text)}


def registered_defaults() -> dict[str, object]:
    data = json.loads(SHAPE.read_text(encoding="utf-8"))
    return data["repository_shapes"]["profile_collection_script"]["defaults_now"]


def test_script_defaults_match_registered_collection_shape():
    defaults = script_defaults()
    registered = registered_defaults()
    violations = []
    for env_name, field in SCRIPT_ENV_TO_FIELD.items():
        if env_name not in defaults:
            violations.append(f"脚本缺少形态参数 {env_name}")
            continue
        if field not in registered:
            violations.append(f"登记 JSON 缺少字段 {field}")
            continue
        actual = defaults[env_name]
        expected = str(registered[field])
        if actual != expected:
            violations.append(f"{field}: 脚本默认 {actual!r} != 登记 {expected!r}")
    assert not violations, "采集脚本与诊断配置登记漂移: " + "; ".join(violations)


def test_request_profiles_match_the_frozen_mod_contract():
    shape = json.loads(SHAPE.read_text())
    evidence = json.loads((ROOT / "docs/mod-acceptance/evidence.json").read_text())
    standard = shape["request_shapes_standard"]
    assert standard["source"]["standard_id"] == evidence["sources"]["standard_id"]
    rows = evidence["contract"]["datasets"]
    assert standard["scopes"] == {"quick": ["mmlu"], "full": [r["id"] for r in rows]}
    assert set(standard["datasets"]) == set(standard["scopes"]["full"])
    for row in rows:
        profile = standard["datasets"][row["id"]]
        for key in ("requests_per_arm", "max_output_tokens", "seed", "temperature"):
            assert profile[key] == row[key]
        assert profile.get("judge_max_tokens") == row.get("judge_max_tokens")
        assert profile["concurrency_budget"] == row["concurrency"]
