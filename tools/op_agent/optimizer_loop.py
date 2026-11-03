#!/usr/bin/env python3
"""Minimal Smolagents-based Operator Optimization Loop (L0 + L3 MVP).

Customized for Qwen3.5-35B (TP=2) operator tuning under repo standards.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from smolagents import CodeAgent, Model, Tool
from smolagents.models import ChatMessage

from tools.op_agent.custody import verify_dir_checksum
from tools.op_agent.evaluator import run_candidate_evaluation
from tools.op_agent.exporter import export_operator_delivery
from tools.op_agent.guard import check_npu_environment
from tools.op_agent.kb import query_ascend_kb, read_kb_card
from tools.op_agent.qwen35_shapes import CANONICAL_CASES

CURRENT_TEST_CASE: dict[str, Any] = {}
BEST_RECORD: dict[str, Any] = {
    "l0_median_us": float("inf"),
    "kernel_code": "",
    "iteration": 0,
}


class L3CheckTool(Tool):
    name = "check_l3"
    description = (
        "Check numerical correctness (L3) of candidate operator code against baseline. "
        "Returns a JSON string with 'l3_passed' (bool) and 'error' (str if any). "
        "Your code MUST define an entrypoint function named `optimized_op(*args)`."
    )
    inputs = {
        "kernel_code": {
            "type": "string",
            "description": "The Python/Triton source code defining `optimized_op`.",
        }
    }
    output_type = "string"

    def forward(self, kernel_code: str) -> str:
        res = run_candidate_evaluation(kernel_code, CURRENT_TEST_CASE)
        if not res["l3_passed"]:
            return json.dumps(
                {
                    "l3_passed": False,
                    "error": res["error"],
                }
            )
        return json.dumps(
            {
                "l3_passed": True,
                "message": "L3 numerical consistency check passed successfully!",
            }
        )


class L0BenchmarkTool(Tool):
    name = "benchmark_l0"
    description = (
        "Benchmark pure operator latency (L0) in µs using Graph/Eager mode. "
        "Returns a JSON string with 'l0_median_us' (float) or error. "
        "Only call this if check_l3 has passed!"
    )
    inputs = {
        "kernel_code": {
            "type": "string",
            "description": "The Python/Triton source code defining `optimized_op`.",
        }
    }
    output_type = "string"

    def forward(self, kernel_code: str) -> str:
        global BEST_RECORD
        res = run_candidate_evaluation(kernel_code, CURRENT_TEST_CASE)
        if not res["l3_passed"]:
            return json.dumps(
                {
                    "status": "failed",
                    "error": f"L3 check failed before benchmarking: {res['error']}",
                }
            )

        latency = res["l0_median_us"]
        improved = latency < BEST_RECORD["l0_median_us"]
        if improved:
            BEST_RECORD["l0_median_us"] = latency
            BEST_RECORD["kernel_code"] = kernel_code

        return json.dumps(
            {
                "status": "success",
                "l0_median_us": latency,
                "best_so_far_us": BEST_RECORD["l0_median_us"],
                "improved": improved,
            }
        )


class SearchKnowledgeBaseTool(Tool):
    name = "search_knowledge_base"
    description = (
        "Search the official Ascend / CANN / Triton knowledge base cards. "
        "Returns titles, paths, and brief summaries of matching best practices."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "Keywords, e.g. 'triton tiling', 'reduce launch overhead'.",
        }
    }
    output_type = "string"

    def forward(self, query: str) -> str:
        results = query_ascend_kb(query, limit=3)
        return json.dumps(results, ensure_ascii=False)


class ReadKnowledgeCardTool(Tool):
    name = "read_knowledge_card"
    description = "Read detailed content from an Ascend knowledge card file path."
    inputs = {
        "file_path": {
            "type": "string",
            "description": "Path to the markdown card file.",
        }
    }
    output_type = "string"

    def forward(self, file_path: str) -> str:
        return read_kb_card(file_path, max_chars=1500)


def setup_qwen35_case(case_name: str = "moe_shared_epilogue_decode") -> dict[str, Any]:
    """Setup real Qwen3.5-35B TP=2 operator shape facts."""
    case_meta = CANONICAL_CASES[case_name]
    M, H = case_meta.get("gate_shape", (32, 1))[0], 2048

    hw = check_npu_environment(required_cards=2)

    def input_gen():
        if hw.is_tp2_ready:
            import torch
            import torch_npu  # noqa: F401

            torch.manual_seed(42)
            gate = torch.randn(M, 1, dtype=torch.bfloat16).npu(0)
            out = torch.randn(M, H, dtype=torch.bfloat16).npu(0)
            return gate, out
        else:
            np.random.seed(42)
            gate = np.random.randn(M, 1).astype(np.float32)
            out = np.random.randn(M, H).astype(np.float32)
            return gate, out

    def baseline_op(gate, out):
        return gate * out

    return {
        "name": case_name,
        "callsite": case_meta["callsite"],
        "shape_facts": {"M": M, "Hidden": H, "phase": case_meta["phase"]},
        "input_gen_fn": input_gen,
        "baseline_fn": baseline_op,
        "iters": 50,
        "required_cards": 2,
    }


class MockKernelModel(Model):
    """Mock Agent model demonstrating auto-optimization on Qwen3.5-35B shapes."""

    def __init__(self):
        super().__init__()
        self.step = 0
        self.candidates = [
            # Candidate 1: Numerical failure (addition instead of multiply)
            """
def optimized_op(gate, out):
    return gate + out
""",
            # Candidate 2: Passes L3
            """
def optimized_op(gate, out):
    return gate * out
""",
            # Candidate 3: Passes L3, cloned operation
            """
def optimized_op(gate, out):
    return out.clone() * gate if hasattr(out, 'clone') else (out * gate).copy()
""",
        ]

    def generate(self, messages, stop_sequences=None, **kwargs) -> ChatMessage:
        self.step += 1
        if self.step == 1:
            # Query knowledge base first per AGENTS.md rule
            action_code = (
                "kb_info = search_knowledge_base(query='triton launch overhead')\n"
                "print('KB Search Findings:', kb_info)\n"
            )
        elif self.step - 2 < len(self.candidates):
            idx = self.step - 2
            code = self.candidates[idx].strip()
            escaped_code = json.dumps(code)

            action_code = (
                f"l3_raw = check_l3(kernel_code={escaped_code})\n"
                f"print('Step {self.step} L3 check:', l3_raw)\n"
                f"if '\"l3_passed\": true' in l3_raw.lower():\n"
                f"    l0_raw = benchmark_l0(kernel_code={escaped_code})\n"
                f"    print('Step {self.step} L0 benchmark:', l0_raw)\n"
            )
        else:
            action_code = "final_answer('Knowledge consult and search finished.')"

        full_content = f"```python\n{action_code}\n```"
        return ChatMessage(role="assistant", content=full_content)


def get_configured_model() -> Model:
    """Instantiate LLM model based on environment variables.

    Supported env variables:
      OP_AGENT_MODEL_ID: model name (e.g. 'deepseek-chat', 'gpt-4o')
      OP_AGENT_API_BASE: custom base URL (e.g. 'http://localhost:8000/v1')
      OP_AGENT_API_KEY: api key (falls back to OPENAI_API_KEY)
    If OP_AGENT_MODEL_ID is omitted, defaults to MockKernelModel.
    """
    model_id = os.environ.get("OP_AGENT_MODEL_ID")
    if not model_id:
        return MockKernelModel()

    api_base = os.environ.get("OP_AGENT_API_BASE") or os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OP_AGENT_API_KEY") or os.environ.get("OPENAI_API_KEY")

    # Use LiteLLMModel or OpenAIServerModel
    from smolagents import OpenAIServerModel

    kwargs: dict[str, Any] = {"model_id": model_id}
    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key

    return OpenAIServerModel(**kwargs)


def create_agent(model=None, tools=None):
    if tools is None:
        tools = [
            L3CheckTool(),
            L0BenchmarkTool(),
            SearchKnowledgeBaseTool(),
            ReadKnowledgeCardTool(),
        ]
    if model is None:
        model = get_configured_model()

    return CodeAgent(
        tools=tools,
        model=model,
        max_steps=5,
        add_base_tools=False,
    )


def run_qwen35_optimization(case_name: str = "moe_shared_epilogue_decode", model=None):
    global CURRENT_TEST_CASE, BEST_RECORD
    CURRENT_TEST_CASE = setup_qwen35_case(case_name)

    hw = check_npu_environment(required_cards=2)
    print("==========================================")
    print("=== NPU Environment Pre-flight Check ===")
    print(f"Device Name: {hw.device_name}")
    print(f"Detected NPU Cards: {hw.device_count}")
    print(f"TP=2 Hardware Ready: {hw.is_tp2_ready}")
    print(f"Notice: {hw.message}")
    print("==========================================\n")

    # Baseline L0 measurement
    baseline_code = """
def optimized_op(gate, out):
    return gate * out
"""
    base_res = run_candidate_evaluation(baseline_code, CURRENT_TEST_CASE)
    base_latency = base_res["l0_median_us"]
    BEST_RECORD["l0_median_us"] = base_latency
    BEST_RECORD["kernel_code"] = baseline_code

    print(f"=== Baseline L0 Latency: {base_latency:.2f} µs ===")

    tools = [
        L3CheckTool(),
        L0BenchmarkTool(),
        SearchKnowledgeBaseTool(),
        ReadKnowledgeCardTool(),
    ]
    agent = create_agent(model=model, tools=tools)

    task = (
        f"Optimize {case_name} operator. "
        "Test candidates using `check_l3`. If it passes, run `benchmark_l0`."
    )

    try:
        agent.run(task)
    except Exception as e:
        print(f"Agent finished: {e}")

    print("\n==========================================")
    print(f"=== Operator Optimization Completed for {case_name} ===")
    print(f"Baseline Latency: {base_latency:.2f} µs")
    print(f"Best Optimized Latency: {BEST_RECORD['l0_median_us']:.2f} µs")
    speedup = base_latency / max(BEST_RECORD["l0_median_us"], 1e-6)
    print(f"L0 Speedup: {speedup:.2f}x")
    print("==========================================\n")

    # Export delivery if improved
    if BEST_RECORD["l0_median_us"] < base_latency:
        out_dir = export_operator_delivery(
            op_name=case_name,
            callsite=CURRENT_TEST_CASE["callsite"],
            baseline_latency=base_latency,
            optimized_latency=BEST_RECORD["l0_median_us"],
            kernel_code=BEST_RECORD["kernel_code"],
            shape_facts=CURRENT_TEST_CASE["shape_facts"],
        )
        ok, msg = verify_dir_checksum(out_dir)
        print(f"✅ Delivered standard operator bundle into: {out_dir}")
        print(f"🔒 Anti-drift Checksum Gate: {'PASS' if ok else 'FAIL'} ({msg})")

    return BEST_RECORD


if __name__ == "__main__":
    run_qwen35_optimization()
