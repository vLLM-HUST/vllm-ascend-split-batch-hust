"""Show the next design task and verify archived inputs without scheduling work."""

import argparse
import hashlib
import json
import re
import shlex
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

RECORD = Path(__file__).resolve().parents[2] / "docs/optimization-next.json"


def check_hash(root, name, expected):
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"路径不在归档内：{name}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"SHA256 不匹配：{path}")
    return path


def verify(record, archive):
    receipt = json.loads(
        check_hash(archive, "final-receipt.json", record["receipt_sha256"]).read_bytes()
    )
    hosts = record["host_commits"]
    if (
        set(hosts) != {"vllm", "vllm-ascend"}
        or not all(re.fullmatch("[0-9a-f]{40}", v) for v in hosts.values())
        or receipt["host_commits"] != hosts
    ):
        raise ValueError("回执与登记的双宿主身份不符")
    if receipt["raw_collection_completed"] is not True or receipt["analysis_exit"] != 0:
        raise ValueError("归档未完成采集或分析")
    if int((archive / "exit-code").read_text()) != receipt["collector_exit"]:
        raise ValueError("原始采集退出码与回执不符")
    sources = {
        name: check_hash(archive, name, sha)
        for name, sha in receipt["source_hashes"].items()
    }
    preflight = json.loads(sources["host_preflight.json"].read_bytes())
    if preflight["returncode"] != 0:
        raise ValueError("归档内宿主预检未通过")
    observed = json.loads(preflight["stdout"])
    if any(observed[k]["revision"] != v for k, v in hosts.items()):
        raise ValueError("归档内服务解释器的宿主身份不符")
    raw = json.loads(sources["result.json"].read_bytes())["source_sha256"]
    if not raw:
        raise ValueError("缺少逐卡原件指纹")
    for name, sha in raw.items():
        check_hash(archive, name, sha)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="挂载在其他位置的同一份归档目录")
    args = parser.parse_args(argv)
    try:
        record = json.loads(RECORD.read_bytes())
        print(
            f"下一项：{record['candidate']}\n范围：{record['scope']}\n依据：{record['reason']}\n{record['next_step']}"
        )
        if "last_closed" in record:
            closed = RECORD.parents[1] / record["last_closed"]
            if not closed.is_file():
                raise ValueError(f"缺研发结案记录：{closed}")
            print(f"上一项结论与重开条件：{record['last_closed']}")
        for step in record["checks"]:
            print(f"判据：{step}")
        for name, reason in record["skip"].items():
            print(f"跳过 {name}：{reason}")
        print(
            "背景热点（不同窗口，仅作选题参考）："
            "python -B -m tools.operator_tools.artifact_view "
            + shlex.quote(record["hotspot_archive"])
            + " --view hotspot --top 5"
        )
        plan = record["reviewed_plan"]
        if plan is None:
            print("无已评审执行计划；不占卡、不起服务、不写回。")
        else:
            path = check_hash(RECORD.parents[1], plan["path"], plan["sha256"])
            print(f"登记计划：{path}（仅核对文件，不执行）")
        uri = urlparse(record["archive_uri"])
        if uri.scheme != "file" or uri.netloc or not Path(uri.path).is_absolute():
            raise ValueError("需要本机归档 file URI，或挂载后重定位")
        archive = args.archive if args.archive is not None else Path(unquote(uri.path))
        print(f"归档：{archive}")
        receipt = verify(record, archive)
        print(
            f"回执、逐卡输入和归档内双宿主身份已核对；collector_exit={receipt['collector_exit']}，analysis_exit={receipt['analysis_exit']}。未核验当前运行时。"
        )
    except FileNotFoundError as exc:
        print(f"缺少 {exc.filename}；只可静态设计，不能重放。")
        return 2
    except (OSError, KeyError, ValueError, TypeError) as exc:
        print(f"核验失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
