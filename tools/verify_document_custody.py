"""Verify original documents and probe availability without executing evidence."""

import concurrent.futures
import json
import re
from pathlib import Path

from benchmark_evidence import download, relative_path, verify_bytes

ROOT = Path(__file__).resolve().parents[1]


def verify():
    source = json.loads((ROOT / "EVIDENCE_AVAILABILITY.json").read_text())
    assert source["repository"] == "vLLM-HUST/vllm-hust-benchmark"
    assert re.fullmatch(r"[a-f0-9]{40}", source["commit"])
    base = (
        f"https://raw.githubusercontent.com/{source['repository']}/{source['commit']}/"
    )

    def metadata(name):
        record = source["integrity"][name]
        relative_path(record["path"])
        return json.loads(verify_bytes(download(base + record["path"]), record))

    manifest = metadata("manifest.json")
    audit = metadata("local-reference-audit.json")
    assert (
        manifest["source_commit"] == audit["source_commit"] == source["source_commit"]
    )
    assert manifest["source_repository"] == "vLLM-HUST/vllm-ascend-split-batch-hust"
    assert len({r["path"] for r in manifest["files"]}) == len(manifest["files"])
    relative_path(source["snapshot"])

    def original(record):
        relative_path(record["path"])
        return len(
            verify_bytes(
                download(base + source["snapshot"] + "/" + record["path"]), record
            )
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        lengths = list(pool.map(original, manifest["files"]))
    refs = audit["references"]
    assert len(refs) == source["local_reference_occurrences"]
    assert (
        sum(r["canonical_raw_url"] is None for r in refs)
        == source["unrecovered_occurrences"]
    )
    return {"original_documents_verified": len(lengths), "bytes": sum(lengths)}


if __name__ == "__main__":
    print(json.dumps(verify()))
