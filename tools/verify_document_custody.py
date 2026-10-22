"""Verify original documents and probe availability without executing evidence."""

import concurrent.futures
import json
import re
from pathlib import Path

from benchmark_evidence import download, relative_path, verify_bytes

ROOT = Path(__file__).resolve().parents[1]


def verify_recovery(audit, previous, fetch):
    assert audit["source_commit"] == previous["source_commit"]
    assert len(audit["references"]) == len(previous["references"])
    recovered = 0
    for before, after in zip(previous["references"], audit["references"], strict=True):
        for key in ("original_reference", "source_file", "location", "declared_sha256"):
            assert before[key] == after[key]
        if "recovery" not in after:
            assert before == after
            continue
        record = after["recovery"]
        assert before["canonical_raw_url"] is None
        assert after["status"] == "recovered-by-source-map-and-sha256"
        assert re.fullmatch(r"[a-f0-9]{40}", record["commit"])
        base = (
            "https://raw.githubusercontent.com/vLLM-HUST/vllm-hust-benchmark/"
            + record["commit"]
            + "/"
        )
        for kind in ("raw", "proof"):
            relative_path(record[kind]["path"])
            data = verify_bytes(fetch(base + record[kind]["path"]), record[kind])
            if kind == "proof":
                mapping = json.loads(data)
        for part in record["json_pointer"].lstrip("/").split("/"):
            key = part.replace("~1", "/").replace("~0", "~")
            mapping = mapping[int(key)] if isinstance(mapping, list) else mapping[key]
        assert mapping[record["mapping_key"]] == after["original_reference"]
        assert mapping["sha256"] == record["raw"]["sha256"]
        assert after["canonical_raw_url"] == (
            "https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/"
            + record["commit"]
            + "/"
            + record["raw"]["path"]
        )
        recovered += 1
    return recovered


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
    recovery = source["recovery_audit"]
    assert re.fullmatch(r"[a-f0-9]{40}", recovery["commit"])
    relative_path(recovery["path"])
    recovered_audit = json.loads(
        verify_bytes(
            download(
                f"https://raw.githubusercontent.com/{source['repository']}/"
                f"{recovery['commit']}/{recovery['path']}"
            ),
            recovery,
        )
    )
    assert recovered_audit["previous_audit"] == {
        "commit": source["commit"],
        "path": source["audit"],
        "sha256": source["integrity"]["local-reference-audit.json"]["sha256"],
    }
    recovered = verify_recovery(recovered_audit, audit, download)
    audit = recovered_audit
    refs = audit["references"]
    assert len(refs) == source["local_reference_occurrences"]
    assert (
        sum(r["canonical_raw_url"] is None for r in refs)
        == source["unrecovered_occurrences"]
    )
    assert (
        sum(r["canonical_raw_url"] is not None for r in refs)
        == source["sha256_recovered_occurrences"]
    )
    return {
        "original_documents_verified": len(lengths),
        "bytes": sum(lengths),
        "newly_recovered_verified": recovered,
    }


if __name__ == "__main__":
    print(json.dumps(verify()))
