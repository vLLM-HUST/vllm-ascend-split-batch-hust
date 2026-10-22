import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from verify_document_custody import verify_recovery  # noqa: E402


def fixture():
    original = "/synthetic/probe.json"
    payload = b'{"synthetic": true}\n'
    digest = hashlib.sha256(payload).hexdigest()
    proof = json.dumps({"files": [{"source": original, "sha256": digest}]}).encode()
    blobs = {"raw.json": payload, "proof.json": proof}
    before = {
        "original_reference": original,
        "source_file": "synthetic.json",
        "location": "/evidence",
        "declared_sha256": None,
        "canonical_raw_url": None,
        "status": "unrecovered-local-original",
    }
    record = {
        "commit": "a" * 40,
        "json_pointer": "/files/0",
        "mapping_key": "source",
        **{
            kind: {
                "path": path,
                "bytes": len(blobs[path]),
                "sha256": hashlib.sha256(blobs[path]).hexdigest(),
            }
            for kind, path in (("raw", "raw.json"), ("proof", "proof.json"))
        },
    }
    after = {
        **before,
        "status": "recovered-by-source-map-and-sha256",
        "recovery": record,
        "canonical_raw_url": "https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/"
        + record["commit"]
        + "/raw.json",
    }
    previous = {"source_commit": "b" * 40, "references": [before]}
    audit = {**previous, "references": [after]}
    return audit, previous, blobs


def test_recovery_checks_source_identity_and_bytes():
    audit, previous, blobs = fixture()
    assert (
        verify_recovery(audit, previous, lambda url: blobs[url.rsplit("/", 1)[1]]) == 1
    )


@pytest.mark.parametrize("damage", ["bytes", "identity", "proof", "url", "missing"])
def test_recovery_rejects_invalid_evidence(damage):
    audit, previous, blobs = fixture()
    audit = copy.deepcopy(audit)
    row = audit["references"][0]
    if damage == "bytes":
        blobs["raw.json"] += b"changed"
    elif damage == "identity":
        row["original_reference"] = "/wrong/probe.json"
    elif damage == "proof":
        row["recovery"]["mapping_key"] = "sha256"
    elif damage == "url":
        row["canonical_raw_url"] = "https://example.com/summary"
    else:
        audit["references"] = []
    with pytest.raises((AssertionError, ValueError)):
        verify_recovery(audit, previous, lambda url: blobs[url.rsplit("/", 1)[1]])
