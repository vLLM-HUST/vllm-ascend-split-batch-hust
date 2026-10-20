import importlib
import sqlite3
from pathlib import Path


def database(path, payload=32768):
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE STRING_IDS(id INTEGER, value TEXT);
        INSERT INTO STRING_IDS VALUES
          (1,'hcom_allReduce__0'),(2,'vllm::all_reduce'),
          (3,'aclmdlRIExecuteAsync'),(4,'[16,2048]');
        CREATE TABLE COMMUNICATION_OP(opName, startNs, endNs, connectionId,
          count, dataType, rankSize, opId);
        CREATE TABLE PYTORCH_API(startNs,endNs,name,inputShapes,callchainId);
        INSERT INTO PYTORCH_API VALUES (0,9,2,4,1);
        CREATE TABLE CANN_API(startNs,endNs,name);
        INSERT INTO CANN_API VALUES (1,3,3),(7,7,3);
        CREATE TABLE OVERLAP_ANALYSIS(startNs,endNs,type);
        INSERT INTO OVERLAP_ANALYSIS VALUES (0,10,3);
        CREATE TABLE TASK(startNs,endNs,modelId,streamId,contextId);
        INSERT INTO TASK VALUES (3,8,1,1,1);
    """)
    con.execute("INSERT INTO COMMUNICATION_OP VALUES (1,3,8,5,?,11,2,0)", (payload,))
    con.commit()
    con.close()


def test_pairing_checks_payload_not_only_names(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    api = importlib.import_module("audit_comm_mechanism")
    paths = [tmp_path / "rank0.db", tmp_path / "rank1.db"]
    database(paths[0])
    database(paths[1], payload=65536)
    result = api.audit(paths)
    assert not result["rank_pair_communication_name_payload_dtype_size_equal"]
    rank = result["ranks"][0]
    assert rank["execute_intervals"] == 2
    assert rank["execute_intervals_with_positive_duration"] == 1
    assert rank["free_intersect_execute_host_api_ms"] == 0.000002
    assert rank["free_outside_execute_host_api_ms"] == 0.000008
    assert rank["allreduce_api_fields"]["vllm::all_reduce"]["nonempty_shapes"] == 1
    assert result["status"].startswith("blocked")
    assert "not device graph completion" in rank["boundary_limit"]
