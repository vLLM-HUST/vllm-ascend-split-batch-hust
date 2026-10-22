"""C 轮 CPU 测试: 变换表对归档 stock orig 的锚点命中 / 指纹产出 / fail-closed。"""

import importlib.util
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "rsp", HERE.parent / "src" / "vllm_ascend_split_batch" / "runtime_source_patches.py"
)
rsp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rsp)

PDIR = pathlib.Path("/root/包体恢复/records/tree-patches")
ORIGS = {
    rsp.GDN: PDIR / "20261021T1355Z_d4_conv_hoist_site2.orig-gdn.py",
    rsp.TD: PDIR / "20261021T1710Z_moeInitTriton.token_dispatcher.py.orig",
    rsp.LIN: PDIR / "20261021T1545Z_nzSkipGate.linear.py.orig",
}

FINGERPRINTS = {
    rsp.GDN: [
        "_PACKED_CONV_WEIGHT_NAME",
        "_PACK_CONV_ENV",
        "_UNPACK_VIEWS_ON",
        "_unpack_mixed_qkv_views",
    ],
    rsp.TD: ["_moe_init_routing_t1", "vllm_ascend_split_batch.moe_init_triton"],
    rsp.LIN: ["VLLM_HUST_NZ_SKIP_GATE", "skip_weight_nz_conversion"],
}


def _tables_for(module):
    return [t for env, m, t in rsp.TABLES if m == module]


def test_gdn_transforms_apply_on_stock_orig():
    src = ORIGS[rsp.GDN].read_text()
    d4 = _tables_for(rsp.GDN)[0]
    views = _tables_for(rsp.GDN)[1]
    src1, _ = rsp.apply_transforms(src, d4)
    assert "_PACKED_CONV_WEIGHT_NAME" in src1
    assert src1.count("conv_weights.transpose(0, 1)") == 1  # 仅 stock 回退
    src2, _ = rsp.apply_transforms(src1, views)
    for fp in FINGERPRINTS[rsp.GDN]:
        assert fp in src2, fp
    assert "_UNPACK_VIEWS_ON" in src2


def test_dispatcher_transforms_apply_on_stock_orig():
    src = ORIGS[rsp.TD].read_text()
    tables = _tables_for(rsp.TD)
    out = src
    for t in tables:
        out, _ = rsp.apply_transforms(out, t)
    for fp in FINGERPRINTS[rsp.TD]:
        assert fp in out, fp
    assert "DeviceOperator.npu_moe_init_routing(" in out  # stock 分支保留


def test_linear_transforms_apply_on_stock_orig():
    src = ORIGS[rsp.LIN].read_text()
    tables = _tables_for(rsp.LIN)
    out = src
    for t in tables:
        out, _ = rsp.apply_transforms(out, t)
    for fp in FINGERPRINTS[rsp.LIN]:
        assert fp in out, fp


def test_anchor_drift_is_fail_closed():
    src = (
        ORIGS[rsp.GDN]
        .read_text()
        .replace("class AscendGatedDeltaNetAttention", "class Renamed", 1)
    )
    try:
        rsp.apply_transforms(src, _tables_for(rsp.GDN)[0])
    except rsp.AnchorError:
        return
    raise AssertionError("锚点漂移必须抛 AnchorError（fail-closed）")


def test_no_partial_application_on_drift():
    src = (
        ORIGS[rsp.GDN]
        .read_text()
        .replace(
            "if spec_sequence_masks is not None:\n            conv_weights_T",
            "if spec_sequence_masks is not None:\n            Xconv_weights_T",
            1,
        )
    )
    try:
        rsp.apply_transforms(src, _tables_for(rsp.GDN)[0])
    except rsp.AnchorError:
        return
    raise AssertionError("中途失配必须整体抛错（不得产出半变换源码）")
