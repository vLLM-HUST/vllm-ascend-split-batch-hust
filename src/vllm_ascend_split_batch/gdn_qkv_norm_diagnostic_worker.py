# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

import os

from vllm_ascend.worker.worker import NPUWorker

from .gdn_qkv_norm_diagnostic import NormDiagnostic


class GdnNormDiagnosticWorker(NPUWorker):
    def load_model(self):
        from vllm_ascend.ops import gdn

        from . import gdn_qkv_norm

        if not gdn._FUSE_GDN_QKV_NORM:
            raise RuntimeError('diagnostic requires the pinned GDN fusion enabled')
        directory = os.environ['GDN_DIAGNOSTIC_DIR']
        self.gdn_probe = NormDiagnostic(
            directory, self.rank,
            int(os.environ.get('GDN_DIAGNOSTIC_MAX_STEPS', '128')),
        )
        self.gdn_probe.install(
            gdn_qkv_norm, gdn.AscendGatedDeltaNetAttention, gdn.l2norm_fwd,
        )
        super().load_model()

    def execute_model(self, scheduler_output):
        self.gdn_probe.begin_step()
        result = super().execute_model(scheduler_output)
        self.gdn_probe.end_step(dict(scheduler_output.num_scheduled_tokens))
        return result
