# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

import os

from vllm_ascend.worker.worker import NPUWorker

from .gdn_qkv_norm_diagnostic import NormDiagnostic, attention_class_for_model


class GdnNormDiagnosticWorker(NPUWorker):
    def load_model(self):
        from vllm_ascend.ops import gdn

        from . import gdn_qkv_norm

        if not gdn._FUSE_GDN_QKV_NORM:
            raise RuntimeError('diagnostic requires the pinned GDN fusion enabled')
        directory = os.environ['GDN_DIAGNOSTIC_DIR']
        super().load_model()
        attention_class = attention_class_for_model(
            self.model_runner.get_model(),
            gdn.AscendGatedDeltaNetAttention._forward_core,
        )
        config = self.vllm_config
        metadata = {
            'model': config.model_config.model,
            'precision': str(config.model_config.dtype), 'format': 'ND',
            'tp': config.parallel_config.tensor_parallel_size,
            'graph': str(config.compilation_config.cudagraph_mode),
            'shape': {'T': 16, 'row_stride': 4096, 'q': 1024, 'k': 1024,
                      'v': 2048, 'head_dim': 128, 'phase': 'pure non-spec decode'},
            'callsite': 'post-CausalConv1d unpack+QK norm, before recurrent GDN',
            'bound_class': attention_class.__module__ + '.' + attention_class.__name__,
        }
        self.gdn_probe = NormDiagnostic(
            directory, self.rank,
            int(os.environ.get('GDN_DIAGNOSTIC_MAX_STEPS', '128')), metadata=metadata,
        )
        self.gdn_probe.install(gdn_qkv_norm, attention_class, gdn.l2norm_fwd)

    def execute_model(self, scheduler_output):
        self.gdn_probe.begin_step()
        result = super().execute_model(scheduler_output)
        self.gdn_probe.end_step(dict(scheduler_output.num_scheduled_tokens))
        return result
