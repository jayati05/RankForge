"""Opt-in soak instrumentation; no changes to deployed service factories or limits."""

import os
import threading
import time

from app.evaluation.windows_native import counters, regions

ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
    "RAYON_NUM_THREADS",
    "OMP_WAIT_POLICY",
    "KMP_BLOCKTIME",
)


def configuration():
    import torch
    from threadpoolctl import threadpool_info

    return {
        "thread_id": threading.get_ident(),
        "native_thread_id": threading.get_native_id(),
        "thread_name": threading.current_thread().name,
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "grad_enabled": torch.is_grad_enabled(),
        "inference_mode_enabled": torch.is_inference_mode_enabled(),
        "threadpools": threadpool_info(),
        "environment": {k: os.environ.get(k) for k in ENV_KEYS},
    }


def install(soak_module=None):
    """Called only by explicitly opted-in benchmark parent/child, never API CLI."""
    from app.evaluation import api_soak, soak_diagnostics
    import app.api.main as api
    from app.services.model_service import ModelService
    from app.retrieval.neural import SentenceEncoder, CrossEncoderScorer

    # A CLI executed with -m has a distinct __main__ module. Patch its globals,
    # not an independently imported canonical copy of api_soak.
    api_soak = soak_module or api_soak
    if getattr(api_soak.SoakSampler, "native_profile", False):
        return
    inference = {}

    class Encoder(SentenceEncoder):
        def encode(self, *args, **kwargs):
            key = f"encoder:{threading.get_ident()}"
            if key not in inference:
                inference[key] = configuration()
            return super().encode(*args, **kwargs)

    class Scorer(CrossEncoderScorer):
        def score(self, *args, **kwargs):
            key = f"scorer:{threading.get_ident()}"
            if key not in inference:
                inference[key] = configuration()
            return super().score(*args, **kwargs)

    original_app = api.create_app

    def create_app(settings=None, resource_factory=None):
        factory = resource_factory or (
            lambda s: ModelService(s, encoder_factory=Encoder, scorer_factory=Scorer)
        )
        return original_app(settings, resource_factory=factory)

    api.create_app = create_app

    class Diagnostics(soak_diagnostics.Diagnostics):
        def snapshot(self, manager, loop):
            result = super().snapshot(manager, loop)
            result["native"] = {
                "counters": counters(os.getpid()),
                "regions": regions(os.getpid()),
                "control_thread_configuration": configuration(),
                "inference_threads": dict(inference),
                "torch_build": __import__("torch").__config__.show(),
                "sampling_note": "Configuration recorded before adapter enters its internal no_grad/inference context; this is not evidence that model operations retain autograd graphs.",
            }
            return result

    soak_diagnostics.Diagnostics = Diagnostics

    class Sampler(api_soak.SoakSampler):
        native_profile = True

        def _sample(self):
            started = time.perf_counter()
            before = len(self.rows)
            super()._sample()
            if len(self.rows) > before:
                row = self.rows[-1]
                try:
                    row.update(native_threads=counters(self.pid)["native_threads"])
                    row.update(regions(self.pid, details=False)["totals"])
                except OSError as exc:
                    self.errors.append("native:" + type(exc).__name__)
                row["native_sampling_ms"] = (time.perf_counter() - started) * 1000

    api_soak.SoakSampler = Sampler
