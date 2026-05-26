"""Minimal Chronos-2 multivariate inference wrapper.

Self-contained so the article reads independently from the SKAB sequel.
Uses BaseChronosPipeline directly. Tested with chronos 2.2.2 + torch 2.12 +cu130.

Run inside the SKAB chronos2 venv:
    ~/works/timeseries-fm-bench/chronos2/.venv/bin/python ...
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np


MODEL_REGISTRY: dict[str, str] = {
    "chronos2-28m": "autogluon/chronos-2-small",
    "chronos2-120m": "amazon/chronos-2",
}

ModelName = Literal["chronos2-28m", "chronos2-120m"]


@dataclass
class Chronos2Predictor:
    """Thin wrapper around BaseChronosPipeline.predict_quantiles for multivariate input.

    Input shape convention is (T, V) per window or (N, T, V) for a batch of
    windows. Chronos-2 itself wants (1, V, T), so transposition is done here.
    """

    model_name: ModelName = "chronos2-28m"
    device: str = "cuda"
    quantile_levels: tuple[float, ...] = (0.1, 0.5, 0.9)

    _pipeline: object | None = field(default=None, init=False, repr=False)

    def load(self) -> None:
        import torch  # type: ignore[import-not-found]
        from chronos import BaseChronosPipeline  # type: ignore[import-not-found]

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        repo = MODEL_REGISTRY[self.model_name]
        self._pipeline = BaseChronosPipeline.from_pretrained(
            repo,
            device_map=self.device,
            dtype=dtype,
        )

    def predict_multivariate(
        self,
        context: np.ndarray,
        horizon: int,
    ) -> tuple[np.ndarray, list[float]]:
        """Predict median quantile for every window.

        context shape:
          (T, V)      -> single window, returns preds (h, V)
          (N, T, V)   -> batch, returns preds (N, h, V)

        Returns (preds, per_window_latency_seconds).
        """
        import torch  # type: ignore[import-not-found]

        if self._pipeline is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        arr = np.asarray(context, dtype=np.float32)
        squeezed = False
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
            squeezed = True
        if arr.ndim != 3:
            raise ValueError(f"context must be (T, V) or (N, T, V); got {arr.shape}")

        # (N, T, V) -> (N, V, T)
        arr = np.transpose(arr, (0, 2, 1))
        n, n_var, _ = arr.shape

        preds = np.empty((n, horizon, n_var), dtype=np.float32)
        latencies: list[float] = []
        for i in range(n):
            ctx_tensor = torch.tensor(arr[i : i + 1], dtype=torch.float32)
            t0 = time.perf_counter()
            quantiles_list, _ = self._pipeline.predict_quantiles(  # type: ignore[union-attr]
                ctx_tensor,
                prediction_length=horizon,
                quantile_levels=list(self.quantile_levels),
            )
            latencies.append(time.perf_counter() - t0)
            q = quantiles_list[0].cpu().numpy()  # (V, h, len(quantiles))
            median_idx = self.quantile_levels.index(0.5)
            preds[i] = q[:, :, median_idx].T  # (h, V)

        if squeezed:
            return preds[0], latencies
        return preds, latencies

    def warmup(self, n_var: int, context_len: int, horizon: int, runs: int = 3) -> None:
        """Run a few throw-away predictions so cuDNN graphs are captured."""
        dummy = np.zeros((1, context_len, n_var), dtype=np.float32)
        for _ in range(runs):
            self.predict_multivariate(dummy, horizon)
