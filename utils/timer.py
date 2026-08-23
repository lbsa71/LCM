"""Per-step runtime and wall-clock timer utility."""

import csv
import json
import os
import time
from typing import Any, Dict, List, Optional


class StepTimer:
    """High-resolution wall-clock timer tracking per-step latencies, throughput, and GPU stats."""

    def __init__(self, output_dir: str, phase_name: str = "train"):
        self.output_dir = output_dir
        self.phase_name = phase_name
        self.step_timings: List[Dict[str, Any]] = []
        self._current_step_t0: Optional[float] = None
        self._total_start_time: float = time.time()
        os.makedirs(output_dir, exist_ok=True)

    def start_step(self):
        self._current_step_t0 = time.perf_counter()

    def end_step(self, step: int, loss: float, lr: float, tokens_processed: int = 0) -> float:
        if self._current_step_t0 is None:
            return 0.0
        elapsed_ms = (time.perf_counter() - self._current_step_t0) * 1000.0
        self.step_timings.append({
            "step": step,
            "elapsed_ms": round(elapsed_ms, 3),
            "loss": round(float(loss), 6),
            "lr": round(float(lr), 8),
            "tokens": tokens_processed,
            "tokens_per_sec": round((tokens_processed / (elapsed_ms / 1000.0)), 2) if elapsed_ms > 0 and tokens_processed > 0 else 0.0,
            "timestamp": round(time.time(), 3)
        })
        self._current_step_t0 = None
        return elapsed_ms

    def export_csv(self, filename: str = "step_metrics.csv") -> str:
        csv_path = os.path.join(self.output_dir, filename)
        if not self.step_timings:
            return csv_path
        keys = ["step", "elapsed_ms", "loss", "lr", "tokens", "tokens_per_sec", "timestamp"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.step_timings)
        return csv_path

    def get_summary(self) -> Dict[str, Any]:
        total_time = time.time() - self._total_start_time
        total_steps = len(self.step_timings)
        avg_ms = sum(t["elapsed_ms"] for t in self.step_timings) / max(1, total_steps)
        total_tokens = sum(t.get("tokens", 0) for t in self.step_timings)
        tok_per_sec = total_tokens / max(0.001, total_time)
        return {
            "phase": self.phase_name,
            "total_steps": total_steps,
            "total_wall_clock_seconds": round(total_time, 2),
            "avg_ms_per_step": round(avg_ms, 2),
            "tokens_per_second": round(tok_per_sec, 2),
            "total_tokens": total_tokens
        }
