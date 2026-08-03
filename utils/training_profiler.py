from collections import defaultdict
from contextlib import contextmanager

import torch


class TrainingProfiler:
    """Low-overhead phase profiler that synchronizes only at report intervals."""

    def __init__(self, enabled=True, interval=100, warmup=10):
        self.enabled = bool(enabled) and torch.cuda.is_available()
        self.interval = max(1, int(interval))
        self.warmup = max(0, int(warmup))
        self._cuda_events = defaultdict(list)
        self._cpu_totals = defaultdict(float)
        self._cpu_counts = defaultdict(int)
        self._iterations = 0
        self._latest_summary = {}

    @contextmanager
    def cuda_phase(self, name):
        if not self.enabled:
            yield
            return

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._cuda_events[str(name)].append((start, end))

    def start_cuda_phase(self):
        if not self.enabled:
            return None
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        return start

    def stop_cuda_phase(self, name, start):
        if not self.enabled or start is None:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._cuda_events[str(name)].append((start, end))

    def record_cuda_events(self, name, start, end):
        if not self.enabled or start is None or end is None:
            return
        self._cuda_events[str(name)].append((start, end))

    def record_cpu(self, name, seconds):
        if not self.enabled:
            return
        self._cpu_totals[str(name)] += max(0.0, float(seconds))
        self._cpu_counts[str(name)] += 1

    def finish_iteration(self, iteration, logger=None):
        if not self.enabled:
            return None
        if int(iteration) <= self.warmup:
            self.reset()
            return None

        self._iterations += 1
        if self._iterations < self.interval:
            return None

        torch.cuda.synchronize()
        divisor = max(self._iterations, 1)
        summary = {}
        for name, pairs in self._cuda_events.items():
            total_ms = sum(start.elapsed_time(end) for start, end in pairs)
            summary[f"{name}_ms"] = total_ms / divisor
        for name, total_seconds in self._cpu_totals.items():
            count = max(self._cpu_counts[name], 1)
            summary[f"{name}_ms"] = total_seconds * 1000.0 / count

        summary["iterations"] = self._iterations
        self._latest_summary = summary
        phase_text = " ".join(
            f"{name}={value:.2f}ms"
            for name, value in sorted(summary.items())
            if name.endswith("_ms")
        )
        print(f"\n[Profiler] iteration={iteration} samples={self._iterations} {phase_text}")

        if logger:
            for name, value in summary.items():
                if name == "iterations":
                    continue
                logger.add_scalar(f"profile/{name}", value, iteration)

        self.reset()
        return summary

    @property
    def latest_summary(self):
        return dict(self._latest_summary)

    def reset(self):
        self._cuda_events.clear()
        self._cpu_totals.clear()
        self._cpu_counts.clear()
        self._iterations = 0
