"""Constraint adapters used by the production ECI sampler."""
from __future__ import annotations

from dataclasses import replace

import torch


class TaskProjectionConstraint:
    """Apply a shared or paired constraint-task projection per sample."""

    def __init__(self, task, config, use_native: bool = True):
        self.task = task
        self.config = config
        self.use_native = use_native
        self.diagnostics: list[dict] = []

    def adjust(self, u: torch.Tensor) -> torch.Tensor:
        from sampling.projectors import damped_gauss_newton_projection

        if isinstance(self.task, (list, tuple)) and len(self.task) != len(u):
            raise ValueError("Paired task count does not match ECI batch size")
        projected = []
        for sample_index, sample in enumerate(u):
            task = (
                self.task[sample_index]
                if isinstance(self.task, (list, tuple))
                else self.task
            )
            flat = sample.reshape(-1)
            if self.use_native:
                result = task.project(flat, self.config)
            else:
                value = flat.to(task.device, task.scales.dtype)
                result = damped_gauss_newton_projection(
                    value,
                    task.normalized_residual,
                    replace(self.config, scales=None),
                )
                result.u = result.u.to(flat.device, flat.dtype)
            projected.append(result.u.reshape_as(sample))
            self.diagnostics.append({
                "sample_index": sample_index,
                "converged": bool(result.converged),
                "feasible": bool(result.feasible),
                "iterations": int(result.iterations),
                "initial_residual_norm": float(result.initial_residual_norm),
                "final_residual_norm": float(result.final_residual_norm),
                "condition_number": result.condition_number,
                "method": result.method,
                "warnings": list(result.warnings),
            })
        return torch.stack(projected, dim=0)
