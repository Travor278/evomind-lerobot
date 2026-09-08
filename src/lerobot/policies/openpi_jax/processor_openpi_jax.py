"""Identity processors for a server-side OpenPI policy."""

from __future__ import annotations

from typing import Any

import torch

from lerobot.processor import (
    DeviceProcessorStep,
    RenameObservationsProcessorStep,
    make_policy_processor_pipelines,
)

from .configuration_openpi_jax import OpenPIJAXConfig


def make_openpi_jax_pre_post_processors(
    config: OpenPIJAXConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[Any, Any]:
    """Rename camera keys locally; normalization and resizing stay on the OpenPI server."""
    del config, dataset_stats
    return make_policy_processor_pipelines(
        input_steps=[RenameObservationsProcessorStep(), DeviceProcessorStep(device="cpu")],
        output_steps=[],
    )
