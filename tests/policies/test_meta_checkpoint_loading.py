from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from lerobot.policies.pretrained import load_safetensors_into_meta_model


class _TinyPolicy(nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.proj = nn.Linear(3, 2, bias=True, dtype=dtype)


def test_meta_checkpoint_parameters_are_assigned_and_cast(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.safetensors"
    weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    bias = torch.tensor([7.0, 8.0])
    save_file({"proj.weight": weight, "proj.bias": bias}, checkpoint)

    with torch.device("meta"):
        model = _TinyPolicy(dtype=torch.bfloat16)

    load_safetensors_into_meta_model(
        model,
        checkpoint,
        "cpu",
        strict=True,
        remap_state_dict=lambda state: state,
    )

    assert model.proj.weight.device.type == "cpu"
    assert model.proj.weight.dtype == torch.bfloat16
    torch.testing.assert_close(model.proj.weight.float(), weight)
    torch.testing.assert_close(model.proj.bias.float(), bias)


def test_meta_checkpoint_rejects_unmaterialized_parameters(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.safetensors"
    save_file({"proj.weight": torch.ones(2, 3)}, checkpoint)

    with torch.device("meta"):
        model = _TinyPolicy()

    with pytest.raises(RuntimeError, match="unmaterialized meta tensors: proj.bias"):
        load_safetensors_into_meta_model(
            model,
            checkpoint,
            "cpu",
            strict=False,
            remap_state_dict=lambda state: state,
        )


def test_remapped_tied_weights_share_checkpoint_storage(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.safetensors"
    save_file({"source": torch.arange(4, dtype=torch.float32)}, checkpoint)

    with torch.device("meta"):
        model = nn.Module()
        model.first = nn.Parameter(torch.empty(4))
        model.second = nn.Parameter(torch.empty(4))

    load_safetensors_into_meta_model(
        model,
        checkpoint,
        "cpu",
        strict=True,
        remap_state_dict=lambda state: {"first": state["source"], "second": state["source"]},
    )

    assert model.first.data_ptr() == model.second.data_ptr()

