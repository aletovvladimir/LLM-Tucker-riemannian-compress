import os

import hydra
import torch
from omegaconf import DictConfig

from ..utils.model_class import LitModel
from .export_to_onnx import export_to_onnx, verify_onnx

from pathlib import Path


@hydra.main(config_path="../configs", config_name="config")
def main(config: DictConfig):
    device = torch.device('cpu')
    checkpoint_path = Path(config.onnx.ckpt_path) / "last.ckpt"

    model = LitModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path, config=config
        ).model
    print(model)

    model.to(device)

    onnx_path = Path(config.onnx.output_path) / "tucker_model.onnx"
    export_to_onnx(config, model, output_path=onnx_path)
    verify_onnx(onnx_path)

if __name__ == "__main__":
    main()