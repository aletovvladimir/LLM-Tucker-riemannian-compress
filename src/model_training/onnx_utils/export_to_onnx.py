import torch
from omegaconf import DictConfig
import torch.nn as nn
from transformers import AutoTokenizer

import onnx
import torch.nn as nn


class ONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

def export_to_onnx(config: DictConfig, model: nn.Module, output_path: str):
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.model_params.model_link)

    dummy = tokenizer(
        "I love sci-fi and am willing to put up with a lot...",
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",  # важно для onnx
    )

    dummy_input_ids = dummy["input_ids"]
    dummy_attention_mask = dummy["attention_mask"]

    onnx_model = ONNXWrapper(model.model)  # model.model = nn.Module внутри LitModel

    torch.onnx.export(
        onnx_model,
        (dummy_input_ids, dummy_attention_mask),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size"},
        },
        opset_version=13,
    )

    print(f"[✓] Exported to ONNX: {output_path}")


def verify_onnx(output_path):
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"[✓] ONNX model is valid: {output_path}")
    return onnx_model