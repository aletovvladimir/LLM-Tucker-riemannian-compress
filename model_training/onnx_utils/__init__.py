import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from model_training.utils.model_class import LitModel


def export_to_onnx(ckpt_path: str, config_path: str, output_path: str = "model.onnx"):
    config = OmegaConf.load(config_path)
    model = LitModel.load_from_checkpoint(ckpt_path, config=config)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.model_params.model_link)

    dummy = tokenizer(
        "export to onnx",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128,
    )
    dummy_input_ids = dummy["input_ids"]
    dummy_attention_mask = dummy["attention_mask"]

    def onnx_forward(input_ids, attention_mask):
        outputs = model.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    torch.onnx.export(
        onnx_forward,
        (dummy_input_ids, dummy_attention_mask),
        output_path,
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
