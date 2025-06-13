import hydra
import torch

from .utils.utils import get_model
from .utils.data_class import get_tokenizer
from pathlib import Path
from omegaconf import DictConfig


@hydra.main(config_path="configs", config_name="config")
def main(config: DictConfig):
    device = config.inference.device

    model = get_model(config).model.to(device)
    model.eval()

    tokenizer = get_tokenizer(config)

    text_path = Path(config.inference.texts_dir) / "review.txt"
    with open(text_path, "r") as f:
        texts = [line.strip() for line in f if line.strip()]

    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=1)

    result_mapping = {0: 'negative',
                      1: 'positive'}
    
    output_path = Path(config.inference.output_dir) / 'prediction.txt'

    with open(output_path, "w") as out_file:
        for text, pred, prob in zip(texts, preds.cpu(), probs.cpu()):
            out_file.write(f"Text: {text}\n")
            out_file.write(f"Prediction: {result_mapping[pred.item()]} (Prob: {prob[pred].item():.4f})\n")
            out_file.write("=" * 50 + "\n")
    
if __name__ == '__main__':
    main()
        
        
    