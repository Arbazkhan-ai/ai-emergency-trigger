from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1

model = pipeline(
    "text-classification",
    model="emergency_model",
    tokenizer="emergency_model",
    device=device,
    top_k=None
)

id2label = {
    0: "LOW RISK",
    1: "MEDIUM RISK",
    2: "HIGH RISK"
}

texts = [
    "Patient not breathing",
    "Severe chest pain",
    "Fever for 2 days",
    "Mild headache",
    "Need hospital address"
]

for text in texts:
    outputs = model(text)
    scores = outputs[0] if isinstance(outputs[0], list) else outputs
    best = max(scores, key=lambda x: x["score"])

    label_id = int(best["label"].split("_")[-1])

    print(f"\nText: {text}")
    print(f"Prediction: {id2label[label_id]}")
    print(f"Confidence: {best['score']:.4f}")