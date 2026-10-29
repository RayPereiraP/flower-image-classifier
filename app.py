from pathlib import Path

import gradio as gr

from src.predict import load_model, predict

MODEL_PATH = Path(__file__).parent / "outputs" / "model.pt"
if not MODEL_PATH.exists():
    raise SystemExit("Rode `python train.py` primeiro para gerar o modelo.")

model, class_names = load_model(MODEL_PATH)


def classify(image):
    if image is None:
        return {}
    return predict(model, class_names, image)


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Envie uma foto de flor"),
    outputs=gr.Label(num_top_classes=3, label="Predição"),
    title="🌸 Classificador de flores",
    description="Transfer learning com MobileNetV2 pré-treinada na ImageNet.",
)

if __name__ == "__main__":
    demo.launch()