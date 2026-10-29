# Flower Image Classifier

Classificador de imagens de flores usando transfer learning com MobileNetV2 e demo interativa em Gradio.

## Dataset
[Flowers Recognition (Kaggle)](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)
5 classes: daisy, dandelion, rose, sunflower e tulip. Extraia em `data/flowers/`.

## Abordagem
- MobileNetV2 pré-treinada na ImageNet como extrator de features (backbone congelado)
- Nova camada final treinada para as 5 classes
- Data augmentation no treino (recorte aleatório e flip horizontal)
- Split de 80% treino e 20% validação com seed fixa
- Por que transfer learning: poucos dados e pouco poder de processamento, então reaproveitar o que a rede já aprendeu em milhões de imagens é muito mais eficiente do que treinar do zero

## Como rodar
```bash
pip install -r requirements.txt
python train.py --epochs 3
python app.py
```

## Resultados
| Métrica | Valor |
|---|---|
| Acurácia (validação) | XX |

![Matriz de confusão](docs/confusion_matrix.png)
![Demo](docs/demo.png)

## Stack
Python, PyTorch, torchvision, Gradio, scikit-learn

## Próximos passos
- Fine-tuning das últimas camadas do backbone
- Deploy no Hugging Face Spaces