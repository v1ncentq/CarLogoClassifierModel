# Система розпізнавання автотранспорту методами глибокого навчання

## Опис проєкту

Цей проєкт реалізує систему розпізнавання автомобільних брендів за зображеннями їхніх емблем із використанням методів глибокого навчання.

Основна задача системи — виконати багатокласову класифікацію зображення логотипа автомобіля та визначити, до якого бренду воно належить.

Проєкт реалізовано мовою програмування **Python** із використанням бібліотеки **PyTorch**.  
Для навчання використовується датасет із зображеннями автомобільних емблем. Зображення попередньо обробляються, переводяться у формат grayscale та подаються на вхід згортковій нейронній мережі.

Важливою особливістю проєкту є те, що модель не використовує кольорові RGB-підказки. Усі зображення переводяться у відтінки сірого, тому нейронна мережа навчається розпізнавати бренди за формою, контуром і структурними особливостями емблем.

---

## Мета проєкту

Метою проєкту є створення програмної системи, яка здатна автоматично розпізнавати автомобільний бренд за зображенням його емблеми за допомогою згорткової нейронної мережі.

---

## Основні можливості

Проєкт підтримує:

- підготовку датасету;
- переведення зображень у grayscale;
- зміну розміру зображень;
- розбиття даних на train / validation / test;
- навчання власної CNN-моделі;
- використання ResNet18, адаптованої під grayscale-зображення;
- оцінювання моделі на тестовій вибірці;
- побудову confusion matrix;
- збереження метрик навчання;
- прогнозування класу для нового зображення;
- виведення top-k прогнозів.

---

## Технології

У проєкті використовуються:

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- scikit-learn
- Pillow
- PyYAML
- tqdm

---

## Структура проєкту

```text
Diploma/
│
├── configs/
│   ├── data_config.yaml
│   ├── inference_config.yaml
│   ├── model_config.yaml
│   └── train_config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── notebooks/
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_training_results.ipynb
│   └── 03_error_analysis.ipynb
│
├── reports/
│   ├── figures/
│   └── metrics/
│
├── saved_models/
│   └── best/
│
├── scripts/
│   ├── evaluate_model.py
│   ├── prepare_data.py
│   ├── run_inference.py
│   └── train_model.py
│
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── loader.py
│   │   ├── preprocess.py
│   │   ├── split.py
│   │   └── transforms.py
│   │
│   ├── evaluation/
│   │   ├── confusion_matrix.py
│   │   └── evaluate.py
│   │
│   ├── inference/
│   │   └── predictor.py
│   │
│   ├── models/
│   │   ├── cnn_model.py
│   │   └── resnet_gray.py
│   │
│   ├── training/
│   │   ├── losses.py
│   │   ├── optimizer_factory.py
│   │   └── trainer.py
│   │
│   ├── utils/
│   │   ├── device.py
│   │   ├── file_manager.py
│   │   └── seed.py
│   │
│   └── visualization/
│       ├── plots.py
│       └── training_curves.py
│
├── main.py
├── requirements.txt
└── README.md