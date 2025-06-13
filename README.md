# 📚 LLM-Tucker-Riemannian-Compress

Реализация сжатия LLM-моделей с использованием тензорного разложения Такера и римановой оптимизации. Обучение классификатора на датасете IMDb.

---

## ⚙️ Установка

1. **Клонируйте репозиторий через SSH:**

```bash
git clone git@github.com:aletovvladimir/LLM-Tucker-riemannian-compress.git
cd LLM-Tucker-riemannian-compress
```

2. **Создайте окружение через Conda:**

```bash
conda create -n llm-compress python=3.10.12 -y
conda activate llm-compress
```

3. **Установите зависимости с помощью uv:**

```bash
pip install uv
uv pip install .
```

📌 Для установки зависимостей для разработки:

```bash
uv pip install --group dev
```

---

## 📂 Датасет

* Используется датасет `imdb` из библиотеки `datasets`.

---

## 🏃 Обучение

```bash
python -m src.training_and_inference.train
```

* Чекпойнты сохраняются в `model_checkpoints/`
* Гиперпараметры конфигурируются через Hydra (`configs/config.yaml`)

---

## 📊 MLflow

Логируются:

* `train_loss`
* `val_loss`
* `val_accuracy`
* `lr`

Запуск MLflow UI:

```bash
python -m src.training_and_inference.utils.mlflow_server
```

Откройте: [http://localhost:8080](http://localhost:8080)

---

## 🧠 Инференс

```bash
python -m src.training_and_inference.inference
```

* Ожидается файл: `texts/review.txt`
* Результаты сохраняются в `prediction.txt`

### 💬 Пример вывода:

```
Text: Interstellar is a visually stunning masterpiece...
Prediction: positive (Prob: 0.9997)
==================================================
```

---

## 📦 ONNX Экспорт

```bash
python -m src.training_and_inference.onnx_utils.convert_and_export
```

* Экспорт: `onnx-model/tucker_model.onnx`
* Валидация ONNX выполняется автоматически

---

## 🗂️ Структура проекта

```
├── src/
│   ├── compress/              # Тензорная алгебра
│   ├── model_compression/     # Сжатие + риманова оптимизация
│   └── model_training/        # Обучение, инференс, ONNX, MLflow
├── compression_tests/         # Тесты
├── texts/                     # Тексты для инференса
├── model_checkpoints/         # Чекпойнты
├── onnx-model/                # ONNX-модели
├── plots/                     # Hydra и MLflow логи
├── pyproject.toml             # Зависимости
```

---

## 🛠️ Зависимости

* Все зависимости указаны в `pyproject.toml`

---

## 📝 License

Apache License.
