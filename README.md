# 📚 LLM-Tucker-Riemannian-Compress

🔎 Реализация сжатия LLM-моделей с использованием тензорного разложения Такера и римановой оптимизации. Обучение классификатора на датасете IMDb.

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

📌 Для установки зависимостей для разработчиков:

```bash
uv pip install --group dev
```

---

## 📂 Датасет

* Используется датасет `imdb` из библиотеки `datasets`.

---

## ⬇️ Получение данных (DVC)

📆 Проект использует [DVC](https://dvc.org/) и Google Drive для хранения артефактов.

📦 Чтобы скачать необходимые файлы:

```bash
dvc pull
```

📌 **Важно:** используется Google service account.

1. Запросите у автора `gdrive-credentials.json`
2. Сохраните его в корневой папке проекта `.dvc`
3. Убедитесь, что в `.dvc/config` указано:

```ini
[core]
    remote = gdrive
['remote "gdrive"']
    url = gdrive://1-GW_D8d4mxCQbwXCj51D9sLAnYW4ackr
    gdrive_use_service_account = true
    gdrive_service_account_file = gdrive-credentials.json
```

4. Добавьте `gdrive-credentials.json` в `.gitignore`

---

## ✅ Запуск из `src/`

```bash
cd src
```

---

## 🏃 Обучение

```bash
python -m training_and_inference.train
```

* Чекпойнты падают в `model_checkpoints/`
* Гиперпараметры: `configs/config.yaml`

---

## 📊 MLflow

* `train_loss`, `val_loss`, `val_accuracy`, `lr`

📅 Запуск UI:

```bash
python -m training_and_inference.utils.mlflow_server
```

[http://localhost:8080](http://localhost:8080)

---

## 🧠 Инференс

```bash
python -m training_and_inference.inference
```

* Вход: `../texts/review.txt`
* Выход: `prediction.txt`

💬 Пример:

```
Text: Interstellar is a visually stunning masterpiece...
Prediction: positive (Prob: 0.9997)
==================================================
```

---

## 📦 ONNX Экспорт

```bash
python -m training_and_inference.onnx_utils.convert_and_export
```

* Экспорт: `onnx-model/tucker_model.onnx`

---

## 🗂️ Структура проекта

```
├── src/
│   ├── compress/              # Тензорная алгебра
│   ├── model_compression/     # Сжатие + риманова оптимизация
│   └── training_and_inference/ # Обучение, инференс, ONNX, MLflow
├── compression_tests/         # Тесты
├── texts/                     # Тексты для инференса
├── model_checkpoints/         # Чекпойнты
├── onnx-model/                # ONNX-модели
├── plots/                     # Hydra и MLflow логи
├── pyproject.toml             # Зависимости
├── dvc.yaml                   # DVC пайплайн
```

---

## 🛠️ Зависимости

* Все зависимости указаны в `pyproject.toml`
* Установка через `uv`

---

## 📍 License

Apache License.
