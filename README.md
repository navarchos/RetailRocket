### Session-based Recommendation System (GRU4Rec)

Pet-проект по разработке сессионной рекомендательной системы для e-commerce данных на основе датасета  **RetailRocket** .

### Цель проекта

Построить воспроизводимый ML-пайплайн для **next-item recommendation** в условиях отсутствия пользовательских профилей и сравнить влияние различных этапов предобработки и архитектурных решений на качество рекомендаций.

### Что реализовано

* Исследование данных и формирование гипотез (`01_EDA.ipynb`)
* Предобработка событийных логов:
  * построение сессий
  * кодирование item и event типов
  * корректное разбиение train / validation по времени
* Генерация обучающих последовательностей для session-based моделей
* Реализация и обучение **GRU4Rec** (PyTorch)
* Оценка качества по метрикам **Recall@20** и **MRR@20**

### Текущие результаты

* Recall@20 ≈ **0.28**
* MRR@20 ≈ **0.17**

### Практическая направленность

Проект воспроизводит индустриальный сценарий рекомендательных систем для e-commerce и фокусируется на корректной работе с сессионными данными, метриками качества и отсутствием data leakage.

### Стек

Python, PyTorch, NumPy, Pandas, Jupyter Notebook

### Как запустить

Через библиотеку kaggle

pip install -r requirements.txt

kaggle datasets download -d retailrocket/ecommerce-dataset -p ./data/raw --unzip

python -m src.data.preprocessing

python -m src.data.sessions

python -m src.training.train_gru
