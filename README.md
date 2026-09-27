# Size Measurer

Программно-аппаратный комплекс измерения габаритов товаров на конвейере
по облаку точек. Проект — DevOps-обвязка вокруг алгоритма из отчёта по
треку «Компьютерное зрение».

## Идея

1. Стереокамеры и энкодер дают облако точек товара в системе координат
   конвейера (`X` — поперёк, `Y` — вдоль, `Z` — вверх, `Z = 0` — лента).
2. Пайплайн обрабатывает облако:
   RANSAC → SOR → воксель → DBSCAN → минимальный 2D-OBB → JSON.
3. Результат уходит в WMS. Логи пишутся в `logs/measurements.jsonl`.
4. Airflow запускает измерение каждые 3 секунды.
5. Streamlit показывает последние измерения за час с цветовой
   индикацией статуса.

## Статусы

| Цвет | Статус | Условие |
|------|--------|---------|
| серый | `ok` | размеры в диапазоне, `confidence ≥ 0.8` |
| жёлтый | `suspicious` | `0.5 ≤ confidence < 0.8` |
| красный | `recheck` | размеры вне `[10, 400]` мм или `confidence < 0.5` |

## Структура

```
src/            — пайплайн, генератор, метрики, логи
dags/           — Airflow DAG (3 с)
streamlit_app/  — дашборд
tests/          — pytest
logs/           — measurements.jsonl и app.log
```

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Запуск

### 1. Сгенерировать облако точек

```bash
python src/generate_cloud.py --out cloud.npz --seed 42
```

### 2. Прогнать пайплайн

```bash
python src/pipeline.py --input cloud.npz --log-measurements
```

Или без файла:

```bash
python src/pipeline.py --random --log-measurements
```

### 3. Streamlit

```bash
streamlit run streamlit_app/app.py
```

Откройте <http://localhost:8501>.

### 4. Airflow

```bash
export AIRFLOW_HOME=$(pwd)/airflow
airflow db init
airflow dags list
airflow scheduler &
airflow webserver --port 8080 &
```

DAG `size_measurer` начнёт писать измерения каждые 3 секунды.

> Планировщик Airflow по умолчанию имеет точность до минуты.
> Для интервала 3 с установите
> `AIRFLOW__SCHEDULER__MIN_FILE_PROCESS_INTERVAL=1` и используйте
> Celery/Dask executor.

### 5. Docker

```bash
docker compose up --build
```

## Метрики

```bash
python src/metrics.py --measurements logs/measurements.jsonl \
                      --ground-truth ground_truth.json
```

Считает MAE, RMSE и hit-rate для `L, W, H` по допуску
`max(5%, 5 мм)`.

## Тесты

```bash
pytest -q
```

## Полезные замечания

- Параметры RANSAC/DBSCAN подбираются под плотность облака.
- При прозрачных и блестящих товарах нужны УФ-подсветка,
  поляризация или SWIR.
- Для наклонённых товаров 2D-OBB заменяется на 3D-OBB
  (`open3d.geometry.OrientedBoundingBox`).
- Логи ротируются штатными средствами ОС.