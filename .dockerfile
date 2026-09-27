FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY streamlit_app/ ./streamlit_app/
COPY dags/ ./dags/
ENV PYTHONPATH=/app/src