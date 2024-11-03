FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY frontend.py .

EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "frontend.py"]