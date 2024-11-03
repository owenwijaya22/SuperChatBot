FROM python:3.12-slim as base

WORKDIR /app

COPY requirements.txt /app/

RUN pip install -r requirements.txt

COPY . /app/

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]