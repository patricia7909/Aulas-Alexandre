FROM python:3.13-slim

# Evita arquivos .pyc e faz os logs aparecerem na hora
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copiamos o requirements primeiro: se só o código mudar, o Docker
# reaproveita a camada com as dependências já instaladas (build mais rápido)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY modelos/ modelos/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]