FROM python:3.11

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r /code/requirements.txt

EXPOSE 8002

ENV PORT 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]