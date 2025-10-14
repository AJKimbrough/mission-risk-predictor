# (Optional) Use the same api.Dockerfile for both to keep it simple.
FROM python:3.11-slim
WORKDIR /code
COPY requirements.txt /code/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /code
EXPOSE 8501