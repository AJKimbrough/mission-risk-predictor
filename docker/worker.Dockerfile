FROM python:3.11-slim
WORKDIR /code
COPY requirements.txt /code/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /code
CMD ["python","-m","app.etl.nws_grid","--lat","32.7767","--lon","-96.7970"]