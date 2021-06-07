FROM tensorflow/tensorflow:2.4.1-gpu-jupyter

RUN mkdir /app

WORKDIR /app

COPY . .

RUN python -m pip install --upgrade pip
RUN pip install -r requirements-docker.txt

CMD ["python", "app.py"]