FROM nvidia/cuda:10.2-cudnn8-devel
FROM continuumio/anaconda3

WORKDIR /app

# Create the environment:

# Make RUN commands use the new environment:
RUN conda create --name myenv python=3.6
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

COPY requirements-docker.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements-docker.txt

# The code to run when container is started:
COPY . .
EXPOSE 5000
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "myenv", "python", "app_detection.py"]