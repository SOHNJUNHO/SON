FROM python:3.9

RUN apt-get update

RUN apt-get upgrade -y

RUN apt-get install -y python3-pip

RUN pip install jupyter

WORKDIR /data/engineer

COPY requirements.txt /data/engineer

RUN pip install -r /data/app/requirements.txt
