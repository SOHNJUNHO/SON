FROM python:3.9

RUN apt-get update

RUN apt-get upgrade -y

RUN apt-get install -y python3-pip

WORKDIR /data/app

COPY train.csv Randomforest.py requirements.txt /data/app

RUN pip install -r /data/app/requirements.txt

CMD ["python", "Randomforest.py"] 
