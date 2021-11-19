FROM python:3.8.8-slim-buster

WORKDIR /
COPY . /

RUN apt-get update

CMD python --version
