# docker build -f ngram_model/Dockerfile . -t ngram

FROM ubuntu:latest

COPY ngram_model /app/ngram_model

COPY data /app/data

WORKDIR /app/ngram_model

RUN apt-get update

RUN apt-get install -y g++ make

RUN apt install -y -V ca-certificates lsb-release wget
RUN wget https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
RUN apt install -y -V ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
RUN apt update
RUN apt install -y -V libarrow-dev
RUN apt install -y -V libparquet-dev

RUN mkdir bin

RUN make

ENTRYPOINT bin/llm