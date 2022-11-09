FROM tiangolo/uvicorn-gunicorn:python3.8-slim
FROM continuumio/miniconda3
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

RUN git clone https://github.com/kexinhuang12345/DeepPurpose.git 
RUN cd DeepPurpose
COPY . ./

RUN pip install -r requirements.txt

RUN chmod +x entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]


