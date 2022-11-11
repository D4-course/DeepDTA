FROM tiangolo/uvicorn-gunicorn:python3.8-slim
COPY ./requirements.txt ./requirements.txt
RUN pip3 install -r ./requirements.txt

COPY . ./

RUN chmod +x entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]
