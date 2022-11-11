FROM tiangolo/uvicorn-gunicorn:python3.8-slim
COPY ./requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r ./requirements.txt

COPY . ./
RUN chmod +x entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]
