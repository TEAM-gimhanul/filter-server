FROM python:3.7-slim

COPY . ./app

RUN pip install -r requirements.txt

EXPOSE 8085

CMD python ./app.py