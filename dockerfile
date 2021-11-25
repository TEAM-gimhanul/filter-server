FROM python:3.7

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8085

RUN apt-get update && apt-get install -y locales git

RUN localedef -f UTF-8 -i ko_KR ko_KR.UTF-8
ENV LC_ALL ko_KR.UTF-8
ENV PYTHONIOENCODING=utf-8

RUN pip install h5py==2.10.0

CMD python ./app.py