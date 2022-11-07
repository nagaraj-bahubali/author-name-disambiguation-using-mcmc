FROM python:3.8-slim-buster
ADD requirements.txt /
RUN pip --no-cache-dir install -r requirements.txt
ADD . /
CMD [ "python3", "main.py" ]