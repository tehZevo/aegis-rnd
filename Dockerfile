FROM python:3.7

WORKDIR /app

# RUN mkdir -p /app/models

RUN apt-get update
RUN apt-get install git -y
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 80
# VOLUME ["/app/models"]

CMD [ "python", "-u", "main.py" ]
