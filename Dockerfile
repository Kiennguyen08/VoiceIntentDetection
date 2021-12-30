FROM tensorflow/tensorflow:2.4.1

WORKDIR /usr/src/app

# RUN apt-get -y update  && apt-get install -y \
#   python3-dev \
#   apt-utils \
#   python-dev \
#   build-essential \
# && rm -rf /var/lib/apt/lists/*

# RUN pip install --upgrade setuptools
# RUN pip install \
#     numpy==1.21.1 \
#     pandas==1.3.1 

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y ca-certificates ffmpeg libsm6 libxext6

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# CMD gunicorn -w 3 -k uvicorn.workers.UvicornWorker server-fai:app --bind 0.0.0.0:$PORT
EXPOSE 8000

CMD  uvicorn server-fai:app --reload --host 0.0.0.0