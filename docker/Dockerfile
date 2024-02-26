FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive

RUN echo "==> Upgrading apk and installing system utilities ...." \
 && apt -y update \
 && apt-get install -y wget \
 && apt-get update \
 && apt-get -y install sudo

RUN echo "==> Installing Python3 and pip ...." \  
 && apt-get install python3 -y \
 && apt install python3-pip -y

RUN echo "==> Install dos2unix..." \
  && sudo apt-get install dos2unix -y 

RUN echo "==> Install openai" \
  && pip install --upgrade openai

RUN echo "==> Install cv2.." \
  && pip install opencv-python

RUN echo "==> Install Pillow.." \
  && pip install Pillow

RUN echo "==> Install requests.." \
  && pip install requests

RUN echo "==> Install python dotenv" \
  && pip install python-dotenv

RUN echo "==> Install IPython" \
  && pip install IPython

RUN echo "==> Install streamlit" \
  && pip install streamlit --upgrade

RUN echo "==> Install required library" \
  && sudo apt-get install libgl1-mesa-glx -y \
  && sudo apt-get install libglib2.0-0 -y

RUN echo "==> Install ffmpeg" \
  && apt-get install -y ffmpeg

RUN echo "==> Install pytube" \
  && pip install pytube

RUN echo "==> Install moveipy" \
  && pip install moviepy

COPY /vaideo /vaideo/
COPY /scripts /scripts/

RUN echo "==> Convert script..." \
  && dos2unix /scripts/startup.sh

CMD ["/bin/bash", "/scripts/startup.sh"]