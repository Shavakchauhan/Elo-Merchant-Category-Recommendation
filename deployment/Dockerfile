FROM python:3.10.9
COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt
RUN pip3 install certifi
RUN pip3 install gunicorn
EXPOSE 5000
CMD [ "gunicorn", "--bind", "0.0.0.0:5000", "app:app" ]


