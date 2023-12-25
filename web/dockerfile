FROM python:3.11

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt

# You will need to add docker ignore file to ignore the .env file and other files that you don't want to be copied to the docker image
COPY . .

EXPOSE 5000

CMD [ "flask", "run", "--host=0.0.0.0", "--port=5000" ]