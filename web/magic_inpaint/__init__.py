from flask import Flask
from os import environ

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = environ.get('SECRET_KEY', 'Random charachter')


from magic_inpaint import routes