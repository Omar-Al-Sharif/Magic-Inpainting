from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField

class UploadForm(FlaskForm):
    image = FileField('Select the target image', validators=[
        FileRequired(),
        FileAllowed(['jpg', 'png', 'jpeg'], "'jpg', 'png', 'jpeg' Images only!")
    ])
    submit = SubmitField('Upload')
