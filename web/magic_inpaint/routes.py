from flask import render_template, url_for, flash, redirect, request, session, jsonify
from magic_inpaint import app
from magic_inpaint.forms import UploadForm
from werkzeug.utils import secure_filename
import os
import base64

@app.route("/")
@app.route("/index")
def index():
    return render_template('index.html', title='Home')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    form = UploadForm()
    if form.validate_on_submit():

        uploaded_file = form.image.data
        directory = os.path.join('magic_inpaint', 'static', 'img')
        file_path = os.path.join(directory, 'temp.png')
        uploaded_file.save(file_path)
        session['file_path'] = f'/img/temp.png'
        return redirect(url_for('paint'))

    return render_template('upload.html', form=form)

@app.route('/paint', methods=['GET', 'POST'])
def paint():

    file_path = session.get('file_path', None)

    if not file_path:
        flash('Image not found, please upload an image first!', 'danger')
        return redirect(url_for('upload'))

    return render_template('paint.html', file_path=file_path)

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # Get the image data from the AJAX request
        image_data = request.form['image_data'].split(',')[1]
        
        # Decode the base64-encoded image data
        image_binary = base64.b64decode(image_data)

        img_file_name = 'saved_image.png'

        with open(img_file_name, 'wb') as f:
            f.write(image_binary)

        # # Perform image processing (e.g., using PIL)
        processed_image = process_image_function(img_file_name)

        # # Convert the processed image to base64 for sending back to the client
        processed_image_data = base64.b64encode(processed_image).decode('utf-8')

        return jsonify({'status': 'success', 'processed_image_data': processed_image_data})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})
    
def process_image_function(img_file_name):

    # Add the code of image processing here, you have the image path, and I want you to add the processed image in a file named 'saved_image.png'
    processed_image_file_name = 'saved_image.png'

    with open(processed_image_file_name, 'rb') as image_file:
        image_content = image_file.read()
    
    return image_content