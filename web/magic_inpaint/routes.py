from flask import render_template, url_for, flash, redirect, request, session, jsonify
from PIL import Image
from magic_inpaint import app
from magic_inpaint.forms import UploadForm
from werkzeug.utils import secure_filename
from magic_inpaint.region_detection import *
from magic_inpaint.commonfunctions import *
from magic_inpaint.inpainting import *
from magic_inpaint.postprocessing import *
import os
import base64
import imageio

@app.route("/")
@app.route("/index")
def index():
    return render_template('index.html', title='Home')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    form = UploadForm()
    if form.validate_on_submit():

        uploaded_file = form.image.data

        image = Image.open(uploaded_file)
        new_size = (256, 256)
        resized_image = image.resize(new_size)

        directory = os.path.join('magic_inpaint', 'static', 'img')
        file_path = os.path.join(directory, 'temp.png')
        resized_image.save(file_path)
        session['file_path'] = f'/img/temp.png'


        # Resize the image to 256x256
        # Open the uploaded image using Pillow
        image = Image.open(file_path)

        # Resize the image to be 256x256 pixels
        new_size = (256, 256)
        resized_image = image.resize(new_size)

        # Save the resized image
        resized_file_path = os.path.join(directory, 'resized_temp.png')
        resized_image.save(resized_file_path)

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
        brushing_mode = request.form['mode']
        # Decode the base64-encoded image data
        image_binary = base64.b64decode(image_data)

        img_file_name = 'selected_image.png'

        with open(img_file_name, 'wb') as f:
            f.write(image_binary)

        if brushing_mode == 'brush':
            processed_image = process_image_brush()
        else:
            processed_image = process_image_rectangle()

        # # Convert the processed image to base64 for sending back to the client
        processed_image_data = base64.b64encode(processed_image).decode('utf-8')

        return jsonify({'status': 'success', 'processed_image_data': processed_image_data})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})
    
def process_image_brush():

    processed_image = io.imread('selected_image.png')
    original_image = io.imread('magic_inpaint/static/img/temp.png')
    target_img_size = (256, 256)
    processed_image_resized = cv2.resize(processed_image, target_img_size)
    processed_image_conversion = processed_image_resized[:,:,:3]
    original_image_resized = cv2.resize(original_image[:,:,:3], target_img_size)

    original_image, dark_mask, cleaned_dark_mask= BrushedRegionDetection(original_image_resized, processed_image_conversion).get_desired_masks()

    inpainted_image = main(original_image_resized,cleaned_dark_mask)
    result = postprocessing(inpainted_image, cleaned_dark_mask)

    cleaned_dark_mask = np.array(cleaned_dark_mask, dtype=np.uint8) * 255
    img = Image.fromarray(cleaned_dark_mask, mode='L')
    img.save('mask.png')

    io.imsave('inpainted.png', inpainted_image)
    io.imsave('result.png', result)

    result_image_file_name = 'result.png'

    with open(result_image_file_name, 'rb') as image_file:
        image_content = image_file.read()
    
    return image_content

def process_image_rectangle():

    processed_image = io.imread('selected_image.png')
    original_image = io.imread('magic_inpaint/static/img/temp.png')
    target_img_size = (256, 256)
    processed_image_resized = cv2.resize(processed_image, target_img_size)
    processed_image_conversion = processed_image_resized[:,:,:3]
    original_image_resized = cv2.resize(original_image[:,:,:3], target_img_size)

    region_detector_1 = RegionDetection(original_image_resized, processed_image_conversion)
    region_mask, binary_mask_test=region_detector_1.get_mask_by_region_detection()
    
    binary_mask_test = binary_mask_test[:,:,0]
    inpainted_image= main(original_image_resized,binary_mask_test)
    result = postprocessing(inpainted_image, binary_mask_test)


    binary_mask_test = np.array(binary_mask_test, dtype=np.uint8) * 255
    img = Image.fromarray(binary_mask_test, mode='L')
    img.save('mask.png')

    io.imsave('inpainted.png', inpainted_image)
    io.imsave('result.png', result)

    result_image_file_name = 'result.png'

    with open(result_image_file_name, 'rb') as image_file:
        image_content = image_file.read()
    
    return image_content