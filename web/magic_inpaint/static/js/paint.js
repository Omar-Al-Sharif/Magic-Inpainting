const canvas = new fabric.Canvas('canvas', {isDrawingMode: false});
let lastSelectionMode = 'brush';

const img = new Image();
img.onload = function () {
    // Set canvas dimensions to match the image
    canvas.setDimensions({ width: img.width, height: img.height });

    // Set the background image
    canvas.setBackgroundImage(img.src, canvas.renderAll.bind(canvas));
};

img.src = imagePath;
let selectedColorRGB = $('#colorPicker').val();
let rectangle;

const fabricColor = new fabric.Color(selectedColorRGB);
canvas.freeDrawingBrush.color = fabricColor.toRgb();

// Update the color when the user selects a new color
$('#colorPicker').on('input', function () {
    selectedColorRGB = $(this).val();
    
    // Convert the RGB color to fabric.js color representation
    const fabricColor = new fabric.Color(selectedColorRGB);
    
    // Set the color of the drawing brush on the canvas
    canvas.freeDrawingBrush.color = fabricColor.toRgb();

    // Set the color of the rectangle stroke
    if (rectangle) {
        rectangle.set({ stroke: fabricColor.toRgb() });
        canvas.renderAll();
    }
});

canvas.freeDrawingBrush.width = 10;

$('#draw').on('click', function () {
    canvas.isDrawingMode = !canvas.isDrawingMode;
    lastSelectionMode = 'brush';
});

$('#rectangle').on('click', function () {
    lastSelectionMode = 'rectangle';
    canvas.isDrawingMode = false;
    rectangle = new fabric.Rect({
        left: 40,
        top: 40,
        width: 60,
        height: 60,
        fill: 'transparent',
        stroke: fabricColor.toRgb(),
        strokeWidth: 2,
    });
    canvas.add(rectangle);
});

$('#circle').on('click', function () {
    lastSelectionMode = 'circle';
    canvas.isDrawingMode = false;
    const circle = new fabric.Circle({
        left: 40,
        top: 40,
        radius: 60,
        fill: 'transparent',
        stroke: selectedColor,
        strokeWidth: 7,
    });
    canvas.add(circle);
});

$('#text').on('click', function () {
    lastSelectionMode = 'text';
    canvas.isDrawingMode = false;
    const text = new fabric.IText('Text', {
        left: 40,
        top: 40,
        objecttype: 'text',
        fontFamily: 'arial black',
        fill: selectedColor,
    });
    canvas.add(text);
});

$('#remove').on('click', function () {
    canvas.isDrawingMode = false;
    canvas.remove(canvas.getActiveObject());
});

canvas.on('selection:created', function () {
    $('#remove').prop('disabled', '');
});
canvas.on('selection:cleared', function () {
    $('#remove').prop('disabled', 'disabled');
});

$('#download').on('click', function () {
    const resultImage = $('#result img')[0]; // Get the processed image from the result div

    // Check if there's an image
    if (resultImage) {
        // Get the data URL of the image
        const dataURL = resultImage.src;

        // Create a link element to trigger the download
        const link = document.createElement('a');
        link.download = 'image.png';
        link.href = dataURL;

        // Append the link to the document, trigger the click, and remove the link
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
});

$('#test').on('click', function () {
    // Get the data URL of the canvas
    const dataURL = canvas.toDataURL({
        width: canvas.width,
        height: canvas.height,
        left: 0,
        top: 0,
        format: 'png',
    });

    // Use AJAX to send the image data to Flask
    $.ajax({
        type: 'POST',
        url: '/process_image',
        data: {
            image_data: dataURL,
            mode: lastSelectionMode
        },
        success: function(response) {
            console.log('Image sent successfully');

            // Replace the content of the SVG div with the processed image
            $('#result').html('<img src="data:image/png;base64,' + response.processed_image_data + '" alt="Processed Image">');
            $('#mask').html('<img src="data:image/png;base64,' + response.mask_image_data + '" alt="Mask Image">');
            $('#inpainted').html('<img src="data:image/png;base64,' + response.inpainted_image_data + '" alt="Inpainted Image">');
            $('#result1').html('<img src="data:image/png;base64,' + response.processed_image_data + '" alt="Processed Image">');
            $('#resultdata').html('<p> Result</p>');
            $('#maskdata').html('<p> Masked Image</p>');
            $('#inpaintdata').html('<p> Image after inpainting </p>');
            $('#smudgedata').html('<p> Image after smudging</p>');
        },
        error: function(error) {
            console.error('Error sending image: ', error);
        }
    });
});