from flask import Flask, request
import os
from PIL import Image
import io
import base64
import time
app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
    folder_name = request.form['folderName']
    if not folder_name: folder_name = "error"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    fileContent = request.form['image']
    if fileContent:
        header, encoded = fileContent.split(',', 1)

        # Determine the file extension (e.g., jpg, png)
        ext = header.split('/')[1].split(';')[0]

        # Decode the base64 string into bytes
        image_bytes = base64.b64decode(encoded)

        # Create a PIL image from the decoded bytes
        img = Image.open(io.BytesIO(image_bytes))

        # Save the image to a file (change the file name and path as needed)
        img.save(folder_name + "/" + folder_name + "-" + str(time.time()) + "." + ext)
        return "Image saved successfully!"
    return "Image upload failed."

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
