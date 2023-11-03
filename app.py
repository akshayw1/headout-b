from flask import Flask, request
import os

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
    folder_name = request.form['folderName']
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    file = request.files['image']
    if file:
        file.save(os.path.join(folder_name, file.filename))
        return "Image saved successfully!"
    return "Image upload failed."

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
