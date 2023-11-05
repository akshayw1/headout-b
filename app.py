from flask import Flask, request , Response
import os
from PIL import Image
from flask_cors import CORS
import io
import base64
import time
app = Flask(__name__)
CORS(app, origins='*')  # Allow CORS from all origins


@app.route('/upload', methods=['POST'])
def upload():
    folder_name = request.form['folderName']
    if not folder_name: folder_name = "error"
    if not os.path.exists("dataset/" + folder_name):
        os.makedirs("dataset/" + folder_name)
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
        img.save("dataset/" + folder_name + "/" + folder_name + "-" + str(time.time()) + "." + ext)
        return "Image saved successfully!",200
    return "Image upload failed.",500

import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import torch.optim as optim
import numpy as np
import cv2
import base64

import json
import pandas as pd
import spacy
from spacy.util import minibatch
from spacy.training.example import Example
from skimage.metrics import structural_similarity

def similarity_score(im1, im2):
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    
    score, diff = structural_similarity(im1, im2, full=True)
    return score


@app.route('/train',methods=['GET'])
def train():


    folder_name = 'dataset'

    current_directory = os.getcwd()

    folder_path = os.path.join(current_directory, folder_name)

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        subfolders = [name for name in os.listdir(
            folder_path) if os.path.isdir(os.path.join(folder_path, name))]

        num_subfolders = len(subfolders)

        print(f'Total number of subfolders in "{folder_name}": {num_subfolders}')
    else:
        print(
            f'The folder "{folder_name}" does not exist in the current directory.')


    num_classes = num_subfolders



    # directory = os.fsencode('dataset')
    dir = 'dataset'

    data = {}


    for folder in os.listdir(dir):
        print(folder)
        for file in os.listdir(dir+'/'+folder):
            filename = os.fsdecode(file)
            if (filename.endswith('jpeg')):
                if (folder in data.keys()):
                    data[folder].append(f"{dir}/{folder}/{filename}")
                else:
                    data[folder] = [f"{dir}/{folder}/{filename}"]
    bm_data = {}

    for key in data:
        # img=cv2.imread(data[key][0])
        for image in data[key]:
            this_image = cv2.imread(image, 1)

            if (key in bm_data.keys()):
                bm_data[key].append(this_image)
            else:
                bm_data[key] = [this_image]

    for key in bm_data:
        avg_image = bm_data[key][0]
        for i in range(len(bm_data[key])):
            if i == 0:
                pass
            else:
                alpha = 1.0/(i + 1)
                beta = 1.0 - alpha
                avg_image = cv2.addWeighted(
                    bm_data[key][i], alpha, avg_image, beta, 0.0)
        bm_data[key] = avg_image


    return Response(status=200)


@app.route('/frame',methods=['POST'])
def frame_upload():

    folder_name = 'dataset'

    current_directory = os.getcwd()

    folder_path = os.path.join(current_directory, folder_name)

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        subfolders = [name for name in os.listdir(
            folder_path) if os.path.isdir(os.path.join(folder_path, name))]

        num_subfolders = len(subfolders)

    else:
        return "No subfolder"


    num_classes = num_subfolders



    # directory = os.fsencode('dataset')
    dir = 'dataset'

    data = {}


    for folder in os.listdir(dir):
        # print(folder)
        for file in os.listdir(dir+'/'+folder):
            filename = os.fsdecode(file)
            if (filename.endswith('jpeg')):
                if (folder in data.keys()):
                    data[folder].append(f"{dir}/{folder}/{filename}")
                else:
                    data[folder] = [f"{dir}/{folder}/{filename}"]
    bm_data = {}

    for key in data:
        # img=cv2.imread(data[key][0])
        for image in data[key]:
            this_image = cv2.imread(image, 1)

            if (key in bm_data.keys()):
                bm_data[key].append(this_image)
            else:
                bm_data[key] = [this_image]

    for key in bm_data:
        avg_image = bm_data[key][0]
        for i in range(len(bm_data[key])):
            if i == 0:
                pass
            else:
                alpha = 1.0/(i + 1)
                beta = 1.0 - alpha
                avg_image = cv2.addWeighted(
                    bm_data[key][i], alpha, avg_image, beta, 0.0)
        bm_data[key] = avg_image

    fileContent = request.form['image']
    header , encoded = fileContent.split(',', 1)
    decoded_data = base64.b64decode(encoded)
    # Convert to NumPy array
    # test_im = np.frombuffer(decoded_data, np.uint8)

    # Read the image using OpenCV
    # img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    fileName = "train.jpeg"
    img = Image.open(io.BytesIO(decoded_data))
    img.save(fileName)
    # cv2.imwrite(fileName,img)
    test_im=cv2.imread(fileName)

    dim=list(bm_data[list(bm_data.keys())[0]].shape)


    # test_im=test_im.reshape((225,300,3))
    test_im=cv2.resize(test_im,dsize=(dim[1], dim[0]), interpolation=cv2.INTER_CUBIC)
    # test_im2=cv2.imread("b2.jpg")
    # print()

    # im1 = cv2.cvtColor(test_im, cv2.COLOR_BGR2GRAY)
    # im1 = cv2.cvtColor(test_im2, cv2.COLOR_BGR2GRAY)

    # first = cv2.imread('jot.jpg')
    # second = cv2.imread('b2.jpg')

    # print(type(first))
    # Convert images to grayscale
    # first_gray = cv2.cvtColor(test_im, cv2.COLOR_BGR2GRAY)
    # second_gray = cv2.cvtColor(test_im2, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between two images
    # score, diff = structural_similarity(first_gray, second_gray, full=True)
    # print("Similarity Score: {:.3f}%".format(score * 100))



    probs=[]

    # print(similarity_score(test_im,test_im2))
    for i in bm_data:
        print(type(bm_data[i]))
        probs.append(similarity_score(bm_data[i],test_im))
    s=0
    for i in probs:
        s+=i
    for i in probs:
        i/=s
    # print(probs)
    


    classes=list(bm_data.keys())
    out={}
    for i in range(len(classes)):
        out[classes[i]]=probs[i]
    
    out=dict(sorted(out.items(), key=lambda x:x[1]))
    rem=0
    c=0
    # print(out)
    okeys=list(out.keys())
    ovals=list(out.values())
    n=len(okeys)
    for i in range(n//2):
        ovals[i]/=2
        ovals[n-i-1]+=ovals[i]
        ovals[1]=1-ovals[0]

    # out = dict(zip(okeys,ovals))
    # print(okeys,ovals)
    out={}
    for i in range(n):
        out[okeys[i]]=ovals[i]




    print(out)

    json_output = json.dumps(out,indent=4)

    return Response(response=json_output,status=200,mimetype='application/json')

@app.route('/text-train',methods=['POST'])
def textModel():
    json_Data = request.form['data']
    
    json_data = json.loads(json_Data)
    print(json_data)
    data = []
    for label, texts in json_data.items():
        data.extend([{'text': text, 'label': label} for text in texts])
    input = pd.DataFrame(data)

    unique_labels = input['label'].unique()

    unique_labels_list = unique_labels.tolist()


    nlp = spacy.blank("en")

    textcat = nlp.add_pipe("textcat")
    for label in unique_labels_list:
        textcat.add_label(label)

    train_texts = input['text'].values
    train_labels = [{'cats': {unique_label: label == unique_label for unique_label in unique_labels_list}} 
                    for label in input['label']]
    train_data = list(zip(train_texts, train_labels))


    spacy.util.fix_random_seed(1)
    optimizer = nlp.begin_training()

    batches = minibatch(train_data, size=8)
    for batch in batches:
        for text, labels in batch:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, labels)
            nlp.update([example], sgd=optimizer)

    nlp.to_disk('spacy_model')

    return Response(status=200,response="Model Trained",mimetype='text/plain')

@app.route('/text-predict',methods=['POST'])
def textPredict():
    text = request.form['data']
    nlp = spacy.load('spacy_model')

    docs = [nlp(text)]
    textcat = nlp.get_pipe('textcat')
    score = textcat.predict(docs)

    def adjust_probabilities(probabilities, increase_by, decrease_by):
        max_prob = np.max(probabilities)
        min_prob = np.min(probabilities)
        
        max_index = np.argmax(probabilities)
        probabilities[max_index] += increase_by
        
        remaining_prob = 1 - probabilities[max_index]
        for i in range(len(probabilities)):
            if i != max_index:
                probabilities[i] -= decrease_by * (1 - increase_by)
        
        probabilities = probabilities / np.sum(probabilities)
        
        return probabilities



    increase_by = 0.15 
    num_labels = len(score[0])
    decrease_by = increase_by / (num_labels - 1)  

    for i in range(len(score)):
        score[i] = adjust_probabilities(score[i], increase_by, decrease_by)

    for i in range(len(score)):
        assert np.isclose(np.sum(score[i]), 1.0), f"Sum is not 1 for row {i}"


    predicted_labels = score.argmax(axis=1)

    predictions = []
    label_probs = {label: float(prob) for label, prob in zip(textcat.labels, score[0])}
    predictions.append(label_probs)

    results = {'text': text, 'label_probs': predictions[0]}
    json_output = json.dumps(results, indent=4)

    return Response(response=json_output,status=200,mimetype='application/json')

if __name__ == '__main__':
    app.run(host='localhost', port=8080, debug=True)
