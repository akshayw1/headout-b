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

import json
import pandas as pd
import spacy
from spacy.util import minibatch
from spacy.training.example import Example



class SqueezeNet(nn.Module):
        

        def __init__(self,  num_classes=10):
            super(SqueezeNet, self).__init__()
            
            self.num_classes = num_classes
            
            self.features = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, stride=2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                    Fire(64, 16, 64, 64),
                    Fire(128, 16, 64, 64),
                    nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                    Fire(128, 32, 128, 128),
                    Fire(256, 32, 128, 128),
                    nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                    Fire(256, 48, 192, 192),
                    Fire(384, 48, 192, 192),
                    Fire(384, 64, 256, 256),
                    Fire(512, 64, 256, 256),
                )
            
            final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                final_conv,
                nn.ReLU(inplace=True),
                nn.AvgPool2d(13)
            )

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if m is final_conv:
                        init.normal(m.weight.data, mean=0.0, std=0.01)
                    else:
                        init.kaiming_uniform(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x.view(x.size(0), self.num_classes)

class Fire(nn.Module):

        def __init__(self, inplanes, squeeze_planes,
                    expand1x1_planes, expand3x3_planes):
            super(Fire, self).__init__()
            self.inplanes = inplanes
            self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
            self.squeeze_activation = nn.ReLU(inplace=True)
            self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                    kernel_size=1)
            self.expand1x1_activation = nn.ReLU(inplace=True)
            self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                    kernel_size=3, padding=1)
            self.expand3x3_activation = nn.ReLU(inplace=True)

        def forward(self, x):
            x = self.squeeze_activation(self.squeeze(x))
            return torch.cat([
                self.expand1x1_activation(self.expand1x1(x)),
                self.expand3x3_activation(self.expand3x3(x))
            ], 1)

@app.route('/train',methods=['GET'])
def train():
    folder_name = 'dataset'
    current_directory = os.getcwd()
    folder_path = os.path.join(current_directory, folder_name)

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        subfolders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
        
        num_subfolders = len(subfolders)
    else:
        return "No subfolder",404
    

    model = SqueezeNet()

    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),          
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
    ])

    dataset_root = './dataset'

    dataset = ImageFolder(root=dataset_root, transform=transform)

    num_classes = len(dataset.classes)

    batch_size = 7

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 




    os.environ["OMP_NUM_THREADS"] = "1"

    criterion = nn.CrossEntropyLoss()

    learning_rate = 0.001
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    num_epochs = 7

    for epoch in range(num_epochs):
        for batch in dataloader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), 'squeezenet_model.pth')

    return Response(status=200)


@app.route('/frame',methods=['POST'])
def frame_upload():
    fileContent = request.form['image']
    if fileContent:
        header, encoded = fileContent.split(',', 1)

        # Decode the base64 string into bytes
        image_bytes = base64.b64decode(encoded)

        # Create a pillow image from the decoded bytes
        img = Image.open(io.BytesIO(image_bytes))

        dataset_folder = "dataset/"
        class_mapping = {}

        class_names = [d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d))]

        for i, class_name in enumerate(class_names):
            class_mapping[i] = class_name

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = transform(img)

        model = SqueezeNet()
        model.load_state_dict(torch.load('squeezenet_model.pth'))
        model.eval()

        with torch.no_grad():
            output = model(image.unsqueeze(0))
            probabilities = torch.softmax(output, dim=1)
            
        class_probabilities = probabilities[0].tolist()
        class_probabilities_dict = {class_name: probability for class_name, probability in zip(class_names, class_probabilities)}

        json_output = json.dumps(class_probabilities_dict, indent=4)

        print(json_output)
        return Response(response=json_output,status=200,mimetype='application/json')

@app.route('/text-train',methods=['POST'])
def textModel():
    json_data = request.form['data']
    
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
    text = request.form['text']
    nlp = spacy.load('spacy_model')

    docs = [nlp(text)]
    textcat = nlp.get_pipe('textcat')
    scores = textcat.predict(docs)
    predicted_labels = scores.argmax(axis=1)

    predictions = []
    label_probs = {label: float(prob) for label, prob in zip(textcat.labels, score[0])}
    predictions.append(label_probs)

    results = {'text': text, 'label_probs': predictions[0]}
    json_output = json.dumps(results, indent=4)

    return Response(response=json_output,status=200,mimetype='application/json')

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
