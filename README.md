# headout-b

This is the backend server for our Machine Learning as a Service (MLaaS) platform.

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Applicaton Start](#1-start-the-application)
  - [API Endpoints](#2-api-endpoints)

## Getting Started

### Prerequisites

- Python 
- Framework Used: Flask
- pip

### Installation

#### 1. Clone this repository:
   ```bash
   git clone https://github.com/akshayw1/headout-b.git
   cd headout-b
   ```

#### 2. Install the dependencies:
   ```cmd
   pip install -r requirements.txt
   ```

## Usage

### 1. Start the Application:
   ```cmd
   python app.py
   ```

The server will start running at `http://localhost:5000` by default. PORT number can be changed by specifying the port number at the end of `app.py` file.

### 2. API Endpoints

#### Image Classification

  - `POST /upload:` Uploads dataset using FormData containing class name and with an image of it, multiple `POST` request need to be sent to train the model. A simple text response will be sent with the message and status code. 
    ```json
    {
      "folderName": CLASS_NAME,
      "image": IMAGE_URI_STRING
    }
    ```
    **Response:**
    ```json
    {
      "status": 200,
      "message": "Image saved successfully!"
    }
    ``` 

  - `GET /train:` Endpoint to train the model based on given dataset. No body is expected here & a simple status code of `200` is expected on successful training of the model. If no upload data is sent a text message of *"No subfolder"* with status code `404` will be sent. 
    **Response:**
    ```json
    {
        "status": 200
    }
    ``` 

  - `POST /frame: ` This will provide the result of live preview of the trained model, when sent with an image frame in FormData. A json response with a status code `200` is expected.
    ```json
    {
      "image": IMAGE_URI_STRING
    }
    ```

    **Response:**
    ```json
    {
      "status": 200,
      "data": {
        "class1": PROBABILITY_SCORE_OF_CLASS1,
        "class2": PROBABILITY_SCORE_OF_CLASS2,
        "class3": PROBABILITY_SCORE_OF_CLASS3
        .
        .
        .
      }
    }
    ``` 

#### Text Classfication

  - `POST /text-train:` Uploads dataset using FormData containing a json object for different classes and their examples. Now, model will be trained based on given json data and status code of `200` is expected on successful training of the model
    ```json
    {
      "data": {
        "class1": [
          SAMPLE_STRING_OF_CLASS1,
          SAMPLE_STRING_OF_CLASS1,
          SAMPLE_STRING_OF_CLASS1
        ],
        "class2": [
          SAMPLE_STRING_OF_CLASS2,
          SAMPLE_STRING_OF_CLASS2,
          SAMPLE_STRING_OF_CLASS2
        ],
        "class3": [
          SAMPLE_STRING_OF_CLASS3,
          SAMPLE_STRING_OF_CLASS3,
          SAMPLE_STRING_OF_CLASS3
        ]
        .
        .
        .
      }
    }
    ```
    **Response:**
    ```json
    {
      "status": 200,
      "message": "Model Trained"
    }
    ``` 

  - `POST /text-predict: ` This will provide the result of the trained model, when sent with a text key containing a string to be predicted in FormData. A json response with a status code `200` is expected containing probabilities based on classes of model trained.
    ```json
    {
      "text": PREDICT_STRING
    }
    ```

    **Response:**
    ```json
    {
      "status": 200,
      "data": {
        "text": STRING_EVALUATED,
        "label_probs": {
          "class1": PROBABILITY_SCORE_OF_CLASS1,
          "class2": PROBABILITY_SCORE_OF_CLASS2,
          "class3": PROBABILITY_SCORE_OF_CLASS3
        .
        .
        .
        }
      }
    }
    ``` 