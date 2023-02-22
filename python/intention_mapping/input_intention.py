import tensorflow as tf
import os
import numpy as np
import pandas as pd
import json
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, GlobalMaxPooling1D, Flatten
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import string
import random

intensions_training_model_file = 'intentions_training_model.json'
intentions_model_file = 'intentions.h5'

def init_model():
    with open(intensions_training_model_file) as content:
        data1 = json.load(content)

    tags = []
    inputs = []
    responses = {}

    for intent in data1['intents']:
        responses[intent['tag']]=intent['responses']
        for lines in intent['input']:
            inputs.append(lines)
            tags.append(intent['tag'])

    data = pd.DataFrame({"inputs": inputs, "tags": tags})
    data = data.sample(frac=1)

    #removing ponctuations
    data['inputs'] = data['inputs'].apply(lambda wrd:[letrs.lower() for letrs in wrd if letrs not in string.punctuation])
    data['inputs'] = data['inputs'].apply(lambda wrd: ''.join(wrd))

    tokenizer = Tokenizer(num_words=2000)
    tokenizer.fit_on_texts(data['inputs'])
    train = tokenizer.texts_to_sequences(data['inputs'])

    #apply padding
    x_train = pad_sequences(train)

    #encoding the outputs
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(data['tags'])

    input_shape = x_train.shape[1]
    vocabulary = len(tokenizer.word_index)
    output_length = label_encoder.classes_.shape[0]

    i = Input(shape=(input_shape,))
    x = Embedding(vocabulary +1, 10)(i)
    x = LSTM(10, return_sequences=True)(x)
    x = Flatten()(x)
    x = Dense(output_length, activation="softmax")(x)
    model = Model(i, x)
    model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

    if os.path.exists(intentions_model_file):
        model = tf.keras.models.load_model(intentions_model_file)
    else:
        train = model.fit(x_train, y_train, epochs=200)
        model.save(intentions_model_file)
    
    return [model, responses, label_encoder, tokenizer, input_shape]

def get_answer(prediction_input, tokenizer, responses, label_encoder, input_shape, model):
    texts_p = []

    #removing ponctuation and converting to lowercase
    prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
    prediction_input = ''.join(prediction_input)
    texts_p.append(prediction_input)
    prediction_input = tokenizer.texts_to_sequences(texts_p)
    prediction_input = np.array(prediction_input).reshape(-1)
    prediction_input = pad_sequences([prediction_input], input_shape)
    
    #getting output from model
    output = model.predict(prediction_input)
    output = output.argmax()

    #finding the right tag and predicting
    response_tag = label_encoder.inverse_transform([output])[0]
    answer = random.choice(responses[response_tag])
    return [answer, response_tag]

def main():
    [model, responses, label_encoder, tokenizer, input_shape] = init_model()

    while True:
        prediction_input = input('You: ')
        [answer, response_tag] = get_answer(prediction_input, tokenizer, responses, label_encoder, input_shape, model)
        print("Bot: ", answer)

        if response_tag == "goodbye":
            break

main()