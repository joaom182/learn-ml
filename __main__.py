import nltk
import numpy as np
import gensim.downloader as api
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define the input data and corresponding labels
# inputs = [
#     'I want to pay my bill',
#     'Can I pay my bill online?',
#     'How do I cancel my subscription?',
#     'I no longer want to subscribe to this service',
#     'Is there a way to cancel my subscription?',
#     'How do I pay my bill?'
# ]

# labels = [
#     'pay a bill',
#     'pay a bill',
#     'cancel a subscription',
#     'cancel a subscription',
#     'cancel a subscription',
#     'pay a bill'
# ]

inputs = [
    'bill',
    'pay bill',
    'subscription',
    'cancel subscription'
]

labels = [
    'pay_bill',
    'pay_bill',
    'cancel_subscription',
    'cancel_subscription'
]


# Preprocess the input data


def preprocess_input_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha()
              and token not in stopwords.words()]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(tokens)


preprocessed_inputs = [preprocess_input_text(
    input_text) for input_text in inputs]

# Load a pre-trained word embedding model (GloVe in this case)
embedding_model = api.load("glove-wiki-gigaword-50")

# Convert input text to vectors using word embeddings


def text_to_vector(text):
    words = text.split()
    vector = np.zeros(50)
    for word in words:
        if word in embedding_model:
            vector += np.array(embedding_model[word])
    return vector / len(words)


vectors = [text_to_vector(input_text) for input_text in preprocessed_inputs]

# Convert labels to one-hot encoding
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
one_hot_labels = np.eye(len(set(labels)))[encoded_labels]

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    vectors, one_hot_labels, test_size=0.2, random_state=42)

# Define a simple neural network model
model = Sequential()
model.add(Dense(64, input_shape=(50,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(set(labels)), activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(np.array(X_train), np.array(y_train), epochs=200, batch_size=8)

# Evaluate the model on test data
loss, accuracy = model.evaluate(np.array(X_test), np.array(y_test))
print('Test accuracy:', accuracy)

# Predict the intent of new user input text


def predict_intent(text):
    preprocessed_text = preprocess_input_text(text)
    vector = text_to_vector(preprocessed_text)
    predicted_label = model.predict(np.array([vector]))
    predicted_intent = label_encoder.inverse_transform(
        [np.argmax(predicted_label)])

    if (predicted_intent == None):
        return 'Unknow'

    return predicted_intent[0]


# Test the model
test_input_1 = 'How can I pay my bill?' # Output: 'pay a bill'
test_input_2 = 'Can I cancel my subscription?' # Output: 'cancel a subscription'
test_input_3 = 'How can I pay my subscription fee?' # Output: 'cancel a subscription'
test_input_4 = 'I wanna pay my bill' # Output: 'pay a bill'

print(predict_intent(test_input_1))
print(predict_intent(test_input_2))
print(predict_intent(test_input_3))
print(predict_intent(test_input_4))
