import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras_preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
import random

# Global constants
window_size = 40  # Length of character sequences
batch_size = 32  # Batch size for learning
rnn_units = 128  # Number of hidden units in the LSTM or GRU cells cell
epochs = 100  # Number of training epochs

corpus_file = 'odyssey.txt'


def load_corpus(file_name, local=True, github_repo='/'):
    """Loads the corpus text from a given file.

    Parameters
    ----------
    file_name : str
        If local is True, the full path to the local file, otherwise the file name for the github repo.
    local : bool
        True if file is local, False (default) if file is in github repo.
    github_repo : str
        Github repo storing the file (only used if local is False).
    """

    if local:
        return open(file_name, 'r').read()

    import requests
    page = requests.get(f'https://raw.githubusercontent.com/{github_repo}/{file_name}')
    return page.text


def preprocess_file(file_name):
    """Read a text file, perform preprocessing, and return text as a string.

    Parameters
    ----------
    file_name : str
        Name of text file to load.

    Returns
    -------
    text : str
        Preprocessed text from the file.
    """

    # TODO: Read in file (using load_corpus()) and convert to lower case
    text = load_corpus(file_name).lower()

    # Optional: perform additional processing
    tokenizer = Tokenizer(char_level=True)
    text = text.translate({ord(c): None for c in '!@#$";123456{-}:'"'"})
    return text


# Load and prepare data
text = preprocess_file(corpus_file)
text = text[:10000]  # Shorten text for testing
print(text[:500])


def make_dataset(text, window_size=40):
    """Create the dataset used to train the RNN.

    Parameters
    ----------
    text : str
        String representing text to learn on.
    window_size : int
        Length of character sequence used to predict next character.

    Returns
    -------
    vocab : list(char)
        List of characters making up the vocabulary of the text.
    x_data : list(list(int))
        List of sequences of size window_size, containing indices into vocab.
        Each sequence represents a sequence of window_size characters found in
        the text. The number of sequences generated will be len(text) - window_size.
    y_data : list(int)
        List of indices corresponding to the characters that follow the
        sequences in x_data.
    """
    # TODO: Determine list of unique characters
    vocab = sorted(list(set(text)))

    x_data = []
    y_data = []

    # TODO: Generate training data
    for i in range(len(text) - window_size):
        x_tmp = []

        for char in text[i: i + window_size]:
            x_tmp.append(vocab.index(char))

        x_data.append(x_tmp)
        y_data.append(vocab.index(text[i + window_size]))

    return x_data, y_data, vocab


x_data, y_data, vocab = make_dataset(text, window_size=window_size)

# Check if everything is working
print("Vocabulary: ")
print(vocab)
print("Vocabulary length: ")
print(len(vocab))
print("First element of x_data: ")
print(x_data[0])
print("First element of y_data: ")
print(y_data[0])


def rnn_model(num_units, window_size, vocab_size, rnn_layer=layers.LSTM):
    """Creates the RNN model.

    Parameters
    ----------
    num_units : int
        Number of hidden units in the LSTM layer.
    window_size : int
        Number of characters in an input sequence.
    vocab_size : int
        Number of unique characters in the vocabulary.
    rnn_layer : Keras RNN layer (RNN, LSTM, GRU)

    Returns
    -------
    model : Keras model
        RNN model.
    """

    # TODO: Build the model
    model = Sequential()
    model.add(layers.Embedding(input_dim=vocab_size, output_dim=vocab_size, trainable=True, input_length=window_size))
    model.add(layers.LSTM(num_units, return_sequences=False))
    model.add(layers.Dense(vocab_size, activation='softmax'))
    # TODO: Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


model = rnn_model(rnn_units, window_size, len(vocab))
model.summary()

epochsTrain = 3
for i in range(1, epochs, 3):
    model.fit(x_data, y_data, batch_size=batch_size, epochs=epochsTrain)
    seed_sequence = random.choice(x_data)
    generatedText = ''.join([str(vocab[ind]) for ind in seed_sequence])

    for _ in range(500):
        predictedIndex = int(np.argmax(model.predict([seed_sequence])))

        generatedText = ''.join((generatedText, vocab[predictedIndex]))
        del seed_sequence[0]
        seed_sequence.append(predictedIndex)

    print(generatedText)

print(generated_text)

modelGRU = Sequential()

modelGRU.add(layers.Embedding(input_dim=len(vocab), output_dim=3000,trainable=True, input_length=window_size))
modelGRU.add(layers.GRU(rnn_units, return_sequences=False))
modelGRU.add(layers.Dense(len(vocab), activation='softmax'))
modelGRU.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

modelGRU.summary()

epochsTrain = 3

modelGRU.fit(x_data, y_data, batch_size=batch_size, epochs=epochsTrain)
seed_sequence = random.choice(x_data)
generatedText = ''.join([str(vocab[ind]) for ind in seed_sequence])

for _ in range(500):
    predictedIndex = int(np.argmax(model.predict([seed_sequence])))
    generatedText = ''.join((generatedText, vocab[predictedIndex]))
    del seed_sequence[0]
    seed_sequence.append(predictedIndex)
print(generatedText)