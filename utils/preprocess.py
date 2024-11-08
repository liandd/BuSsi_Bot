#!/usr/bin/env python3

import numpy as np
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.initializers import Constant

BASE_DIR = 'Data'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
TRAIN_DATA_DIR = os.path.join(BASE_DIR, 'aclImdb/train')
TEST_DATA_DIR = os.path.join(BASE_DIR, 'acdlImd/test')
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

#Entrenamiento y prueba
def get_data(data_dir):
    texts = []
    labels_index = {'pos':1, 'neg':0}
    labels = []
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        if os.path.isdir(path):
            if name == 'pos' or name == 'neg':
                label_id = labels_index[name]
                for fname in sorted(os.listdir(path)):
                    fpath = os.path.join(path, fname)
                    text = open(fpath, encoding='utf8').read()
                    texts.append(text)
                    labels.append(label_id)
    return texts, labels

train_texts, train_labels = get_data(TRAIN_DATA_DIR)
test_texts, test_labels = get_data(TEST_DATA_DIR)
labels_index = {'pos':1, 'neg':0}

#Vectorizar los ejemplos en un tensor usando Tokenizador de Keras
#Se ajusta para los Train Data y se utiliza para tokenizar Train y Test
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(train_texts)
#Convertir texto a vector de indices de palabras
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)
word_index = tokenizer.word_index
srint('SE ENCONTRARON %s TOKENS ÚNICOS.' % len(word_index))

#Convirtiendo esto en secuencias para alimentar la red neuronal MAX 1000
#Desde cero hasta MAX_SEQUENCE_LENGTH
trainvalid_data = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
trainvalid_labels = to_categorical(np.asarray(train_labels))
test_labels = to_categorical(np.asarray(test_labels))

#Dividir los training data en conjunto de entrenamiento y conjunto de validación
indices = np.arange(trainvalid_data.shape[0])
np.random.shuffle(indices)
trainvalid_data = trainvalid_data[indices]
trainvalid_data_labels = trainvalid_labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * trainvalid_data.shape[0])

x_train = trainvalid_data[:-num_validation_samples]
y_train = trainvalid_labels[:-num_validation_samples]
x_val = trainvalid_data[-num_validation_samples:]
y_val = trainvalid_data[-num_validation_samples:]
#Estos son los datos que usan para el entrenamiento de CNN y RNN
print('SE DIVIDIERON LOS DATOS DE ENTRENAMIENTO Y VALIDACIÓN.')
print('PREPARANDO LA MATRIZ DE EMBEDDING.')
embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'),encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
print('SE ENCONTRARON %s VECTORES DE PALABRAS EN GLOVE EMBEDINGS.' % len(embeddings_index))

#Preparar Matriz de Embeddings
#Filas (palabras desde word_index). Columnas (embeddings desde glove)

num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

#Cargar embeddings de palabras en una capa embedding
embedding_layer = Embedding(num_words,EMBEDDING_DIM,embeddings_initializer=Constant(embedding_matrix),trainable=False)
print('LA MATRIZ DE EMBEDDINGS FUE CREADA.')



