#!/home/blyaat/Desktop/ian/GitHub/BuSsi_Bot/my_env/bin python3

import numpy as np
import os
import json
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import Constant
from tensorflow.keras.utils import to_categorical

import warnings
warnings.filterwarnings('ignore')

# Configuración
BASE_DIR = '/home/blyaat/Desktop/ian/GitHub/BuSsi_Bot/Data'
#BASE_DIR = 'Data' #Carpeta con los datos corpus
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B.100d.txt')
CORPUS_JSON = os.path.join(BASE_DIR, 'Corpus_negocios.json')
CORPUS_TXT = os.path.join(BASE_DIR, 'Corpus_negocios.txt')
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100

# Cargar datos desde corpus.json
def load_json_data(corpus_json):
    with open(corpus_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = [item['text'] for item in data['json_input']]
    labels = [item['label'] for item in data['json_input']]
    return texts, labels

# Cargar datos adicionales desde corpus.txt
def load_txt_data(corpus_txt):
    with open(corpus_txt, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    texts = [line.strip() for line in lines if not line.startswith('#') and line.strip() != '']
    return texts

# Cargar ambos conjuntos de datos
json_texts, json_labels = load_json_data(CORPUS_JSON)
txt_texts = load_txt_data(CORPUS_TXT)
all_texts = json_texts + txt_texts

# Vectorización usando Tokenizer
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(all_texts)
json_sequences = tokenizer.texts_to_sequences(json_texts)
word_index = tokenizer.word_index
print(f'SE ENCONTRARON {len(word_index)} TOKENS ÚNICOS.')

data = pad_sequences(json_sequences, maxlen=MAX_SEQUENCE_LENGTH, dtype='int32')
labels = np.asarray(json_labels)

x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.2)
print('SE DIVIDIERON LOS DATOS DE ENTRENAMIENTO Y VALIDACIÓN.')

# Cargar embeddings de GloVe
embeddings_index = {}
with open(os.path.join(GLOVE_DIR), encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
print(f'SE ENCONTRARON {len(embeddings_index)} VECTORES DE PALABRAS EN GLOVE EMBEDINGS.')

# Preparar la matriz de embeddings
num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Cargar embeddings en una capa embedding no entrenable
embedding_layer = Embedding(num_words, EMBEDDING_DIM, embeddings_initializer=Constant(embedding_matrix), trainable=False)
print('LA MATRIZ DE EMBEDDINGS FUE CREADA.')
