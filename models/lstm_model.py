#!/usr/bin/env python3

import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import preprocess as pr

#Validar si ya está el archivo de entrenamiento
#Global
#BASE_DIR ='C:\\PROYECTO\\models'
BASE_DIR = 'models/'
LSTM_DIR = os.path.join(BASE_DIR, 'BuSsi_lstm_model.sav')

if os.path.exists(LSTM_DIR):
    print('MODELO ENCONTRADO CON EXITO!')
    sys.exit(0)

print('DEFINIR Y ENTRENAR LSTM. ENTRENAR EMBEDDING')

#Modelo LSTM
rnnmodel = pr.Sequential()
#rnnmodel.add(Embedding(pr.MAX_NUM_WORDS, 128))
rnnmodel.add(pr.embedding_layer)
rnnmodel.add(pr.LSTM(128, dropout=0.2, recurrent_dropout=0.2))
rnnmodel.add(pr.Dense(11,activation='sigmoid'))
rnnmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print('ENTRENANDO RNN...')
y_train = pr.to_categorical(pr.y_train, num_classes=11)
y_val = pr.to_categorical(pr.y_val, num_classes=11)
rnnmodel.fit(pr.x_train, y_train, batch_size=32, epochs=30,validation_data=(pr.x_val, y_val))
print('ENTRENAMIENTO COMPLETO.')

filename = "BuSsi_lstm_model.sav"
file_path = os.path.join(BASE_DIR, filename)
with open(file_path, 'wb') as f:
    pr.pickle.dump(rnnmodel, f)
print('MODELO EXPORTADO CON EXITO.')
