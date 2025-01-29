#!/usr/bin/env python3

import nltk
import random
import string
import warnings
import sys
import os

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from utils import preprocess as pr

warnings.filterwarnings("ignore")

#Global
#BASE_DIR = 'C:\\PROYECTO\\models'
BASE_DIR = 'models/'
LSTM_DIR = os.path.join(BASE_DIR, 'BuSsi_lstm_model.sav')

#Clase chatbot
class Chatbot:
    def __init__(self):
        self.saludos_inputs = ["hola", "buenas", "saludos", "qué tal", "buenos días", "hola amigo", "buen día", "hey", "holaa", "cómo estás", "lucy cómo estás", "como estas", "buenos dias","Hola","Buenas","Saludos","Qué tal","Buenos días","Hola amigo","Buen día", "Hey", "Cómo estás", "Lucy cómo estás"]
        self.saludos_outputs = ["Hola, soy Lucy", "Buenas Tardes, ¿en qué te puedo ayudar?", "Hola, cuentame de ti", "Hey, aquí Lucy", "¿Cómo puedo ayudarte?", "Saludos amigo", "Hola, amigo!", "Hola, ¿Cómo te puedo ayudar?"]
        
        self.load_model()

        #Guardar respuestas
        self.sent_tokens = []
        self.vectorizer = TfidfVectorizer(tokenizer=self.lem_normalize, stop_words=stopwords.words('spanish'))

    def load_model(self):
        #Validar si ya está el archivo de entrenamiento
        if os.path.exists(LSTM_DIR):
            print('MODELO ENCONTRADO CON EXITO!')

            with open(LSTM_DIR, 'rb') as file:
                self.lstm_model = pr.pickle.load(file)
            print('MODELO CARGADO EXITOSAMENTE!')
        else:
            print('NO SE ENCONTRO EL MODELO!')
            sys.exit(0)

    def preprocess_input(self, text):
        sequences = pr.tokenizer.texts_to_sequences([text])
        padded_seq = pr.pad_sequences(sequences, maxlen=pr.MAX_SEQUENCE_LENGTH, dtype='int32')
        return padded_seq

    def lem_normalize(self, text):
        lemmer = nltk.stem.WordNetLemmatizer()
        remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

        def lem_tokens(tokens):
            return [lemmer.lemmatize(token) for token in tokens]

        return lem_tokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

    def generate_res(self, text, tokenizer, model):
        response = text
        for _ in range(50):
            token_list = tokenizer.texts_to_sequences([response])[0]
            token_list = pr.pad_sequences([token_list], maxlen=pr.MAX_SEQUENCE_LENGTH, padding='pre')
            predict = model.predict(token_list, verbose=0)
            next_word_index = pr.np.argmax(predict)

            prob = predict[0][next_word_index]
            if prob < 0.2:
                return 'Lo siento, no te entiendo'

            next_word = tokenizer.index_word.get(next_word_index)

            if next_word is None:
                break;
            response += " "+next_word
        return response

    def get_response(self, user_response):
        if self.is_greeting(user_response):
            return random.choice(self.saludos_outputs)
        else:
            gen_response = self.generate_res(user_response,pr.tokenizer, self.lstm_model)

            if "Lo siento, no te entiendo" in gen_response:
                return "Lo siento, no es mi dominio no te entiendo. Póngase en contacto con equipo1@ucp.edu.co"

            return gen_response

    def is_greeting(self, sentence):
        for word in sentence.split():
            if word in self.saludos_inputs:
                return True
        return False

    def chat(self, text):
        # Modo conversación
        yield "Soy Lucy, Un ChatBot bajo el dominio Negocios. Contestaré a tus preguntas acerca de Negocios. Si quiere salir escribe salir."
        flag:bool = True
        user_response = text.lower()
        while flag:
            if user_response == 'salir':
                yield "Fue un gusto hablar contigo, ¡Cuídate!"
                flag = False
            else:
                yield self.get_response(user_response)
                user_response = (yield)


