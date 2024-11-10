#!/usr/bin/env python3

import ntlk
import random
import string
import warnings

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
warnings.filterwarnings("ignore")

#Clase chatbot
class Chatbot:
    def __init__(self):
        pass

    def load_model():
        pass

    def preprocess():
        lemmer = nltk.stem.WordNetLemmatizer()

        def lemTokens(tokens):
            return [lemmer.lemmatize(token) for token in tokens]

        remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

        def lemNormalize(text):
            return lemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

    def response(user_response):
        bot_response = ''
        sent_tokens.append(user_response)
        TfidfVec = TfidfVectorizer(tokenizer=lemNormalize, stop_words=stopwords.words('spanish'))
        tfidf = TfidfVec.fit_transform(sent_tokens)
        vals = cosine_similarity(tfidf[-1], tfidf)

        idx = vals.argsort()[0][-2]
        flat = vals.flatten()
        flat.sort()
        req_tfidf = flat[-2]

        threeshold = 0.3
        if req_tfidf < threeshold:
            bot_response = "Lo siento, no te BuSsi_entiendo. Póngase en contacto con Equipo1@ucp.edu.co"
        else:
            bot_response = sent_tokens[idx]
        sent_tokens.pop()
        return bot_response
    
    self.saludos_inputs = ["hola", "buenas", "saludos", "qué tal", "buenos días", "hola amigo", "buen día", "hey", "bussi hola", "cómo estás", "bussi cómo estás"]
    self.saludos_outputs = ["Hola, soy BuSsi", "Buenas Tardes, ¿en qué te puedo ayuda?", "Hola, cuentame de ti", "Hey, aquí BuSsi", "¿Cómo puedo ayudarte?", "Saludos amigo", "Hola, amigo!", "Hola, ¿Cómo te puedo ayudar?"]

    def greetings(sentence):
        for word in sentence.split():
            if word.lower() in self.saludos_inputs:
                return random.choice(self.saludos_output)

    def response_sent():
        bot_text = "Soy BuSsi, Un ChatBot bajo el dominio Negocios. Contestaré a tus preguntas acerca de Negocios. Si quiere salir escribe 'salir'"
        flag:bool = True
        while flag:
            user_response = input().lower()
            if user_response != 'salir':
                if user_response in ['gracias', 'muchas gracias']:
                    print('Con gusto, vuelve pronto')
                elif greetings(user_response):
                    print('BuSsi:', greetings(user_response))
                else:
                    print('BuSsi:', response(user_response))
            else:
                flag = False
                print('Fué un gusto hablar contigo, ¡Cuídate!')

