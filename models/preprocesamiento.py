from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import numpy as np

stemmer = PorterStemmer()


def tokenize(sentencia):
    return word_tokenize(sentencia)


def stem(palabra):
    return stemmer.stem(palabra.lower())


def bag_of_words_sequence(sentencia_tokenizada, todas_las_palabras, max_len):
    sentencia_tokenizada = [stem(p) for p in sentencia_tokenizada]
    secuencia = []
    for palabra in sentencia_tokenizada:
        vector = np.zeros(len(todas_las_palabras), dtype=np.float32)
        if palabra in todas_las_palabras:
            vector[todas_las_palabras.index(palabra)] = 1.0
        secuencia.append(vector)

    # Rellenar con ceros hasta max_len
    while len(secuencia) < max_len:
        secuencia.append(np.zeros(len(todas_las_palabras), dtype=np.float32))
    return np.array(secuencia[:max_len])


"""
sentencia = ["hola", "como", "estas"]
palabras = ["hola", "ey", "estas", "bien", "el", "tu"]
bog = bag_of_words(sentencia,palabras)

print(sentencia)
print(palabras)
print(bog)

"""

"""
a = "Cual es el concepto de un plan de negocios"

print(a)

a = tokenize(a)

print(a)

stemmed_words = [stem(palabra) for palabra in a]
print(stemmed_words)

"""