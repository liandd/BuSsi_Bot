import random
import json
import torch
from models.modelo import red_neuronal
from models.preprocesamiento import bag_of_words_sequence, tokenize

# Cargar modelo y datos
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('./data/datos.json', 'r') as f:
    datos = json.load(f)

FILE = "C:\\Users\\joshu\\PycharmProjects\\LucyBot\\data\\data.pth"
data = torch.load(FILE)
tamano_entrada = data["tamano_entrada"]
tamano_oculto = data["tamano_oculto"]
tamano_salida = data["tamano_salida"]
todas_las_palabras = data["todas_las_palabras"]
etiquetas = data["etiquetas"]
modelo_status = data["modelo_status"]

modelo = red_neuronal(tamano_entrada, tamano_oculto, tamano_salida).to(device)
modelo.load_state_dict(modelo_status)
modelo.eval()

def get_response(input_text):
    """Genera la respuesta del chatbot."""
    # Preprocesar entrada del usuario
    sentencia = tokenize(input_text)
    X = bag_of_words_sequence(sentencia, todas_las_palabras, max_len=20)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)

    # Predicción
    salida = modelo(X)
    _, prediccion = torch.max(salida, dim=1)
    etiqueta = etiquetas[prediccion.item()]
    probabilidad = torch.softmax(salida, dim=1)[0][prediccion.item()]

    # Generar respuesta
    if probabilidad.item() > 0.75:
        for intento in datos["intentos"]:
            if etiqueta == intento["tag"]:
                return random.choice(intento["responses"])
    else:
        return "Lo siento, no entiendo tu mensaje."
