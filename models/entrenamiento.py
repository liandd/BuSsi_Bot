import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import json
import re
import numpy as np
import torch
import torch.nn as nn
from models.modelo import red_neuronal
from torch.utils.data import Dataset, DataLoader
from models.preprocesamiento import tokenize, stem, bag_of_words_sequence

# Cargar datos
with open('C:\\Users\\joshu\\PycharmProjects\\LucyBot\\data\\datos.json', 'r') as f:
    intentos = json.load(f)

todas_las_palabras = []
etiquetas = []
xy = []

# Procesar datos
for intento in intentos['intentos']:
    etiqueta = intento['tag']
    etiquetas.append(etiqueta)
    for pattern in intento['patterns']:
        p = tokenize(pattern)
        todas_las_palabras.extend(p)
        xy.append((p, etiqueta))

# Limpiar palabras
caracteres = ['?', ',', '.', '!']
todas_las_palabras = [stem(p) for p in todas_las_palabras if p not in caracteres]
todas_las_palabras = sorted(set(todas_las_palabras))
etiquetas = sorted(set(etiquetas))

max_len = 20  # Longitud máxima de secuencia

X_entrenamiento = []
Y_entrenamiento = []

for (sentencia_pattern, etiqueta) in xy:
    bolsa = bag_of_words_sequence(sentencia_pattern, todas_las_palabras, max_len)
    X_entrenamiento.append(bolsa)
    label = etiquetas.index(etiqueta)
    Y_entrenamiento.append(label)

X_entrenamiento = np.array(X_entrenamiento)
Y_entrenamiento = np.array(Y_entrenamiento)


# Crear dataset
class ChatDataset(Dataset):
    def __init__(self):
        self.n_ejemplos = len(X_entrenamiento)
        self.x_data = torch.tensor(X_entrenamiento, dtype=torch.float32)
        self.y_data = torch.tensor(Y_entrenamiento, dtype=torch.long)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_ejemplos


# Hiperparámetros
batch_size = 8
tamano_oculto = 32
tamano_salida = len(etiquetas)
tamano_entrada = len(todas_las_palabras)
ratio_aprendizaje = 0.0001
numero_epocas = 2000

# Preparar datos
dataset = ChatDataset()
cargador_de_entrenamiento = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Configurar modelo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modelo = red_neuronal(tamano_entrada, tamano_oculto, tamano_salida).to(device)

# Pérdida y optimización
criterio = nn.CrossEntropyLoss()
optimizador = torch.optim.Adam(modelo.parameters(), lr=ratio_aprendizaje)

# Entrenamiento
for epoca in range(numero_epocas):
    for (palabras, labels) in cargador_de_entrenamiento:
        palabras = palabras.to(device)
        labels = labels.to(device)

        # Reenviar
        salidas = modelo(palabras)
        perdida = criterio(salidas, labels)

        # Devolverse y optimizar
        optimizador.zero_grad()
        perdida.backward()
        optimizador.step()

    if (epoca + 1) % 100 == 0:
        print(f'Época {epoca + 1}/{numero_epocas}, Pérdida={perdida.item():.4f}')

print(f'Pérdida final: {perdida.item():.4f}')

data = {
    "modelo_status": modelo.state_dict(),
    "tamano_entrada": tamano_entrada,
    "tamano_salida": tamano_salida,
    "tamano_oculto": tamano_oculto,
    "todas_las_palabras": todas_las_palabras,
    "etiquetas": etiquetas

}

FILE = "../data/data.pth"
torch.save(data, FILE)
print(f'Entrenamiento completado. archivo guardado como {FILE}')

