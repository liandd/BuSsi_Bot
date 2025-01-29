import torch
import torch.nn as nn

class red_neuronal(nn.Module):
    def __init__(self, tamano_entrada, tamano_oculto, numero_clases):
        super(red_neuronal, self).__init__()
        self.lstm = nn.LSTM(input_size=tamano_entrada, hidden_size=tamano_oculto, num_layers=1, batch_first=True)
        self.fc = nn.Linear(tamano_oculto, numero_clases)  # Capa final para mapear a las clases

    def forward(self, x):
        # LSTM toma secuencias
        salida, (hidden, cell) = self.lstm(x)
        # Tomamos la última salida oculta
        salida = self.fc(hidden[-1])  # `hidden[-1]` representa la salida de la última capa
        return salida
