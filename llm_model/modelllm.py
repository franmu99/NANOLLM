import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tokenizers import Tokenizer

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_query = nn.Linear(d_model, d_model)
        self.W_key = nn.Linear(d_model, d_model)
        self.W_value = nn.Linear(d_model, d_model)
        self.W_out = nn.Linear(d_model, d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_query.weight)
        nn.init.xavier_uniform_(self.W_key.weight)
        nn.init.xavier_uniform_(self.W_value.weight)
        nn.init.xavier_uniform_(self.W_out.weight)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        print(f"Input shape: {x.shape}")

        Q = self.W_query(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_key(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_value(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask.unsqueeze(1), float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
 
        
        output = self.W_out(attention_output)
      
        return output, attention_weights

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        attn_output, attention_weights = self.self_attn(x, mask)
        x = self.norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x, attention_weights

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, embedding_matrix):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self._init_weights(embedding_matrix)

    def _init_weights(self, embedding_matrix):
        nn.init.xavier_uniform_(self.fc_out.weight)
        
        # Inicializa la capa de embedding con los pesos preentrenados
        self.embedding.weight.data.copy_(torch.tensor(embedding_matrix, dtype=torch.float32))
        print("Embedding layer initialized with pretrained weights.")

    def forward(self, x, mask):
        x = self.embedding(x)
        attention_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attention_weights.append(attn_weights)
        output = self.fc_out(x)
        return output, attention_weights

def create_causal_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask

# Cargar la matriz de embeddings y el tokenizador
embedding_matrix = np.load('embedding_matrix.npy')
tokenizer = Tokenizer.from_file("tokenizer-wordpiece.json")

# Configuración del modelo
num_tokens, embedding_dim = embedding_matrix.shape
d_model = embedding_dim  # Usar la dimensión de los embeddings preentrenados (100)
num_heads = 4  # Ajustado para ser divisible por d_model
d_ff = 4 * d_model
num_layers = 6
vocab_size = num_tokens  # Asegúrate de que esto sea correcto

# Crear el modelo
model = TransformerDecoder(num_layers, d_model, num_heads, d_ff, vocab_size, embedding_matrix)

# Ejemplo de uso
sequence_length = 10
batch_size = 2
input_ids = torch.randint(0, num_tokens, (batch_size, sequence_length))
mask = create_causal_mask(sequence_length).unsqueeze(0).repeat(batch_size, 1, 1)

output, attention_weights = model(input_ids, mask)

print(f"\nForma de la salida final: {output.shape}")

# Mostrar una parte de la salida final
print("\nPrimeros 5 tokens de la salida final para el primer elemento del batch:")
print(output[0, :5, :5].tolist())

# Analizar los pesos de atención para cada cabeza en la última capa
print("\nPesos de atención promedio para cada cabeza en la última capa:")
last_layer_attention = attention_weights[-1]
for i in range(num_heads):
    head_attention = last_layer_attention[0, i, :, :]  # Tomamos solo el primer elemento del batch
    avg_attention = head_attention.mean().item()
    print(f"Cabeza {i+1}: {avg_attention:.4f}")

print("\nAnálisis detallado de los pesos de atención en la última capa:")
last_layer_attention = attention_weights[-1]
for i in range(num_heads):
    head_attention = last_layer_attention[0, i, :, :]  # Tomamos solo el primer elemento del batch
    print(f"\nCabeza {i+1}:")
    print(f"  Promedio: {head_attention.mean().item():.4f}")
    print(f"  Máximo: {head_attention.max().item():.4f}")
    print(f"  Mínimo: {head_attention.min().item():.4f}")
    print(f"  Desviación estándar: {head_attention.std().item():.4f}")

print(f"Tamaño del vocabulario (vocab_size): {vocab_size}")
print(f"Forma de la capa de salida (fc_out): {model.fc_out.weight.shape}")

# Prueba de predicción
with torch.no_grad():
    input_ids = torch.randint(0, vocab_size, (1, 5))  # Secuencia de entrada aleatoria
    mask = create_causal_mask(5).unsqueeze(0)
    output, _ = model(input_ids, mask)
    
    print("\nPrueba de predicción:")
    print(f"Entrada: {input_ids[0]}")
    print(f"Salida (logits para los primeros 5 tokens):")
    print(output[0, -1, :5])  # Mostramos los logits para los primeros 5 tokens de la última posición
    print(f"Token predicho: {output[0, -1].argmax().item()}")