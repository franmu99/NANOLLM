import numpy as np
import torch
from tokenizers import Tokenizer
from gensim.models import Word2Vec

# Cargar el tokenizador que has entrenado
tokenizer = Tokenizer.from_file("tokenizer-wordpiece.json")

# Función para tokenizar un texto
def tokenize_text(text):
    return tokenizer.encode(text).tokens

# Función para leer y tokenizar el corpus
def read_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [tokenize_text(line.strip()) for line in f]

# Ruta al archivo de texto para entrenamiento
corpus_file = r"C:\Users\franm\OneDrive\Documentos\tokenizador.txt"

# Leer y tokenizar el corpus
tokenized_corpus = read_corpus(corpus_file)

# Entrenar el modelo Word2Vec
embedding_size = 100  # Puedes ajustar este valor según tus necesidades
model = Word2Vec(sentences=tokenized_corpus, vector_size=embedding_size, window=5, min_count=1, workers=4)

# Crear la matriz de embedding
vocab = tokenizer.get_vocab()
vocab_size = len(vocab)
embedding_matrix = np.zeros((vocab_size, embedding_size))

for token, token_id in vocab.items():
    if token in model.wv:
        embedding_matrix[token_id] = model.wv[token]
    else:
        # Para tokens que no están en el modelo Word2Vec, inicializamos con valores aleatorios
        embedding_matrix[token_id] = np.random.uniform(-0.25, 0.25, embedding_size)

# Convertir a tensor de PyTorch
embedding_tensor = torch.FloatTensor(embedding_matrix)

print(f"Matriz de embedding creada. Forma: {embedding_matrix.shape}")
print(f"Tensor de embedding creado. Forma: {embedding_tensor.shape}")

# Guardar la matriz de embedding
np.save('embedding_matrix.npy', embedding_matrix)
torch.save(embedding_tensor, 'embedding_tensor.pt')

print(f"Matriz y tensor de embedding guardados.")

# Función para obtener el embedding de un token
def get_token_embedding(token):
    token_id = vocab.get(token)
    if token_id is not None:
        return embedding_tensor[token_id]
    else:
        return None

# Función para obtener embeddings para una secuencia de tokens
def get_sequence_embeddings(tokens):
    return torch.stack([get_token_embedding(token) for token in tokens if get_token_embedding(token) is not None])

# Ejemplo de uso
test_words = ["inteligencia", "artificial", "aprendizaje", "máquina"]
for word in test_words:
    tokens = tokenizer.encode(word).tokens
    print(f"'{word}' se tokeniza como: {tokens}")
    embeddings = get_sequence_embeddings(tokens)
    if embeddings.size(0) > 0:
        print(f"Embedding para '{word}': {embeddings.mean(dim=0)[:5]}...")  # Promedio de los embeddings de los tokens
    else:
        print(f"No se encontró embedding para '{word}'")

# Función para preparar datos para self-attention
def prepare_for_self_attention(sentence, max_length):
    tokens = tokenizer.encode(sentence).ids[:max_length]
    input_ids = torch.tensor(tokens)
    attention_mask = torch.ones_like(input_ids)
    
    # Padding
    if len(tokens) < max_length:
        padding = max_length - len(tokens)
        input_ids = torch.cat([input_ids, torch.zeros(padding, dtype=torch.long)])
        attention_mask = torch.cat([attention_mask, torch.zeros(padding, dtype=torch.float)])
    
    # Obtener embeddings
    embedded_input = embedding_tensor[input_ids]
    
    return embedded_input, attention_mask

# Ejemplo de uso con frases
test_sentences = [
    "La inteligencia artificial está cambiando el mundo",
    "El aprendizaje profundo es una técnica poderosa",
    "Los modelos de lenguaje son cada vez más avanzados"
]

max_length = 20  # Ajusta según tus necesidades

for sentence in test_sentences:
    embedded_input, attention_mask = prepare_for_self_attention(sentence, max_length)
    print(f"\nFrase: '{sentence}'")
    print(f"Forma del tensor de entrada embedido: {embedded_input.shape}")
    print(f"Forma de la máscara de atención: {attention_mask.shape}")
    print(f"Primeros valores del primer token: {embedded_input[0, :5]}...")

print("\nPreparación completa para self-attention.")