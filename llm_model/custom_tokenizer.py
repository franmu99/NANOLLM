from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
import random
import os
from collections import Counter

def get_training_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield line.strip()

# Ruta al archivo de texto para entrenamiento
corpus_file = r"C:\Users\franm\OneDrive\Documentos\tokenizador.txt"

# Verificar si el archivo existe
if not os.path.exists(corpus_file):
    raise FileNotFoundError(f"El archivo {corpus_file} no existe.")

# Inicializar el tokenizador WordPiece
tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
trainer = WordPieceTrainer(
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    min_frequency=4,  # Frecuencia mínima para considerar una subpalabra
    vocab_size=800,  # Tamaño máximo del vocabulario
    show_progress=True
)

# Pre-tokenizar por espacios en blanco
tokenizer.pre_tokenizer = Whitespace()

# Entrenar el tokenizador
print(f"Iniciando el entrenamiento con el archivo: {corpus_file}")
tokenizer.train_from_iterator(get_training_corpus(corpus_file), trainer=trainer)

# Guardar el tokenizador entrenado
tokenizer.save("tokenizer-wordpiece.json")

print("Tokenizador entrenado y guardado como 'tokenizer-wordpiece.json'")

# Cargar el tokenizador (para uso futuro)
loaded_tokenizer = Tokenizer.from_file("tokenizer-wordpiece.json")

def evaluate_tokenizer(tokenizer, text):
    if not text.strip():
        print("\nTexto original: [Línea vacía]")
        return
    
    encoded = tokenizer.encode(text)
    print(f"\nTexto original: {text}")
    print("Tokens y sus IDs:")
    for token, id in zip(encoded.tokens, encoded.ids):
        print(f"Token: {token:<15} ID: {id}")
    print(f"Número de tokens: {len(encoded.tokens)}")
    if len(encoded.tokens) > 0:
        unk_percentage = encoded.tokens.count('[UNK]') / len(encoded.tokens) * 100
        print(f"Porcentaje de tokens desconocidos: {unk_percentage:.2f}%")
    else:
        print("No se generaron tokens.")
    return encoded.tokens

def evaluate_token_distribution(tokens):
    lengths = [len(token) for token in tokens]
    counter = Counter(lengths)
    print("\nDistribución del tamaño de los tokens:")
    for length, count in sorted(counter.items()):
        print(f"Longitud {length}: {count} tokens")

def evaluate_corpus_coverage(tokenizer, corpus_file):
    total_tokens = 0
    unknown_tokens = 0
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            encoded = tokenizer.encode(line.strip())
            total_tokens += len(encoded.tokens)
            unknown_tokens += encoded.tokens.count('[UNK]')
    
    coverage = (1 - unknown_tokens / total_tokens) * 100
    print(f"\nCobertura del vocabulario en el corpus: {coverage:.2f}%")

# Texto de prueba original
test_text = 'me llamo francisco tengo 25 años soy del año 1999 y soy estoico En el transcurso de su historia, que abarca tres milenios, llegó a extender sus dominios sobre toda la cuenca del Mediterráneo y gran parte de Europa, Oriente Próximo y África del Norte. Como capital de la República y del Imperio romano, llegó a ser la primera gran metrópolis de la humanidad,5​6​ centro de una de las civilizaciones antiguas más importantes. Influyó en la sociedad, la cultura, la lengua, la literatura, la música, el arte, la arquitectura, la filosofía, la política, la gastronomía, la religión, el derecho y la moral de los siglos sucesivos'

# Evaluar con el texto de prueba original
tokens = evaluate_tokenizer(loaded_tokenizer, test_text)
evaluate_token_distribution(tokens)

# Evaluar con algunas frases adicionales
additional_texts = [
    "El aprendizaje automático es una rama de la inteligencia artificial.",
    "La tokenización es un paso importante en el procesamiento del lenguaje natural.",
    "Python es un lenguaje de programación muy popular para el análisis de datos.",
    "La pandemia de COVID-19 ha acelerado la adopción de tecnologías digitales.",
    "El cambio climático es uno de los mayores desafíos que enfrenta la humanidad."
]

for text in additional_texts:
    evaluate_tokenizer(loaded_tokenizer, text)

# Evaluar la cobertura en el corpus de entrenamiento
evaluate_corpus_coverage(loaded_tokenizer, corpus_file)

# Imprimir estadísticas generales
print(f"\nEstadísticas generales:")
print(f"Tamaño del vocabulario: {loaded_tokenizer.get_vocab_size()}")

# Mostrar algunas subpalabras
vocab = loaded_tokenizer.get_vocab()
subwords = [word for word, _ in sorted(vocab.items(), key=lambda x: x[1])[:20]]
print("\nAlgunas subpalabras en el vocabulario:")
print(", ".join(subwords))

print("\nEvaluación completa del tokenizador finalizada.")