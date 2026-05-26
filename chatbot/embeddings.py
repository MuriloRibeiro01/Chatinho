from sentence_transformers import SentenceTransformer

# Define o modelo do transformer a ser usado
model = SentenceTransformer(
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
)

# Gera vetorização para o objeto criado pelo usuário
def gerar_embedding(texto):
    return model.encode(texto)