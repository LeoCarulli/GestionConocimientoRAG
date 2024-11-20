from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from query import ask_question

# Configuración
VECTOR_DB_PATH = "db_test"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Inicializar la base de datos vectorial
vectorstore = Chroma(
    collection_name="document_embeddings",
    persist_directory=VECTOR_DB_PATH,
    embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
)

# Probar una pregunta
question = "¿Qué es una ontología?"
response = ask_question(question, vectorstore)
print("Respuesta del modelo:")
print(response)
