import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from query import ask_question
import logging
import time

# Configuración del logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuración del modelo y la base de datos
VECTOR_DB_PATH = "db_test"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Inicializar la base de datos vectorial
logger.info("Inicializando la base de datos vectorial...")
vectorstore = Chroma(
    collection_name="document_embeddings",
    persist_directory=VECTOR_DB_PATH,
    embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
)
logger.info("Base de datos vectorial cargada correctamente.")

# Interfaz de usuario en Streamlit
st.title("Chat de Clases con Ollama")
st.write("Haz preguntas sobre el contenido de tus documentos cargados.")

# Entrada de texto para la pregunta del usuario
question = st.text_input("Escribe tu pregunta:")

if st.button("Enviar pregunta"):
    if question:
        logger.info(f"Usuario hizo una pregunta: {question}")
        
        # Temporizador para medir el tiempo de ejecución
        start_time = time.time()
        
        try:
            # Generar la respuesta
            response = ask_question(question, vectorstore)
            end_time = time.time()
            
            # Mostrar la respuesta y el tiempo de ejecución
            st.write("Respuesta del modelo:")
            st.write(response)
            st.write(f"Tiempo total de procesamiento: {end_time - start_time:.2f} segundos")
            
            logger.info(f"Respuesta generada en {end_time - start_time:.2f} segundos.")
        except Exception as e:
            logger.error(f"Error al generar la respuesta: {e}")
            st.write("Ocurrió un error al procesar la pregunta.")
    else:
        st.write("Por favor, escribe una pregunta antes de enviarla.")