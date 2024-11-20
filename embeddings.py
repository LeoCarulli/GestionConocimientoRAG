from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import TokenTextSplitter
from utils import read_txt, read_pdf
import logging

# Configuración del logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuración
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DB_PATH = "db_test"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Inicializar el modelo de embeddings
embedding_service = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Inicializar el divisor de texto basado en tokens
text_splitter = TokenTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

def create_and_store_embeddings(file_path, collection_name="document_embeddings"):
    """
    Lee un archivo, genera embeddings y los almacena en ChromaDB.
    
    :param file_path: Ruta del archivo a procesar.
    :param collection_name: Nombre de la colección en ChromaDB.
    """
    logger.info(f"Leyendo archivo {file_path}...")

    # Leer el archivo según el formato
    if file_path.endswith(".txt"):
        text = read_txt(file_path)
    elif file_path.endswith(".pdf"):
        text = read_pdf(file_path)
    else:
        logger.error("Formato de archivo no soportado. Usa .txt o .pdf.")
        return

    # Dividir el texto en fragmentos
    fragments = text_splitter.split_text(text)
    logger.info(f"Fragmentos generados: {len(fragments)}")

    # Inicializar la base de datos vectorial
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embedding_service
    )

    # Generar y almacenar embeddings
    for idx, fragment in enumerate(fragments):
        try:
            vectorstore.add_texts([fragment], ids=[f"doc_{idx}"])
            logger.info(f"Fragmento {idx + 1} almacenado en la base de datos.")
        except Exception as e:
            logger.error(f"Error al almacenar el fragmento {idx + 1}: {e}")

    logger.info("Embeddings generados y almacenados correctamente en ChromaDB.")
