import logging
from langchain_ollama.llms import OllamaLLM
import time

# Configuración del logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def ask_question(question, vectorstore, model_name="phi3.5", top_k=5):
    """
    Recupera los fragmentos relevantes del vectorstore y genera una respuesta usando Ollama.

    :param question: Pregunta del usuario.
    :param vectorstore: Base de datos vectorial de ChromaDB.
    :param model_name: Nombre del modelo Ollama a utilizar.
    :param top_k: Número de fragmentos relevantes a recuperar.
    :return: Respuesta generada por el modelo.
    """
    start_time = time.time()
    logger.info(f"Pregunta recibida: {question}")

    # Buscar fragmentos relevantes
    fragment_search_start = time.time()
    results = vectorstore.similarity_search(query=question, k=top_k)
    fragment_search_end = time.time()

    relevant_chunks = [doc.page_content for doc in results]
    logger.info(f"Fragmentos relevantes encontrados: {len(relevant_chunks)} en {fragment_search_end - fragment_search_start:.2f} segundos")

    # Crear el contexto
    context = "\n\n".join(relevant_chunks)
    # logger.debug(f"Contexto creado: {context[:300]}...")

    # Crear el prompt
    prompt = f"""
    Eres un experto en análisis de texto. El contenido es la transcripción de una clase de Gestión del conocimiento. Basándote en los fragmentos relevantes, responde la pregunta del usuario.

    Tu respuesta tiene que ser lo más completa y detallada posible.
    
    Fragmentos relevantes:
    {context}

    Pregunta:
    {question}

    Respuesta:
    """
    logger.debug(f"Prompt creado en {time.time() - fragment_search_end:.2f} segundos")

    # Configurar e invocar el modelo Ollama
    model = OllamaLLM(model=model_name, temperature=0.0, max_tokens=1024)
    response = model.invoke(prompt)

    logger.info(f"Respuesta generada en {time.time() - start_time:.2f} segundos")
    return response
