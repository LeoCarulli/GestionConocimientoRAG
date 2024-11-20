import pdfplumber

def read_txt(file_path):
    """
    Lee el contenido de un archivo .txt.
    
    :param file_path: Ruta del archivo .txt.
    :return: Texto completo del archivo.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        raise ValueError(f"Error al leer el archivo .txt: {e}")

def read_pdf(file_path):
    """
    Extrae el texto de un archivo PDF.
    
    :param file_path: Ruta del archivo .pdf.
    :return: Texto completo del PDF.
    """
    try:
        with pdfplumber.open(file_path) as pdf:
            return ''.join([page.extract_text() for page in pdf.pages])
    except Exception as e:
        raise ValueError(f"Error al leer el archivo PDF: {e}")
