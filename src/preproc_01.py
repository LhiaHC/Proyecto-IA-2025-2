"""
Funciones de preprocesamiento de texto.
Incluye: limpieza, tokenización, lematización, stopwords y corrección de cacografía.
"""

import re
import string
from typing import List
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Descargar recursos NLTK necesarios (solo se ejecuta una vez)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("corpora/omw-1.4")
except LookupError:
    nltk.download("omw-1.4", quiet=True)


# Diccionario de corrección de cacografía común en phishing
# Basado en Abiramasundari y Ramaswamy (2025)
CACOGRAPHY_DICT = {
    "paypa1": "paypal",
    "p4ypal": "paypal",
    "paypa!": "paypal",
    "g00gle": "google",
    "g0ogle": "google",
    "g00g1e": "google",
    "amaz0n": "amazon",
    "am4zon": "amazon",
    "micr0soft": "microsoft",
    "micros0ft": "microsoft",
    "facebk": "facebook",
    "faceb00k": "facebook",
    "tw1tter": "twitter",
    "twitt3r": "twitter",
    "app1e": "apple",
    "appl3": "apple",
    "netf1ix": "netflix",
    "netfl1x": "netflix",
    "b4nk": "bank",
    "acc0unt": "account",
    "acc0unt": "account",
    "ver1fy": "verify",
    "verif1": "verify",
    "c1ick": "click",
    "cl1ck": "click",
    "upd4te": "update",
    "updat3": "update",
    "secur1ty": "security",
    "sec0rity": "security",
}


def correct_cacography(text: str) -> str:
    """
    Corrige errores ortográficos intencionales comunes en phishing.
    Utiliza un diccionario de sustituciones predefinidas.

    Args:
        text: Texto con posibles errores intencionales.

    Returns:
        Texto con correcciones aplicadas.
    """
    words = text.split()
    corrected_words = []

    for word in words:
        # Buscar en diccionario (case-insensitive)
        word_lower = word.lower()
        if word_lower in CACOGRAPHY_DICT:
            corrected_words.append(CACOGRAPHY_DICT[word_lower])
        else:
            corrected_words.append(word)

    return " ".join(corrected_words)


def clean_text(text: str) -> str:
    """
    Limpia el texto básico: convierte a minúsculas, elimina caracteres especiales
    y normaliza espacios.

    Args:
        text: Texto crudo.

    Returns:
        Texto limpio.
    """
    if not isinstance(text, str):
        text = str(text)

    # Convertir a minúsculas
    text = text.lower()

    # Eliminar URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Eliminar emails
    text = re.sub(r"\S+@\S+", "", text)

    # Eliminar números (opcional, puedes comentar si quieres mantenerlos)
    # text = re.sub(r'\d+', '', text)

    # Eliminar puntuación
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Normalizar espacios múltiples, tabs, newlines
    text = re.sub(r"\s+", " ", text)

    # Quitar espacios al inicio y final
    text = text.strip()

    return text


def tokenize_text(text: str) -> List[str]:
    """
    Tokeniza el texto en palabras individuales.

    Args:
        text: Texto limpio.

    Returns:
        Lista de tokens.
    """
    return word_tokenize(text)


def remove_stopwords(tokens: List[str], language: str = "english") -> List[str]:
    """
    Elimina stopwords de la lista de tokens.

    Args:
        tokens: Lista de tokens.
        language: Idioma para stopwords.

    Returns:
        Lista de tokens sin stopwords.
    """
    stop_words = set(stopwords.words(language))
    return [token for token in tokens if token not in stop_words and len(token) > 2]


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """
    Lematiza los tokens (reduce palabras a su forma base).

    Args:
        tokens: Lista de tokens.

    Returns:
        Lista de tokens lematizados.
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]


def preprocess_full(text: str) -> str:
    """
    Pipeline completo de preprocesamiento según metodología propuesta:
    1. Limpieza básica
    2. Corrección de cacografía
    3. Tokenización
    4. Eliminación de stopwords
    5. Lematización

    Args:
        text: Texto crudo del correo.

    Returns:
        Texto preprocesado como string (tokens unidos por espacios).
    """
    # 1. Limpieza básica
    text = clean_text(text)

    # 2. Corrección de cacografía
    text = correct_cacography(text)

    # 3. Tokenización
    tokens = tokenize_text(text)

    # 4. Eliminar stopwords
    tokens = remove_stopwords(tokens)

    # 5. Lematización
    tokens = lemmatize_tokens(tokens)

    # Unir tokens en string
    return " ".join(tokens)
