"""
Generación de embeddings para correos electrónicos.
Implementa: Word2Vec, FastText y BERT (bert-base-uncased).
"""
import numpy as np
from typing import List, Optional
from gensim.models import Word2Vec, FastText
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA
import torch


class Word2VecEmbedder:
    """
    Generador de embeddings usando Word2Vec entrenado localmente.
    """

    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 2,
                 workers: int = 4, epochs: int = 10, seed: int = 42):
        """
        Args:
            vector_size: Dimensionalidad de los embeddings (100, 200, 300).
            window: Tamaño de ventana contextual.
            min_count: Frecuencia mínima de palabras.
            workers: Número de hilos para entrenamiento.
            epochs: Número de épocas de entrenamiento.
            seed: Semilla aleatoria.
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.seed = seed
        self.model = None

    def train(self, texts: List[str]):
        """
        Entrena el modelo Word2Vec con el corpus proporcionado.

        Args:
            texts: Lista de textos preprocesados (tokenizados).
        """
        # Tokenizar textos
        sentences = [text.split() for text in texts]

        print(f"Entrenando Word2Vec (dim={self.vector_size})...")
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
            seed=self.seed
        )
        print(f"Word2Vec entrenado. Vocabulario: {len(self.model.wv)} palabras.")

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transforma textos a vectores embeddings (promedio de palabras).

        Args:
            texts: Lista de textos preprocesados.

        Returns:
            Matriz de embeddings (n_texts, vector_size).
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado. Llama a train() primero.")

        embeddings = []
        for text in texts:
            words = text.split()
            # Obtener vectores de palabras que estén en el vocabulario
            word_vectors = [self.model.wv[word] for word in words if word in self.model.wv]

            if word_vectors:
                # Promedio de vectores de palabras
                embeddings.append(np.mean(word_vectors, axis=0))
            else:
                # Si no hay palabras en vocabulario, vector cero
                embeddings.append(np.zeros(self.vector_size))

        return np.array(embeddings)


class FastTextEmbedder:
    """
    Generador de embeddings usando FastText entrenado localmente.
    """

    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 2,
                 workers: int = 4, epochs: int = 10, seed: int = 42):
        """
        Args:
            vector_size: Dimensionalidad de los embeddings (100, 200, 300).
            window: Tamaño de ventana contextual.
            min_count: Frecuencia mínima de palabras.
            workers: Número de hilos para entrenamiento.
            epochs: Número de épocas de entrenamiento.
            seed: Semilla aleatoria.
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.seed = seed
        self.model = None

    def train(self, texts: List[str]):
        """
        Entrena el modelo FastText con el corpus proporcionado.

        Args:
            texts: Lista de textos preprocesados (tokenizados).
        """
        # Tokenizar textos
        sentences = [text.split() for text in texts]

        print(f"Entrenando FastText (dim={self.vector_size})...")
        self.model = FastText(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
            seed=self.seed
        )
        print(f"FastText entrenado. Vocabulario: {len(self.model.wv)} palabras.")

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transforma textos a vectores embeddings (promedio de palabras).
        FastText puede generar embeddings para palabras OOV usando subwords.

        Args:
            texts: Lista de textos preprocesados.

        Returns:
            Matriz de embeddings (n_texts, vector_size).
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado. Llama a train() primero.")

        embeddings = []
        for text in texts:
            words = text.split()
            # FastText puede manejar OOV words
            word_vectors = [self.model.wv[word] for word in words if word]

            if word_vectors:
                # Promedio de vectores de palabras
                embeddings.append(np.mean(word_vectors, axis=0))
            else:
                # Si no hay palabras, vector cero
                embeddings.append(np.zeros(self.vector_size))

        return np.array(embeddings)


class BERTEmbedder:
    """
    Generador de embeddings usando BERT preentrenado (bert-base-uncased).
    Soporta reducción dimensional opcional con PCA.
    """

    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 512,
                 batch_size: int = 16, device: str = None, pca_dim: Optional[int] = None,
                 random_state: int = 42):
        """
        Args:
            model_name: Nombre del modelo BERT preentrenado.
            max_length: Longitud máxima de tokens (BERT tiene límite de 512).
            batch_size: Tamaño de batch para procesamiento.
            device: Dispositivo ('cuda' o 'cpu'). Si None, se autodetecta.
            pca_dim: Dimensionalidad objetivo con PCA (100, 200, 300). Si None, usa 768 original.
            random_state: Semilla aleatoria para PCA.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.pca_dim = pca_dim
        self.random_state = random_state
        self.pca = None

        # Detectar dispositivo
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Cargando modelo BERT ({model_name}) en {self.device}...")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # Modo evaluación

        original_dim = self.model.config.hidden_size
        if pca_dim:
            print(f"BERT cargado. Dimensión original: {original_dim} → PCA a {pca_dim} dims")
        else:
            print(f"BERT cargado. Dimensión de salida: {original_dim}")

    def transform(self, texts: List[str], fit_pca: bool = False) -> np.ndarray:
        """
        Transforma textos a embeddings BERT usando el token [CLS].
        Aplica PCA si pca_dim fue especificado en __init__.

        Args:
            texts: Lista de textos preprocesados.
            fit_pca: Si True, ajusta PCA con estos datos (usar con train set).
                     Si False, usa PCA ya ajustado (usar con val/test set).

        Returns:
            Matriz de embeddings (n_texts, pca_dim) si PCA activado, sino (n_texts, 768).
        """
        embeddings = []

        # Procesar en batches
        print(f"Generando embeddings BERT para {len(texts)} textos...")
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]

            # Tokenizar
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            # Mover a dispositivo
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)

            # Generar embeddings
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                # Usar el embedding del token [CLS] (primera posición)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            embeddings.append(cls_embeddings)

        # Concatenar todos los batches
        embeddings_full = np.vstack(embeddings)
        print(f"Embeddings BERT generados: {embeddings_full.shape}")

        # Aplicar PCA si está configurado
        if self.pca_dim is not None:
            if fit_pca:
                print(f"Entrenando PCA para reducir de {embeddings_full.shape[1]} a {self.pca_dim} dims...")
                self.pca = PCA(n_components=self.pca_dim, random_state=self.random_state)
                embeddings_reduced = self.pca.fit_transform(embeddings_full)
                variance_explained = np.sum(self.pca.explained_variance_ratio_)
                print(f"PCA entrenado. Varianza explicada: {variance_explained:.4f}")
            else:
                if self.pca is None:
                    raise ValueError("PCA no ha sido entrenado. Llama a transform(texts, fit_pca=True) primero.")
                embeddings_reduced = self.pca.transform(embeddings_full)

            print(f"Embeddings después de PCA: {embeddings_reduced.shape}")
            return embeddings_reduced
        else:
            return embeddings_full
