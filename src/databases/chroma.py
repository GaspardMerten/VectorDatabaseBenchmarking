import os
from typing import List, Iterable

import chromadb

from src.databases.base import AVectorDatabase

Vector = List[float]


class ChromaVectorDatabase(AVectorDatabase):
    """
    ChromaDB implementation of the vector database.

    This class interacts with ChromaDB for efficient vector operations.
    """

    def __init__(self, name: str, vector_dim: int) -> None:
        """
        Initialize the ChromaDB database, create a collection.
        """
        self.chroma_client = chromadb.PersistentClient(settings=chromadb.Settings(allow_reset=True), path='./.chroma')
        self.reset()
        self.vector_dim = vector_dim
        self.collection_name = name
        self.collection = self.chroma_client.create_collection(name=name)

    def upsert(self, vector: Vector, id: str) -> None:
        """
        Upsert a single vector using ChromaDB.
        """
        self.collection.add(embeddings=vector, ids=[id])

    def batch_upsert(self, vectors: Iterable, size: int = 2000) -> None:
        """
        Upsert a batch of vectors using pgvector.
        """

        batch_size = min(size, 1000)

        for batch in range(0, size, batch_size):
            batch_vectors = []

            for vec in vectors:
                batch_vectors.append(vec)
                if len(batch_vectors) == batch_size:
                    break

            self.collection.add(embeddings=batch_vectors, ids=[str(batch + i) for i in range(len(batch_vectors))])

    def query(self, vector: Vector, top_k: int) -> List[str]:
        """
        Query the database for similar vectors using ChromaDB, top-k.
        """
        query = dict(self.collection.query(n_results=top_k, query_embeddings=vector))

        return query

    def reset(self) -> None:
        """
        Reset the database, delete the collection.
        """
        self.chroma_client.reset()

    def create_index(self) -> None:
        """
        Create an index on the vector column.
        """
        pass

    def close(self) -> None:
        """
        Close the database connection.
        """
        pass

    def get_storage_size(self) -> int:
        """
        Get the storage size of the database.

        To do so, compute size of folder .chroma
        """

        size = 0
        for root, dirs, files in os.walk('.chroma'):
            size += sum([os.path.getsize(os.path.join(root, name)) for name in files])
        return size
