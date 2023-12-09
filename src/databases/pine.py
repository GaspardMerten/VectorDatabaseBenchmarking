from typing import List, Iterable

import pinecone

from .base import AVectorDatabase, Vector

pinecone.init(api_key="7afac8b3-c564-4dd8-a25b-b07c373ac004", environment="gcp-starter")


class PineVectorDatabase(AVectorDatabase):
    def __init__(self, name: str, vector_dim: int) -> None:
        """
        Initialize the Pinecone database, create an index, etc.
        """
        for index in pinecone.list_indexes():
            pinecone.delete_index(index)
        pinecone.create_index(name, dimension=vector_dim, metric="euclidean")
        pinecone.describe_index(name)
        self.index = pinecone.Index(name)
        self.count = 0
        self.vector_dim = vector_dim

    def batch_upsert(self, vectors: Iterable, size: int = 2000) -> None:
        """
        Upsert a batch of vectors using pgvector.
        """
        print("Starting batch upsert")

        batch_size = min(size, 350)

        self.count += size

        for batch in range(0, size, batch_size):
            batch_vectors = []

            for i, vec in enumerate(vectors):
                batch_vectors.append((str(batch + i), vec))
                if len(batch_vectors) == batch_size:
                    break

            self.index.upsert(batch_vectors)

    def upsert(self, vector: Vector) -> None:
        self.index.upsert([vector])

    def query(self, vector: Vector, top_k: int) -> List[str]:
        """
        Query the database for similar vectors, top-k.

        Args:
            vector (Vector): Vector to query
            top_k (int): Number of results to return

        Returns:
            List[str]: List of ids of the top-k results
        """
        return self.index.query(vector, top_k=top_k).ids

    def reset(self, name: str) -> None:
        """
        :param name: name of the index
        :return: None
        Delete the index called 'name'
        """
        pinecone.delete_index(name)

    def create_index(self) -> None:
        pass

    def get_storage_size(self) -> int:
        # See: https://docs.pinecone.io/docs/choosing-index-type-and-size#dimensionality-of-vectors for more info
        return self.count * self.vector_dim * 4

    def close(self) -> None:
        """
        Close the database connection.
        """
        pass
