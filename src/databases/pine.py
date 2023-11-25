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

    def batch_upsert(self, vectors: Iterable, size: int = 2000) -> None:
        """
        Upsert a batch of vectors using pgvector.
        """

        batch_size = min(size, 1000)

        for batch in range(0, size, batch_size):
            batch_vectors = []

            for vec in vectors:
                batch_vectors.append(tuple(vec))
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
        self.index.query(
            vector=vector,
            top_k=top_k,
        )

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
        # Get number of vectors in the index
        print(self.index.describe_index_stats())

    def close(self) -> None:
        """
        Close the database connection.
        """
        pass
