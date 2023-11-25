from typing import List
import pinecone

from .base import AVectorDatabase, Vector


class Pine(AVectorDatabase):
    def __init__(self, name: str, vector_dim: int) -> None:
        """
        Initialize the Pinecone database, create an index, etc.
        """
        pinecone.create_index(name, dimension=vector_dim, metric="euclidean")
        pinecone.describe_index(name)
        self.index = pinecone.Index(name)

    def batch_upsert(self, vectors: List[Vector]) -> None:
        for vector in vectors:
            self.upsert(vector)

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
            include_values=True
        )

    def reset(self, name: str) -> None:
        """
        :param name: name of the index
        :return: None
        Delete the index called 'name'
        """
        pinecone.delete_index(name)
