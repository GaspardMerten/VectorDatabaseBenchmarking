import abc
from typing import List

Vector = List[float]


class AVectorDatabase(abc.ABC):
    """
    Abstract class for vector database.

    Defines the interface for a vector database. Focusing
    on the following methods:
        - __init__: Initialize the database, create a collection, if it exists, reset it
        - batch_upsert: Upsert a batch of vectors
        - upsert: Upsert a single vector
        - query: Query the database for similar vectors, top-k
        - reset: Reset the database, delete the collection
        - create_index: Create an index on the vector column
    """

    @abc.abstractmethod
    def __init__(self, name: str, vector_dim: int) -> None:
        """
        Initialize the database, create a collection, if it exists, reset it.

        Args:
            name (str): Name of the database
            vector_dim (int): Dimension of the vectors
        """
        pass

    @abc.abstractmethod
    def batch_upsert(self, vectors: List[Vector], size: int = 2000) -> None:
        """
        Upsert a batch of vectors.

        Args:
            vectors (List[Vector]): List of vectors to upsert
            size (int): Number of vectors to upsert
        """
        pass

    @abc.abstractmethod
    def upsert(self, vector: Vector) -> None:
        """
        Upsert a single vector.

        Args:
            vector (Vector): Vector to upsert
        """
        pass

    @abc.abstractmethod
    def query(self, vector: Vector, top_k: int) -> List[str]:
        """
        Query the database for similar vectors, top-k.

        Args:
            vector (Vector): Vector to query
            top_k (int): Number of results to return

        Returns:
            List[str]: List of ids of the top-k results
        """
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        """
        Reset the database, delete the collection.
        """
        pass

    @abc.abstractmethod
    def create_index(self) -> None:
        """
        Create an index on the vector column.
        """
        pass

    @abc.abstractmethod
    def close(self) -> None:
        """
        Close the connection to the database.
        """
        pass

    @abc.abstractmethod
    def get_storage_size(self) -> int:
        """
        Get the storage size of the database in bytes.
        """
        pass

    def __del__(self):
        self.close()
