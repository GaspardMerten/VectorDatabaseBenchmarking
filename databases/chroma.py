from typing import List, Iterable
from chromadb import ChromaDB

Vector = List[float]

class ChromaVectorDatabase:
    """
    ChromaDB implementation of the vector database.

    This class interacts with ChromaDB for efficient vector operations.
    """

    def __init__(self, name: str, vector_dim: int) -> None:
        """
        Initialize the ChromaDB database, create a collection, etc.
        """
        self.db = ChromaDB(name)
        self.collection = self.db.create_collection(name, dimension=vector_dim)

    def batch_upsert(self, vectors: Iterable) -> None:
        """
        Upsert a batch of vectors using ChromaDB.
        """
        for vector in vectors:
            self.collection.insert([vector])

    def upsert(self, vector: Vector) -> None:
        """
        Upsert a single vector using ChromaDB.
        """
        self.batch_upsert([vector])

    def query(self, vector: Vector, top_k: int) -> List[str]:
        """
        Query the database for similar vectors using ChromaDB, top-k.
        """
        result = self.collection.query(vector, top_k)
        return [str(item["_id"]) for item in result]

    def reset(self) -> None:
        """
        Reset the database, delete the collection.
        """
        self.db.drop_collection(self.collection.name)

    def create_index(self) -> None:
        """
        ChromaDB doesn't use traditional indexes, so this method might not be applicable.
        """
        pass

    def close(self) -> None:
        """
        Close the connection to the database.
        """
        pass  # ChromaDB doesn't require explicit closing

    def get_storage_size(self) -> int:
        """
        Get the storage size of the database.
        """
        return 0  # ChromaDB may not provide direct access to storage size information
