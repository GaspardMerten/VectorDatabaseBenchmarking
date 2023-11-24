from base import AVectorDatabase
import chromadb
from typing import List

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
        self.chroma_client = chromadb.Client(settings = chromadb.Settings(allow_reset=True))
        self.vector_dim = vector_dim
        self.collection_name = name
        self.collection = self.chroma_client.create_collection(name=name)

    def upsert(self, vector: Vector, id: str) -> None:
        """
        Upsert a single vector using ChromaDB.
        """
        self.collection.add( embeddings=vector, ids=id)

    def batch_upsert(self, vectors: List[Vector]) -> None:
        """
        Upsert a batch of vectors using ChromaDB.
        """
        for vector in vectors:
            self.upsert(vector)
        

    def query(self, vector: Vector, top_k: int) -> List[str]:
        """
        Query the database for similar vectors using ChromaDB, top-k.
        """
        return self.collection.query(
            n_results=top_k,
            query_embeddings=vector
            )
    


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
        pass

