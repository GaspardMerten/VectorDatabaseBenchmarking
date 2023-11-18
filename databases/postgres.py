from typing import List, Generator, Iterable

import psycopg2
from psycopg2.extras import execute_batch

from databases.base import AVectorDatabase

Vector = List[float]


class PgVectorDatabase(AVectorDatabase):
    """
    pgvector implementation of the vector database.

    This class interacts with a PostgreSQL database using pgvector
    for efficient vector operations.
    """

    def __init__(self, name: str, vector_dim: int) -> None:
        """
        Initialize the pgvector database, create a table, if it exists, reset it.
        """
        self.connection = psycopg2.connect("postgresql://postgres:postgres@localhost:5432/postgres")
        self.cursor = self.connection.cursor()
        self.table_name = name
        self.vector_dim = vector_dim
        self.reset()

        # Create table with vector column
        self.cursor.execute(
            f"CREATE TABLE IF NOT EXISTS {self.table_name} (id SERIAL PRIMARY KEY, vector vector({self.vector_dim}))"
        )
        self.connection.commit()

    def batch_upsert(self, vectors: Iterable, size: int = 2000) -> None:
        """
        Upsert a batch of vectors using pgvector.
        """

        batch_size = min(size, 500)

        for batch in range(0, size, batch_size):
            batch_vectors = []

            for vec in vectors:
                batch_vectors.append(vec)
                if len(batch_vectors) == batch_size:
                    break


            query = f"INSERT INTO {self.table_name} (vector) VALUES (%s) ON CONFLICT (id) DO UPDATE SET vector = EXCLUDED.vector"
            execute_batch(self.cursor, query, [(vec,) for vec in batch_vectors])

        self.connection.commit()

    def upsert(self, vector: Vector) -> None:
        """
        Upsert a single vector using pgvector.
        """
        self.batch_upsert([vector])

    def query(self, vector: Vector, top_k: int) -> List[str]:
        """
        Query the database for similar vectors using pgvector, top-k.
        """
        self.cursor.execute(
            f"SELECT id FROM {self.table_name} ORDER BY vector <-> %s LIMIT %s", (str(vector), top_k)
        )
        return [str(row[0]) for row in self.cursor.fetchall()]

    def reset(self) -> None:
        """
        Reset the database, delete the table.
        """
        # Drop all indexes
        self.cursor.execute(
            f"SELECT indexname FROM pg_indexes WHERE tablename = '{self.table_name}' AND schemaname = 'public'"
        )
        self.cursor.execute(f"DROP TABLE IF EXISTS {self.table_name}")
        self.connection.commit()

    def create_index(self) -> None:
        """
        Create an index on the vector column.
        """
        self.cursor.execute(f"CREATE INDEX vector_index ON {self.table_name} USING IVFFLAT(vector)")
        self.connection.commit()

    def close(self) -> None:
        """
        Close the connection to the database.
        """
        self.cursor.close()
        self.connection.close()

    def get_storage_size(self) -> int:
        """
        Get the storage size of the database.
        """
        self.cursor.execute(f"SELECT pg_total_relation_size('{self.table_name}')")
        return self.cursor.fetchone()[0]
