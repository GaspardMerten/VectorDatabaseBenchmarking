from typing import List

from .base import AVectorDatabase, Vector


class Chroma(AVectorDatabase):
    def __init__(self, name: str, vector_dim: int) -> None:
        pass

    def batch_upsert(self, vectors: List[Vector]) -> None:
        pass

    def upsert(self, vector: Vector) -> None:
        pass

    def query(self, vector: Vector, top_k: int) -> List[str]:
        pass

    def reset(self) -> None:
        pass
