import random

from src.data.base import BenchmarkDataset


class RandomDataset(BenchmarkDataset):
    def get_vector_size(self) -> int:
        return 764

    def retrieve_generator(self, count: int):
        for _ in range(count):
            yield [random.random() for _ in range(self.get_vector_size())]

    @classmethod
    def get_name(cls) -> str:
        return "random"
