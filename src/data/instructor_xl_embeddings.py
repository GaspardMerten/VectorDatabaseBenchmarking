from datasets import load_dataset

from src.data.base import BenchmarkDataset


class InstructorXLDataset(BenchmarkDataset):
    def get_vector_size(self) -> int:
        return 384

    def retrieve_generator(self, count: int):
        dataset = load_dataset("Qdrant/arxiv-titles-instructorxl-embeddings")
        return dataset

    @classmethod
    def get_name(cls) -> str:
        return "instructor_xl"
