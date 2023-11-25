import time
from typing import Type, Dict

from src.benchmark.steps.base import BenchmarkStep
from src.data.base import BenchmarkDataset
from src.databases.base import AVectorDatabase


class QueryStep(BenchmarkStep):
    """
    Query step for the benchmark.
    """

    def __init__(self, dataset: BenchmarkDataset, count: int, database_class: Type[AVectorDatabase]):
        """
        Initialize the query step.

        dataset (BenchmarkDataset): Dataset to query
        count (int): Number of vectors to query
        """
        super().__init__(dataset, count, database_class)
        self.dataset = dataset
        self.count = count
        self.database_class = database_class

    def get_name(self) -> str:
        """
        Get the name of the benchmark step.
        """
        return f"query_{self.database_class.get_name()}_{self.count}"

    def get_description(self) -> str:
        """
        Get the description of the benchmark step.
        """
        return "Query the database."

    def run(self) -> Dict:
        database = self.database_class("benchmark", self.dataset.get_vector_size())
        generator = self.dataset.retrieve_generator(self.count + 6)
        first_six = [generator.__next__() for _ in range(6)]
        database.create_index()
        database.batch_upsert(generator, self.count)

        measures = []

        for index, query in enumerate(first_six):
            # Measure time for bulk query
            start = time.time()
            database.query(query, top_k=100)
            end = time.time()

            if index > 0:
                measures.append(end - start)

        return {
            "time": sum(measures) / len(measures),
        }
