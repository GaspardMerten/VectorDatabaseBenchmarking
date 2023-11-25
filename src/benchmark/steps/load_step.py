import time
from typing import Type, Dict

from src.benchmark.steps.base import BenchmarkStep
from src.data.base import BenchmarkDataset
from src.databases.base import AVectorDatabase


class LoadStep(BenchmarkStep):
    """
    Load step for the benchmark.
    """

    def __init__(self, dataset: BenchmarkDataset, count: int, database_class: Type[AVectorDatabase]):
        """
        Initialize the load step.

        dataset (BenchmarkDataset): Dataset to load
        count (int): Number of vectors to load
        """
        super().__init__(dataset, count, database_class)
        self.dataset = dataset
        self.count = count
        self.database_class = database_class

    def get_name(self) -> str:
        """
        Get the name of the benchmark step.
        """
        return f"load_{self.database_class.get_name()}_{self.count}"

    def get_description(self) -> str:
        """
        Get the description of the benchmark step.
        """
        return "Load a dataset into the database."

    def run(self) -> Dict:
        database = self.database_class("benchmark", self.dataset.get_vector_size())

        # Measure time for bulk insertion with index
        start = time.time()
        database.batch_upsert(self.dataset.retrieve_generator(self.count), self.count)
        end = time.time()

        results = {}

        results["time"] = end - start
        results["size"] = database.get_storage_size()

        return results
