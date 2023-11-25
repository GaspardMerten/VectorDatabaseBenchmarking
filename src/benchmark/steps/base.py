import abc
from typing import Dict, Type

from src.data.base import BenchmarkDataset
from src.databases.base import AVectorDatabase


class BenchmarkStep(abc.ABC):
    """
    Abstract class for benchmark steps.

    Defines the interface for a benchmark step.
    """

    def __init__(self, dataset: BenchmarkDataset, count: int, database_class: Type[AVectorDatabase]):
        pass

    @abc.abstractmethod
    def get_name(self) -> str:
        """
        Get the name of the benchmark step.
        """
        pass

    @abc.abstractmethod
    def get_description(self) -> str:
        """
        Get the description of the benchmark step.
        """
        pass

    @abc.abstractmethod
    def run(self) -> Dict:
        """
        Run the benchmark step.

        database (AVectorDatabase): Database to run the step on
        """
        pass
