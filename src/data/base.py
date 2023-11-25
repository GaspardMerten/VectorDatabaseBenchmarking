import abc


class BenchmarkDataset(abc.ABC):
    @abc.abstractmethod
    def get_vector_size(self) -> int:
        """
        Get the size of the vectors in the dataset.
        """
        pass

    @abc.abstractmethod
    def retrieve_generator(self, count: int):
        """
        Get a generator for the dataset.

        count (int): Number of vectors to generate
        """
        pass

    @classmethod
    def get_name(cls) -> str:
        """
        Get the name of the dataset.
        """
        pass
