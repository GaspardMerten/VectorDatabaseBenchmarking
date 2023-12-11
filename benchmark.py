from src.benchmark.steps.load_step import LoadStep
from src.benchmark.steps.query import QueryStep
from src.data.random import RandomDataset
from src.databases.chroma import ChromaVectorDatabase
from src.databases.pgvector import PgVectorDatabase
from src.databases.pgvectorwithindex import PgVectorWithIndexDatabase
from src.databases.pine import PineVectorDatabase

databases = [ChromaVectorDatabase, PgVectorWithIndexDatabase, PgVectorDatabase, PineVectorDatabase]

_steps = [
    (LoadStep, {"count": 1000}),
    (LoadStep, {"count": 10_000}),
    (LoadStep, {"count": 100_000}),
    (LoadStep, {"count": 1_000_000}),
    (QueryStep, {"count": 1000}),
    (QueryStep, {"count": 10_000}),
    (QueryStep, {"count": 100_000}),
    (QueryStep, {"count": 1_000_000}),
]


def benchmark(steps, output_file="benchmark.csv"):
    with open(output_file, "w") as file:
        file.write("database,step,count,time,storage_size\n")

    for database in databases:
        for step, kwargs in steps:
            file = open(output_file, "a")
            print(f"Running {step.__name__} on {database.__name__} with {kwargs}")
            result = step(RandomDataset(), database_class=database, **kwargs).run()
            file.write(f"{database.__name__},{step.__name__},{kwargs['count']},{','.join(map(str, result.values()))}\n")


if __name__ == "__main__":
    benchmark(_steps)
