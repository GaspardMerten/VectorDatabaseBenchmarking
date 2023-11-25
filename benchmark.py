from src.benchmark.steps.load_step import LoadStep
from src.benchmark.steps.load_with_index_step import LoadWithIndexStep
from src.benchmark.steps.query import QueryStep
from src.data.random import RandomDataset
from src.databases.chroma import ChromaVectorDatabase
from src.databases.pgvector import PgVectorDatabase
from src.databases.pine import PineVectorDatabase

databases = [PineVectorDatabase]

steps = [
    (LoadStep, {"count": 1000}),
    (LoadStep, {"count": 10_000}),
    (QueryStep, {"count": 1000}),
    (QueryStep, {"count": 10_000}),
]

for database in databases:
    for step, kwargs in steps:
        print(f"Running {step.__name__} on {database.__name__} with {kwargs}")
        print(step(RandomDataset(), database_class=database, **kwargs).run())
