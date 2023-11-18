import time

import numpy as np
from dotenv import load_dotenv

from databases.postgres import PgVectorDatabase

load_dotenv()  # take environment variables from .env.

# Set this to True if you are on the Chroma team, False if you are on the PineCone team
I_AM_CHROMA_TEAM = True

VECTOR_SIZE = 1028


def generate_random_vectors(count: int):
    rand = np.random.rand(count, VECTOR_SIZE)
    # Normalize the vectors by dividing each vector by its norm
    return (rand / np.linalg.norm(rand, axis=1)[:, None]).tolist()


db = PgVectorDatabase("test_wikipedia", VECTOR_SIZE)
db.batch_upsert(generate_random_vectors(count=10000), size=10000)
start = time.time()
print(db.query(generate_random_vectors(count=1)[0], top_k=10))
end = time.time()
print(end - start)
db.create_index()
start = time.time()
print(db.query(generate_random_vectors(count=1)[0], top_k=10))
end = time.time()
print(end - start)
