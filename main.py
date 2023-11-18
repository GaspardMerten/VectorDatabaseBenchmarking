import numpy as np
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

# Set this to True if you are on the Chroma team, False if you are on the PineCone team
I_AM_CHROMA_TEAM = True

VECTOR_SIZE = 1028

if I_AM_CHROMA_TEAM:
    from databases.chroma import Chroma

    db = Chroma("test_wikipedia", VECTOR_SIZE)
else:
    from databases.pine import Pine

    db = Pine("test_wikipedia", VECTOR_SIZE)


def generate_random_vectors():
    rand = np.random.rand(VECTOR_SIZE, 128)
    # Normalize the vectors by dividing each vector by its norm
    return (rand / np.linalg.norm(rand, axis=1)[:, None]).tolist()


db.batch_upsert(generate_random_vectors())
