from typing import List
from random import random
from chromavector import ChromaVectorDatabase  # Assure-toi d'importer correctement ta classe


# Test des fonctionnalités de ChromaVectorDatabase
def test_chroma_vector_db():
    db_name = "my_vectors"
    vector_dim = 10
    num_vectors = 10

    # Création d'une instance de la base de données ChromaVectorDatabase
    vector_db = ChromaVectorDatabase(db_name, vector_dim)

    # Création de vecteurs aléatoires
    vectors_to_insert = [[random() for _ in range(vector_dim)] for _ in range(num_vectors)]
    ids_to_insert = [str(i) for i in range(num_vectors)]

    # Insertion de vecteurs
    vector_db.batch_upsert(vectors_to_insert, ids_to_insert)

    # Requête pour trouver les vecteurs similaires
    query_vector = vectors_to_insert[0]
    top_k = 5
    similar_vectors = vector_db.query(query_vector, top_k)

    print(f"Similar vectors to query vector {query_vector}: {similar_vectors}")

    # Réinitialisation de la base de données
    vector_db.reset()
    print("Database reset successful.")

# Exécution du test
if __name__ == "__main__":
    test_chroma_vector_db()
