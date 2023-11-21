from chroma import ChromaVectorDatabase  
import random

def generate_random_vector(dim):
    return [random.random() for _ in range(dim)]

def test_chroma_vector_database():
    # Initialize the ChromaDB vector database
    db_name = "test"
    vector_dim = 64  # Adjust the vector dimension as needed
    chroma_db = ChromaVectorDatabase(db_name, vector_dim)

    # Generate random vectors for testing
    num_vectors = 100
    vectors = [generate_random_vector(vector_dim) for _ in range(num_vectors)]

    # Test batch upsert
    chroma_db.batch_upsert(vectors)

    # Test single vector upsert
    single_vector = generate_random_vector(vector_dim)
    chroma_db.upsert(single_vector)

    # Test vector query
    query_vector = generate_random_vector(vector_dim)
    top_k = 5
    results = chroma_db.query(query_vector, top_k)
    print(f"Query results for vector {query_vector}: {results}")

    # Test reset
    chroma_db.reset()

    # Close the database connection (if applicable)
    chroma_db.close()

if __name__ == "__main__":
    test_chroma_vector_database()



#Pour test : python test_chroma_vector_database.py
