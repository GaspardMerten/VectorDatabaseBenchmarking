import time

import numpy as np

from databases.postgres import PgVectorDatabase


def benchmark_database(database_class, dataset_generator, steps):
    def generate_database():
        return database_class("test_wikipedia", 1028)

    results = []

    for step in steps:
        # Generate a dataset of the given size
        dataset = lambda :(dataset_generator() for _ in range(step))

        # Measure time for bulk insertion with index
        database = generate_database()
        database.create_index()  # Create an index before bulk insertion
        start_time = time.time()
        database.batch_upsert(dataset(), size=step)
        bulk_insert_time_with_index = time.time() - start_time

        database.reset()  # Reset the database before each test
        database.close()


        # Measure time for bulk insertion without index
        database = generate_database()  # Reset the database before each test
        start_time = time.time()
        database.batch_upsert(dataset(), size=step)
        bulk_insert_time_no_index = time.time() - start_time

        # Generate a sample query vector
        query_vector = dataset_generator()

        # Measure time for search without index
        start_time = time.time()
        database.query(query_vector, top_k=10)
        search_time_no_index = time.time() - start_time

        # Measure time for search with index
        database.create_index()  # Create an index
        start_time = time.time()
        database.query(query_vector, top_k=10)
        search_time_with_index = time.time() - start_time

        # Record the results for this step size
        results.append({
            "Step Size": step,
            "Bulk Insert Time (No Index)": bulk_insert_time_no_index,
            "Bulk Insert Time (With Index)": bulk_insert_time_with_index,
            "Search Time (No Index)": search_time_no_index,
            "Search Time (With Index)": search_time_with_index,
            "Database size": database.get_storage_size(),
        })

        # Beautify the results
        print(f"Step Size: {step}")
        print(f"Bulk Insert Time (No Index): {bulk_insert_time_no_index}")
        print(f"Bulk Insert Time (With Index): {bulk_insert_time_with_index}")
        print(f"Search Time (No Index): {search_time_no_index}")
        print(f"Search Time (With Index): {search_time_with_index}")
        print(f"Database size: {database.get_storage_size() / 1024 / 1024} MB")
        print()
        database.close()

    return results


if __name__ == '__main__':
    benchmark_database(PgVectorDatabase, lambda: np.random.rand(1028).tolist(),
                       [10, 11, 1000, 10000, 100000, 1000000])
