"""
load_movie_reviews.py
"""

from langchain_community.document_loaders.csv_loader import CSVLoader
from datetime import datetime, timedelta

def load_movie_reviews():
    documents = []

    for i in range(1, 5):
        loader = CSVLoader(
            file_path=f"./data/john_wick_{i}.csv",
            metadata_columns=["Review_Date", "Review_Title", "Review_Url", "Author", "Rating"]
        )

        movie_docs = loader.load()
        for doc in movie_docs:

            # Add the "Movie Title" (John Wick 1, 2, ...)
            doc.metadata["Movie_Title"] = f"John Wick {i}"

            # convert "Rating" to an `int`, if no rating is provided - assume 0 rating
            doc.metadata["Rating"] = int(doc.metadata["Rating"]) if doc.metadata["Rating"] else 0

            # newer movies have a more recent "last_accessed_at"
            doc.metadata["last_accessed_at"] = datetime.now() - timedelta(days=4-i)

        print(f'number of reviews for john_wick_{i} is: {len(movie_docs)} ')
        documents.extend(movie_docs)

    return documents
