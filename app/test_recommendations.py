"""
Simple test script to find similar movie reviews without Streamlit.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_objectbox.vectorstores import ObjectBox
from utils import huggingface_instruct_embedding, find_similar_reviews


def setup_vector_store(csv_path='../data/IMDB Dataset.csv', db_path='../objectbox_movies'):
    """
    Initialize the ObjectBox vector store from a CSV file.
    
    Args:
        csv_path: path to movie reviews CSV
        db_path: path where ObjectBox database will be stored
    
    Returns:
        ObjectBox vector store instance
    """
    print("Loading embeddings model...")
    embeddings = huggingface_instruct_embedding()
    
    print(f"Loading reviews from {csv_path}...")
    loader = CSVLoader(csv_path)
    docs = loader.load()
    
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs)
    
    print(f"Creating ObjectBox vector store ({len(final_documents)} chunks)...")
    vectors = ObjectBox.from_documents(
        final_documents, 
        embeddings, 
        embedding_dimensions=768, 
        db_directory=db_path
    )
    
    print("✅ Vector store ready!\n")
    return vectors


def get_movie_recommendations(user_preferences, vector_store=None):
    """
    Get top 5 movie recommendations based on user preferences.
    
    Args:
        user_preferences: str - what the user is looking for
        vector_store: ObjectBox instance (if None, will initialize)
    
    Returns:
        list of dicts with review, similarity_score, and match_percentage
    """
    if vector_store is None:
        vector_store = setup_vector_store()
    
    print(f"🔍 Searching for reviews matching: '{user_preferences}'\n")
    
    recommendations = find_similar_reviews(vector_store, user_preferences, k=5)
    
    print("=" * 80)
    print("TOP 5 MATCHING MOVIE REVIEWS")
    print("=" * 80)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n#{i} - Match: {rec['match_percentage']}% (Score: {rec['similarity_score']})")
        print("-" * 80)
        print(rec['review'][:500] + "..." if len(rec['review']) > 500 else rec['review'])
        print()
    
    return recommendations


if __name__ == "__main__":
    # Initialize the vector store once
    print("INITIALIZING VECTOR STORE...\n")
    vectors = setup_vector_store()
    
    # Test with different preferences
    test_preferences = [
        "I like action movies with great reviews",
        "Recommend me thrilling sci-fi films",
        "I want movies with positive sentiment and a great sense of humor"
    ]
    
    for preference in test_preferences:
        results = get_movie_recommendations(preference, vectors)
        print("\n" + "=" * 80 + "\n")
