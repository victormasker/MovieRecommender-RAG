from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


# setup local LLM via Ollama
def local_llm():
    llm = ChatOllama(model='llama3:8b')
    return llm

# setup huggingface_instruct_embedding
def huggingface_instruct_embedding():
    embeddings = HuggingFaceBgeEmbeddings(
                model_name='BAAI/bge-small-en-v1.5',  #sentence-transformers/all-MiniLM-l6-v2
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
    )

    return embeddings


# Function to find top 5 similar movie reviews
def find_similar_reviews(vector_store, user_preferences, k=5):
    """
    Find the k most similar movie reviews based on user preferences.
    
    Args:
        vector_store: ObjectBox vector store instance
        user_preferences: str - user's movie preferences (e.g., "I like action movies with great reviews")
        k: int - number of results to return (default 5)
    
    Returns:
        list of tuples: [(review_text, similarity_score), ...]
        similarity_score is between 0 and 1 (1.0 = perfect match)
    """
    results = vector_store.similarity_search_with_scores(user_preferences, k=k)
    
    # Convert results to list of tuples with formatted output
    recommendations = [
        {
            'review': doc.page_content,
            'similarity_score': round(score, 4),
            'match_percentage': round(score * 100, 2)
        }
        for doc, score in results
    ]
    
    return recommendations