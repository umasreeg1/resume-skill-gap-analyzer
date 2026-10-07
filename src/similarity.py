import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.text_preprocessing import clean_text


def calculate_similarity(resume_raw: str, job_raw: str) -> dict:
    """
    Computes TF-IDF vector representations and Cosine Similarity 
    between raw resume text and job description text.
    
    Returns:
        dict: {
            "match_percentage": float,
            "cosine_sim": float,
            "top_overlapping_words": list
        }
    """
    if not resume_raw or not resume_raw.strip():
        raise ValueError("Resume text is empty or missing.")
    if not job_raw or not job_raw.strip():
        raise ValueError("Job description text is empty or missing.")

    # Preprocess both texts
    resume_clean = clean_text(resume_raw, remove_stopwords=True)
    job_clean = clean_text(job_raw, remove_stopwords=True)

    if len(resume_clean.split()) < 1:
        raise ValueError("Resume content is empty or unreadable.")
    if len(job_clean.split()) < 1:
        raise ValueError("Job description content is empty or unreadable.")


    # Initialize TF-IDF Vectorizer with unigrams and bigrams
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    
    # Fit and transform texts into TF-IDF sparse matrix
    tfidf_matrix = vectorizer.fit_transform([resume_clean, job_clean])
    
    # Calculate cosine similarity between vector 0 (Resume) and vector 1 (Job Description)
    sim_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    cosine_sim = float(sim_matrix[0][0])
    
    # Convert to match percentage (0 to 100%)
    match_percentage = round(cosine_sim * 100, 1)

    # Extract top overlapping terms based on TF-IDF weight product
    feature_names = np.array(vectorizer.get_feature_names_out())
    r_vec = tfidf_matrix[0].toarray()[0]
    j_vec = tfidf_matrix[1].toarray()[0]
    
    # Overlap weight is the element-wise product of TF-IDF scores
    overlap_weights = r_vec * j_vec
    top_indices = np.argsort(overlap_weights)[::-1]
    
    top_overlapping = []
    for idx in top_indices:
        if overlap_weights[idx] > 0 and len(top_overlapping) < 10:
            top_overlapping.append(feature_names[idx])

    return {
        "match_percentage": match_percentage,
        "cosine_sim": round(cosine_sim, 4),
        "top_overlapping_words": top_overlapping
    }
