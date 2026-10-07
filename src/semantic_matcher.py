import os
import re
import numpy as np
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.skill_extractor import extract_skills_from_text, load_skills_dictionary
from src.text_preprocessing import clean_text

# Global model cache to avoid re-loading on every call
_ST_MODEL = None
_USING_ST = False

# Domain context enricher for sentence transformers semantic embedding
SKILL_CONTEXT = {
    "Scikit-learn": "Scikit-learn Machine Learning classification and regression library",
    "Pandas": "Pandas Data Analysis and data manipulation library",
    "NumPy": "NumPy numerical computing Data Analysis",
    "TensorFlow": "TensorFlow Deep Learning and neural network framework",
    "PyTorch": "PyTorch Deep Learning and neural network framework",
    "Keras": "Keras Deep Learning framework",
    "SQL": "SQL relational database querying and management",
    "PostgreSQL": "PostgreSQL SQL database management",
    "MySQL": "MySQL SQL database management",
    "AWS": "AWS Amazon Web Services Cloud infrastructure",
    "Azure": "Microsoft Azure Cloud infrastructure",
    "Google Cloud": "GCP Google Cloud Platform infrastructure",
    "Docker": "Docker containerization DevOps tool",
    "Kubernetes": "Kubernetes container orchestration Cloud DevOps",
    "React": "React frontend web development library",
    "Node.js": "Node.js backend web server framework",
    "HTML": "HTML web development frontend markup",
    "CSS": "CSS web styling frontend design",
    "Git": "Git version control development tool",
    "OpenCV": "OpenCV Computer Vision and image processing",
    "NLP": "Natural Language Processing and text analytics",
    "Natural Language Processing": "NLP Natural Language Processing text analytics"
}


def _get_sentence_transformer():
    """
    Lazy loader for SentenceTransformer ('all-MiniLM-L6-v2').
    Falls back gracefully to TF-IDF N-gram Semantic Vector Space if PyTorch/ST is loading.
    """
    global _ST_MODEL, _USING_ST
    if _ST_MODEL is not None:
        return _ST_MODEL, _USING_ST

    try:
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        _USING_ST = True
        return _ST_MODEL, True
    except Exception:
        _ST_MODEL = None
        _USING_ST = False
        return None, False


def compute_text_embeddings(texts: List[str]):
    """
    Generates semantic embedding vectors for a list of text strings.
    Uses SentenceTransformer if available, or TF-IDF N-gram dense vector space fallback.
    """
    st_model, using_st = _get_sentence_transformer()

    if using_st and st_model is not None:
        embeddings = st_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings, "SentenceTransformer (all-MiniLM-L6-v2)"

    # Fallback: TF-IDF N-Gram Vector Space Embeddings
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer='char_wb')
    matrix = vectorizer.fit_transform([t.lower() for t in texts]).toarray()
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized_matrix = matrix / norms
    return normalized_matrix, "TF-IDF N-Gram Character Vector Space"


def perform_semantic_skill_matching(
    resume_text: str,
    job_text: str,
    skills_dict: Dict[str, List[str]] = None
) -> dict:
    """
    Executes core AI semantic skill matching using Sentence Embeddings and Cosine Similarity.
    
    Returns:
        dict: {
            "strong_matches": list of dicts,
            "partial_matches": list of dicts,
            "missing_skills": list of dicts,
            "embedding_model_name": str,
            "job_skills": list,
            "resume_skills": list
        }
    """
    if skills_dict is None:
        skills_dict = load_skills_dictionary()

    resume_cat_skills = extract_skills_from_text(resume_text, skills_dict)
    job_cat_skills = extract_skills_from_text(job_text, skills_dict)

    resume_skills = sorted(list(set(s for cat in resume_cat_skills.values() for s in cat)))
    job_skills = sorted(list(set(s for cat in job_cat_skills.values() for s in cat)))

    if not job_skills:
        tokens = re.findall(r'\b[A-Za-z0-9\+\#\.-]{2,}\b', job_text)
        job_skills = sorted(list(set(t for t in tokens if len(t) > 2))[:10])

    if not resume_skills:
        tokens_res = re.findall(r'\b[A-Za-z0-9\+\#\.-]{2,}\b', resume_text)
        resume_skills = sorted(list(set(t for t in tokens_res if len(t) > 2))[:10])

    # Build rich candidate phrases for vector embedding
    candidate_phrases = list(resume_skills)
    resume_sentences = [s.strip() for s in re.split(r'[\.\n;]', resume_text) if len(s.strip().split()) >= 3]
    candidate_phrases.extend(resume_sentences[:20])

    # Enriched texts for embedding calculation
    enriched_job_texts = [SKILL_CONTEXT.get(s, s) for s in job_skills]
    enriched_candidate_texts = [SKILL_CONTEXT.get(p, p) for p in candidate_phrases]

    all_texts_to_embed = enriched_job_texts + enriched_candidate_texts
    embeddings, model_name = compute_text_embeddings(all_texts_to_embed)

    job_embeddings = embeddings[:len(job_skills)]
    candidate_embeddings = embeddings[len(job_skills):]

    # Calculate pairwise cosine similarity matrix
    sim_matrix = cosine_similarity(job_embeddings, candidate_embeddings)

    strong_matches = []
    partial_matches = []
    missing_skills = []

    for idx, req_skill in enumerate(job_skills):
        sim_scores = sim_matrix[idx]
        best_match_idx = int(np.argmax(sim_scores))
        best_score = float(sim_scores[best_match_idx])
        best_match_phrase = candidate_phrases[best_match_idx]

        is_exact = req_skill.lower() in [s.lower() for s in resume_skills]

        if is_exact or best_score >= 0.82:
            strong_matches.append({
                "skill": req_skill,
                "resume_match": req_skill if is_exact else best_match_phrase,
                "score": round(100.0 if is_exact else best_score * 100, 1),
                "type": "Strong Match"
            })
        elif best_score >= 0.45:
            partial_matches.append({
                "skill": req_skill,
                "resume_match": best_match_phrase,
                "score": round(best_score * 100, 1),
                "type": "Partial Match",
                "explanation": f"Related concept '{best_match_phrase}' found in resume with {round(best_score*100, 1)}% semantic closeness."
            })
        else:
            missing_skills.append({
                "skill": req_skill,
                "score": round(best_score * 100, 1),
                "type": "Missing",
                "explanation": f"No direct or semantically related concept for '{req_skill}' was found (similarity < 45%)."
            })

    return {
        "strong_matches": strong_matches,
        "partial_matches": partial_matches,
        "missing_skills": missing_skills,
        "embedding_model_name": model_name,
        "job_skills": job_skills,
        "resume_skills": resume_skills
    }
