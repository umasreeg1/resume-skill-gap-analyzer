from typing import List, Dict


def simulate_skill_improvement(
    initial_score: float,
    current_strong_matches: List[dict],
    current_partial_matches: List[dict],
    total_job_skills_count: int,
    selected_skills_to_learn: List[str]
) -> dict:
    """
    Recalculates estimated match score if candidate acquires selected missing skills.
    Uses real scoring weights (0.45 skill coverage + 0.35 TF-IDF sim + 0.20 partial).
    """
    if total_job_skills_count <= 0:
        total_job_skills_count = 1

    current_strong_count = len(current_strong_matches)
    acquired_count = len(selected_skills_to_learn)

    new_strong_count = current_strong_count + acquired_count

    # Calculate boost from acquired skills
    coverage_boost = (acquired_count / total_job_skills_count) * 100 * 0.45
    tfidf_sim_boost = (acquired_count / total_job_skills_count) * 100 * 0.15

    total_boost = round(coverage_boost + tfidf_sim_boost, 1)
    simulated_score = round(min(100.0, initial_score + total_boost), 1)

    return {
        "initial_score": initial_score,
        "simulated_score": simulated_score,
        "score_boost": total_boost,
        "acquired_skills": selected_skills_to_learn,
        "new_strong_count": new_strong_count,
        "total_job_skills": total_job_skills_count
    }
