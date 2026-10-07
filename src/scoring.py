import numpy as np
from typing import Dict, List
from src.skill_extractor import load_skills_dictionary, extract_skills_from_text


def calculate_explainable_scores(
    tfidf_sim_score: float,
    semantic_match_result: dict,
    resume_text: str,
    job_text: str,
    skills_dict: Dict[str, List[str]] = None
) -> dict:
    """
    Computes transparent, explainable match scores across multiple categories.
    
    Formula:
    Overall Score = 0.35 * TFIDF_Sim + 0.45 * (Strong_Matches / Total_Job_Skills) + 0.20 * (Partial_Matches_Weight / Total_Job_Skills)
    """
    if skills_dict is None:
        skills_dict = load_skills_dictionary()

    strong_matches = semantic_match_result.get("strong_matches", [])
    partial_matches = semantic_match_result.get("partial_matches", [])
    missing_skills = semantic_match_result.get("missing_skills", [])
    job_skills = semantic_match_result.get("job_skills", [])

    total_job_skills_count = max(len(job_skills), 1)

    strong_count = len(strong_matches)
    partial_count = len(partial_matches)

    strong_ratio = strong_count / total_job_skills_count
    partial_ratio = (partial_count * 0.5) / total_job_skills_count

    # Weighted Overall Score
    tfidf_contrib = tfidf_sim_score * 0.35
    skill_coverage_contrib = (strong_ratio * 100) * 0.45
    partial_contrib = (partial_ratio * 100) * 0.20

    overall_score = round(min(100.0, tfidf_contrib + skill_coverage_contrib + partial_contrib), 1)

    # Category Level Breakdown
    res_cat_skills = extract_skills_from_text(resume_text, skills_dict)
    job_cat_skills = extract_skills_from_text(job_text, skills_dict)

    category_scores = {}
    for cat in skills_dict.keys():
        j_skills = set(job_cat_skills.get(cat, []))
        r_skills = set(res_cat_skills.get(cat, []))

        if not j_skills:
            category_scores[cat] = {
                "score": 100.0 if r_skills else 0.0,
                "matched_count": len(r_skills),
                "required_count": 0,
                "status": "Not Required"
            }
        else:
            matched_count = len(j_skills.intersection(r_skills))
            cat_match_pct = round((matched_count / len(j_skills)) * 100, 1)
            category_scores[cat] = {
                "score": cat_match_pct,
                "matched_count": matched_count,
                "required_count": len(j_skills),
                "status": f"{matched_count}/{len(j_skills)} Matched"
            }

    explanation_formula = (
        f"Overall Score ({overall_score}%) = "
        f"35% TF-IDF Document Similarity ({round(tfidf_contrib, 1)}%) + "
        f"45% Direct Skill Coverage ({round(skill_coverage_contrib, 1)}%) + "
        f"20% Partial Concept Coverage ({round(partial_contrib, 1)}%)"
    )

    return {
        "overall_score": overall_score,
        "tfidf_sim_score": tfidf_sim_score,
        "skill_coverage_score": round(strong_ratio * 100, 1),
        "partial_match_score": round(partial_ratio * 100, 1),
        "category_scores": category_scores,
        "explanation_formula": explanation_formula
    }
