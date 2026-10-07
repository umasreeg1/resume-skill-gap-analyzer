from typing import List, Dict


def generate_explainable_insights(
    strong_matches: List[dict],
    partial_matches: List[dict],
    missing_skills: List[dict],
    prioritized_skills: List[dict],
    embedding_model_name: str
) -> List[dict]:
    """
    Generates transparent 'Why?' explanations for AI decision-making during resume evaluation.
    """
    insights = []

    # Model explanation
    insights.append({
        "topic": "AI Vector Embedding Model",
        "type": "Architecture",
        "explanation": f"Evaluated using '{embedding_model_name}'. Resume concepts and job requirements were converted into dense numerical embedding vectors, and matched via multi-dimensional Cosine Similarity."
    })

    # Strong Match explanations
    if strong_matches:
        top_strong = strong_matches[:3]
        skills_str = ", ".join([s["skill"] for s in top_strong])
        insights.append({
            "topic": f"Strong Matches ({len(strong_matches)} skills)",
            "type": "Strong Match",
            "explanation": f"Skills like {skills_str} were explicitly detected in your resume with high confidence (semantic similarity >= 85%)."
        })

    # Partial Match explanations
    if partial_matches:
        for pm in partial_matches[:3]:
            insights.append({
                "topic": f"Partial Match: {pm['skill']}",
                "type": "Partial Match",
                "explanation": f"Skill '{pm['skill']}' is not explicitly written in the resume, but related concept '{pm['resume_match']}' was detected with {pm['score']}% semantic similarity."
            })

    # High Priority Missing Skill explanations
    high_priority_missing = [p for p in prioritized_skills if p["priority"] == "HIGH"]
    if high_priority_missing:
        for hp in high_priority_missing[:3]:
            insights.append({
                "topic": f"Why is {hp['skill']} marked as High Priority Missing?",
                "type": "High Priority",
                "explanation": f"{hp['reason']} Semantic similarity between resume text and '{hp['skill']}' is below the 50% threshold."
            })

    return insights
