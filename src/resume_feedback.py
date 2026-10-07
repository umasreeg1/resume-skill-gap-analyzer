import re
from typing import List, Dict

ACTION_VERBS = [
    "developed", "engineered", "built", "designed", "architected",
    "optimized", "implemented", "deployed", "scaled", "automated",
    "created", "spearheaded", "accelerated", "reduced", "increased"
]


def analyze_resume_formatting_and_content(resume_text: str, missing_skills: List[str] = None) -> dict:
    """
    Analyzes resume content quality, action verb density, impact metrics, and provides concrete rewrite tips.
    """
    if not resume_text or not resume_text.strip():
        return {}

    lines = [line.strip() for line in resume_text.split('\n') if line.strip()]

    # 1. Action Verb Check
    found_action_verbs = []
    for line in lines:
        first_words = line.lower().split()[:2]
        for verb in ACTION_VERBS:
            if verb in first_words and verb not in found_action_verbs:
                found_action_verbs.append(verb)

    # 2. Metric / Quantifiable Achievement Check
    metric_pattern = r'\b(\d+%\b|\$\d+|\d+\+|\d+x\b|\d+\s*(percent|users|records|datasets|ms|sec))\b'
    lines_with_metrics = [line for line in lines if re.search(metric_pattern, line, re.IGNORECASE)]

    # 3. Suggestions Generation
    suggestions = []

    if len(found_action_verbs) < 3:
        suggestions.append({
            "category": "Action Verbs",
            "issue": "Low use of strong action verbs at start of bullet points.",
            "recommendation": "Begin project bullets with strong action verbs like 'Engineered', 'Optimized', 'Architected', or 'Deployed'."
        })

    if len(lines_with_metrics) == 0:
        suggestions.append({
            "category": "Quantifiable Metrics",
            "issue": "No measurable achievements or quantifiable metrics detected.",
            "recommendation": "Add quantifiable impact metrics (e.g. 'Improved model inference speed by 25%', 'Processed 10,000+ customer records')."
        })

    if missing_skills and len(missing_skills) > 0:
        missing_top = missing_skills[:3]
        suggestions.append({
            "category": "Target Role Keywords",
            "issue": f"Missing key target keywords: {', '.join(missing_top)}.",
            "recommendation": f"Add project examples or coursework demonstrating hands-on familiarity with {', '.join(missing_top)}."
        })

    # Example rewrite transformation
    rewrite_example = {
        "weak_bullet": "Worked on machine learning projects and models.",
        "strong_bullet": "Engineered a machine learning classification model using Python and Scikit-learn, achieving 88% prediction accuracy on 10,000+ data samples."
    }

    return {
        "action_verb_count": len(found_action_verbs),
        "found_action_verbs": found_action_verbs,
        "metrics_count": len(lines_with_metrics),
        "lines_with_metrics": lines_with_metrics[:3],
        "suggestions": suggestions,
        "rewrite_example": rewrite_example
    }
