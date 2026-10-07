import re
from typing import List, Dict

# Core domain skills that carry high structural weight
CORE_DOMAIN_SKILLS = {
    "Python", "Java", "C++", "SQL", "Machine Learning", "Deep Learning",
    "Data Analysis", "React", "Node.js", "AWS", "Docker"
}


def classify_skill_priorities(missing_skills: List[dict], job_text: str) -> List[dict]:
    """
    Classifies missing skills into HIGH, MEDIUM, or LOW priority with explicit reasoning.
    
    Priority Logic:
    - HIGH: Skill appears 2+ times in JD OR is a core foundational skill (e.g. Python, ML, SQL).
    - MEDIUM: Secondary frameworks/cloud tools required by the job.
    - LOW: Auxiliary tools or utility packages.
    """
    job_text_lower = job_text.lower()
    prioritized_list = []

    for item in missing_skills:
        skill_name = item.get("skill") if isinstance(item, dict) else str(item)
        skill_lower = skill_name.lower()

        # Count frequency of skill in job description
        occurrences = len(re.findall(rf'\b{re.escape(skill_lower)}\b', job_text_lower))

        if skill_name in CORE_DOMAIN_SKILLS or occurrences >= 2:
            priority = "HIGH"
            reason = f"Essential core skill for the role; mentioned {occurrences} time(s) in job description."
        elif occurrences == 1 or any(kw in skill_lower for kw in ["cloud", "devops", "database", "ai", "web"]):
            priority = "MEDIUM"
            reason = "Required secondary framework/tool mentioned in role requirements."
        else:
            priority = "LOW"
            reason = "Auxiliary tool or nice-to-have supporting technology."

        prioritized_list.append({
            "skill": skill_name,
            "priority": priority,
            "reason": reason,
            "occurrences": occurrences
        })

    # Sort priority order: HIGH -> MEDIUM -> LOW
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    prioritized_list.sort(key=lambda x: priority_order.get(x["priority"], 3))

    return prioritized_list
