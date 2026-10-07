from typing import List, Dict


def generate_personalized_roadmap(prioritized_skills: List[dict]) -> List[dict]:
    """
    Generates a week-by-week learning roadmap tailored to candidate's actual missing skills.
    """
    if not prioritized_skills:
        return [{
            "week": "Week 1 - 4",
            "title": "Skill Maintenance & Interview Prep",
            "focus_skills": ["Current Resume Skills"],
            "tasks": "Your resume already covers all required technical skills for this role! Focus on system design mock interviews and coding practice."
        }]

    high_skills = [s["skill"] for s in prioritized_skills if s["priority"] == "HIGH"]
    med_skills = [s["skill"] for s in prioritized_skills if s["priority"] == "MEDIUM"]
    low_skills = [s["skill"] for s in prioritized_skills if s["priority"] == "LOW"]

    all_missing_sorted = high_skills + med_skills + low_skills

    roadmap = []

    # Week 1: Foundational High Priority Skill
    w1_skills = all_missing_sorted[:2] if len(all_missing_sorted) >= 2 else all_missing_sorted[:1]
    roadmap.append({
        "week": "Week 1",
        "title": f"Core Foundations: {', '.join(w1_skills)}",
        "focus_skills": w1_skills,
        "tasks": f"Master core concepts, syntax, and foundational tutorials in {', '.join(w1_skills)}."
    })

    # Week 2: High/Medium Priority Applied Projects
    w2_skills = all_missing_sorted[2:4] if len(all_missing_sorted) >= 4 else all_missing_sorted[1:2]
    if not w2_skills:
        w2_skills = w1_skills
    roadmap.append({
        "week": "Week 2",
        "title": f"Applied Practice: {', '.join(w2_skills)}",
        "focus_skills": w2_skills,
        "tasks": f"Build practical hands-on mini-projects incorporating {', '.join(w2_skills)}."
    })

    # Week 3: Medium/Low Priority Frameworks & Tools
    w3_skills = all_missing_sorted[4:6] if len(all_missing_sorted) >= 6 else all_missing_sorted[2:3]
    if not w3_skills:
        w3_skills = w2_skills
    roadmap.append({
        "week": "Week 3",
        "title": f"Infrastructure & Advanced Tools: {', '.join(w3_skills)}",
        "focus_skills": w3_skills,
        "tasks": f"Explore containerization, deployment, or advanced querying with {', '.join(w3_skills)}."
    })

    # Week 4: Capstone Integration & Portfolio Project
    capstone_skills = all_missing_sorted[:4]
    roadmap.append({
        "week": "Week 4",
        "title": "Capstone Portfolio Integration",
        "focus_skills": capstone_skills,
        "tasks": f"Integrate {', '.join(capstone_skills)} into an end-to-end GitHub portfolio project and add it to your resume."
    })

    return roadmap
