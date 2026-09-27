import os
import json
import re
from typing import Dict, List, Set


def load_skills_dictionary(filepath: str = None) -> Dict[str, List[str]]:
    """
    Loads categorized technical skills dictionary from JSON.
    """
    if filepath is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, 'data', 'skills.json')

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Skills dictionary file not found at: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        skills_dict = json.load(f)

    return skills_dict


def _build_skill_regex(skill: str) -> re.Pattern:
    """
    Builds a case-insensitive, word-boundary-aware regex pattern for a skill.
    Correctly handles special characters like C++, C#, .NET while preventing
    partial word matches (e.g. 'C' matching 'CSS' or 'Cloud').
    """
    escaped_skill = re.escape(skill)

    # For skills ending/starting with non-word chars (+, #), standard \b fails
    # Use negative lookbehind and lookahead assertions for accurate boundary checking
    pattern = rf'(?i)(?<![\w]){escaped_skill}(?![\w\+\#])'
    return re.compile(pattern)


def extract_skills_from_text(text: str, skills_dict: Dict[str, List[str]] = None) -> Dict[str, List[str]]:
    """
    Extracts skills present in text, returning a dictionary mapping category -> list of extracted canonical skills.
    Also returns a set of all extracted canonical skills.
    """
    if skills_dict is None:
        skills_dict = load_skills_dictionary()

    if not text or not text.strip():
        return {}

    found_by_category = {}

    for category, skill_list in skills_dict.items():
        found_in_cat = []
        for skill in skill_list:
            pattern = _build_skill_regex(skill)
            if pattern.search(text):
                found_in_cat.append(skill)
        found_by_category[category] = found_in_cat

    return found_by_category


def analyze_skill_gaps(resume_text: str, job_text: str, skills_dict: Dict[str, List[str]] = None) -> dict:
    """
    Performs complete skill gap comparison between resume and job description.
    
    Returns:
        dict: {
            "resume_skills": list,
            "job_skills": list,
            "matching_skills": list,
            "missing_skills": list,
            "additional_skills": list,
            "category_breakdown": list of dicts
        }
    """
    if skills_dict is None:
        skills_dict = load_skills_dictionary()

    resume_cat_skills = extract_skills_from_text(resume_text, skills_dict)
    job_cat_skills = extract_skills_from_text(job_text, skills_dict)

    all_resume_skills = set(s for cat_skills in resume_cat_skills.values() for s in cat_skills)
    all_job_skills = set(s for cat_skills in job_cat_skills.values() for s in cat_skills)

    matching_skills = sorted(list(all_resume_skills.intersection(all_job_skills)))
    missing_skills = sorted(list(all_job_skills.difference(all_resume_skills)))
    additional_skills = sorted(list(all_resume_skills.difference(all_job_skills)))

    # Category breakdown table generation
    category_breakdown = []
    for category in skills_dict.keys():
        res_cat = set(resume_cat_skills.get(category, []))
        job_cat = set(job_cat_skills.get(category, []))

        matched_cat = sorted(list(res_cat.intersection(job_cat)))
        missing_cat = sorted(list(job_cat.difference(res_cat)))
        additional_cat = sorted(list(res_cat.difference(job_cat)))

        category_breakdown.append({
            "Category": category,
            "Job Required": len(job_cat),
            "Matched": len(matched_cat),
            "Missing": len(missing_cat),
            "Additional": len(additional_cat),
            "Matched Skills": ", ".join(matched_cat) if matched_cat else "-",
            "Missing Skills": ", ".join(missing_cat) if missing_cat else "-"
        })

    return {
        "resume_skills": sorted(list(all_resume_skills)),
        "job_skills": sorted(list(all_job_skills)),
        "matching_skills": matching_skills,
        "missing_skills": missing_skills,
        "additional_skills": additional_skills,
        "category_breakdown": category_breakdown
    }
