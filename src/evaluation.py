import time
from typing import List, Dict

# Ground truth benchmark evaluation dataset (5 annotated test pairs)
EVALUATION_DATASET = [
    {
        "id": "Pair_1",
        "resume": "Python, SQL, Pandas, NumPy, Machine Learning, Scikit-learn, Git, GitHub",
        "job": "Python, Machine Learning, Deep Learning, TensorFlow, SQL, AWS, Git",
        "ground_truth_matching": ["Python", "Machine Learning", "SQL", "Git"],
        "ground_truth_missing": ["Deep Learning", "TensorFlow", "AWS"]
    },
    {
        "id": "Pair_2",
        "resume": "Java, Spring Boot, MySQL, REST API, HTML, CSS, JavaScript, Git",
        "job": "Java, Spring Boot, Microservices, Docker, Kubernetes, PostgreSQL, Git",
        "ground_truth_matching": ["Java", "Spring Boot", "Git"],
        "ground_truth_missing": ["Microservices", "Docker", "Kubernetes", "PostgreSQL"]
    },
    {
        "id": "Pair_3",
        "resume": "Python, Data Analysis, SQL, Power BI, Tableau, Excel, Communication",
        "job": "Python, SQL, Pandas, NumPy, Data Analysis, Power BI, Tableau",
        "ground_truth_matching": ["Python", "SQL", "Data Analysis", "Power BI", "Tableau"],
        "ground_truth_missing": ["Pandas", "NumPy"]
    },
    {
        "id": "Pair_4",
        "resume": "AWS, Docker, Linux, Bash, Python, Jenkins, Terraform, CI/CD, Git",
        "job": "AWS, Azure, Docker, Kubernetes, CI/CD, Linux, Python, Git",
        "ground_truth_matching": ["AWS", "Docker", "CI/CD", "Linux", "Python", "Git"],
        "ground_truth_missing": ["Azure", "Kubernetes"]
    },
    {
        "id": "Pair_5",
        "resume": "React, JavaScript, HTML, CSS, Node.js, Express, MongoDB, REST API",
        "job": "React, TypeScript, Node.js, GraphQL, MongoDB, Docker, Git",
        "ground_truth_matching": ["React", "Node.js", "MongoDB"],
        "ground_truth_missing": ["TypeScript", "GraphQL", "Docker", "Git"]
    }
]


def run_academic_evaluation_benchmark() -> dict:
    """
    Evaluates the NLP semantic pipeline on a reproducible ground-truth dataset of test pairs.
    Calculates real, empirical Precision, Recall, F1-Score, and Latency.
    """
    from src.semantic_matcher import perform_semantic_skill_matching

    total_tp = 0
    total_fp = 0
    total_fn = 0
    latencies = []

    pair_results = []

    for test_pair in EVALUATION_DATASET:
        start_t = time.time()
        res = perform_semantic_skill_matching(test_pair["resume"], test_pair["job"])
        end_t = time.time()

        latency_ms = (end_t - start_t) * 1000
        latencies.append(latency_ms)

        predicted_matching = set([m["skill"] for m in res["strong_matches"]] + [m["skill"] for m in res["partial_matches"]])
        ground_matching = set(test_pair["ground_truth_matching"])

        tp = len(predicted_matching.intersection(ground_matching))
        fp = len(predicted_matching.difference(ground_matching))
        fn = len(ground_matching.difference(predicted_matching))

        total_tp += tp
        total_fp += fp
        total_fn += fn

        pair_results.append({
            "id": test_pair["id"],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "latency_ms": round(latency_ms, 2)
        })

    precision = round((total_tp / (total_tp + total_fp)) * 100, 1) if (total_tp + total_fp) > 0 else 0.0
    recall = round((total_tp / (total_tp + total_fn)) * 100, 1) if (total_tp + total_fn) > 0 else 0.0
    f1_score = round((2 * precision * recall / (precision + recall)), 1) if (precision + recall) > 0 else 0.0
    avg_latency = round(sum(latencies) / len(latencies), 1)

    return {
        "dataset_size": len(EVALUATION_DATASET),
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "avg_latency_ms": avg_latency,
        "total_true_positives": total_tp,
        "total_false_positives": total_fp,
        "total_false_negatives": total_fn,
        "pair_results": pair_results
    }
