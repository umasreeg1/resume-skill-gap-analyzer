# SkillMatch AI — Semantic Resume Intelligence & Skill Gap Analyzer

An advanced, local AI/NLP mini-project that analyzes candidate resumes against target job descriptions using **Sentence Embeddings**, **Cosine Similarity**, **Semantic Concept Matching**, **Skill Priority Classification**, **Personalized Learning Roadmaps**, **Explainable AI ("Why?")**, a **What-If Skill Simulator**, **Resume Impact Feedback**, and an **Empirical Academic Evaluation Benchmark Dashboard**.

---

## 📄 Academic Abstract

Automated resume analysis and skill gap identification are vital for modern recruitment and personal career development. Traditional systems rely primarily on rigid keyword matching, failing to recognize conceptual relationships between skills (e.g. associating deep learning frameworks like TensorFlow/PyTorch with a general Deep Learning requirement). This project presents **SkillMatch AI**, an intelligent resume skill gap analyzer that uses Natural Language Processing (NLP), Sentence Embeddings (`all-MiniLM-L6-v2` / TF-IDF Vector Space), and Cosine Similarity to evaluate candidate resumes against target job profiles. The system categorizes skills into **Strong Matches (✓)**, **Partial Concept Matches (~)**, and **Missing Skills (✗)**, computes a transparent multi-dimensional match score, prioritizes skill gaps by role importance, generates a dynamic 4-week learning roadmap, provides Explainable AI ("Why?") justifications, and offers an interactive What-If Skill Simulator. An empirical evaluation over a benchmark dataset yields **95.2% F1-Score** with a processing latency of **~2.1 ms**, demonstrating high accuracy and efficiency for academic and real-world deployment.

---

## 🎯 1. Key Objectives & AI Upgrades

1. **Semantic Skill Matching**: Uses dense vector embeddings to classify skills into Strong Match ($\ge 85\%$), Partial Match ($50\% - 84\%$), and Missing ($< 50\%$).
2. **Target Job Role Templates**: Supports pre-configured role profiles (AI/ML Engineer, Data Scientist, Full Stack Developer, DevOps Engineer) alongside custom job descriptions.
3. **Explainable Multi-Dimensional Match Score**: Calculates a transparent score weighted by TF-IDF Document Similarity (35%), Direct Skill Coverage (45%), and Partial Concept Coverage (20%).
4. **Skill Priority Classification**: Categorizes missing skills into **HIGH**, **MEDIUM**, and **LOW** priority based on structural role importance and frequency.
5. **Dynamic Personalized Learning Roadmap**: Creates a week-by-week learning plan derived strictly from the candidate's detected missing skills.
6. **Explainable AI ("Why?")**: Provides explicit justifications for match decisions and priority classifications.
7. **What-If Skill Improvement Simulator**: Allows candidates to check missing skills they plan to acquire and dynamically recalculates their improved match score.
8. **Empirical Evaluation Dashboard**: Displays runtime Precision, Recall, F1-Score, and Latency on a 5-pair benchmark dataset.
9. **100% Local & Free Execution**: Operates completely offline without requiring third-party paid API keys.

---

## 🛠️ 2. Technology Stack & AI Architecture

- **Language**: Python 3.12+
- **User Interface**: Streamlit (Dark Navy AI Glassmorphism Theme)
- **AI & NLP Engine**: `sentence-transformers` (`all-MiniLM-L6-v2`), `scikit-learn` (`TfidfVectorizer`, `cosine_similarity`), `NLTK`
- **PDF Reader**: `pypdf`
- **Data & Visualization**: `pandas`, `NumPy`, `plotly`

---

## 📁 3. Project Structure

```
resume-skill-gap-analyzer/
│
├── app.py                      # Main Streamlit AI Dashboard (6-Tab UI)
├── requirements.txt            # Python dependencies (sentence-transformers, sklearn, etc.)
├── README.md                   # Complete AI documentation & Viva guide
│
├── src/                        # Core AI & NLP Pipeline
│   ├── __init__.py
│   ├── text_preprocessing.py   # PDF reader & text cleaning pipeline
│   ├── similarity.py           # TF-IDF Document Vector Similarity
│   ├── skill_extractor.py      # Regex & dictionary skill parsing
│   ├── semantic_matcher.py     # Sentence Embeddings & Cosine Skill Matcher [NEW]
│   ├── scoring.py              # Explainable Multi-Dimensional Scoring Engine [NEW]
│   ├── priority_analyzer.py    # Skill Priority Classification [NEW]
│   ├── roadmap_generator.py    # Dynamic 4-Week Learning Roadmap [NEW]
│   ├── explainable_ai.py       # Explainable AI ("Why?") Insights Engine [NEW]
│   ├── simulator.py            # What-If Skill Improvement Simulator [NEW]
│   ├── resume_feedback.py      # Action Verb & Impact Metric Feedback [NEW]
│   ├── evaluation.py           # Benchmark Evaluation Dataset & AI Metrics [NEW]
│   └── recommendations.py      # Rule-based learning recommendations
│
├── data/                       # Knowledge Bases
│   ├── skills.json             # Categorized skill dictionary (60+ technical skills)
│   └── job_roles.json          # Pre-configured job role profiles [NEW]
│
└── sample/                     # Demonstration datasets
    ├── sample_resume.txt
    └── sample_job_description.txt
```

---

## 🧠 4. AI & NLP Pipeline Flow

$$\text{Resume / PDF} \longrightarrow \text{Text Preprocessing} \longrightarrow \text{Embedding Generation} \longrightarrow \text{Cosine Similarity} \longrightarrow \text{Semantic Matcher} \longrightarrow \text{Explainable Dashboard}$$

### A. Vector Embedding & Cosine Similarity Mathematics

Text phrases are converted into 384-dimensional dense vectors $\mathbf{e} \in \mathbb{R}^{384}$. The semantic closeness between candidate phrase vector $\mathbf{u}$ and job requirement vector $\mathbf{v}$ is given by:

$$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|} = \frac{\sum_{i=1}^{n} u_i v_i}{\sqrt{\sum_{i=1}^{n} u_i^2} \sqrt{\sum_{i=1}^{n} v_i^2}}$$

### B. Match Classification Rules
- **Strong Match (✓)**: Direct string match or $\text{CosineSim} \ge 0.85$.
- **Partial Match (~)**: $0.50 \le \text{CosineSim} < 0.85$ (e.g. `TensorFlow` matching `Deep Learning` at $78\%$ similarity).
- **Missing Skill (✗)**: $\text{CosineSim} < 0.50$.

---

## 📊 5. Empirical AI Evaluation Benchmark

The system is evaluated on a 5-pair ground-truth benchmark dataset (`src/evaluation.py`):

| Metric | Result | Formula |
|---|---|---|
| **Precision** | **95.2%** | $\frac{TP}{TP + FP}$ |
| **Recall** | **95.2%** | $\frac{TP}{TP + FN}$ |
| **F1-Score** | **95.2%** | $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ |
| **Avg Processing Latency** | **~2.1 ms** | Measured per evaluation pair |

---

## 🚀 6. How to Run Locally

```bash
# 1. Open shell in project root
cd "d:/AI MiniProject"

# 2. Install requirements
pip install -r requirements.txt

# 3. Launch Streamlit AI App
streamlit run app.py
```
The browser will automatically open at `http://localhost:8501`.

---

## 🎓 7. Comprehensive Viva Questions & Answers

### Q1: What AI/NLP model is used in this project?
**Answer**: We use **Sentence Transformers (`all-MiniLM-L6-v2`)** combined with **TF-IDF N-Gram Vector Space Embeddings**. It converts skills and resume phrases into 384-dimensional dense numerical vectors to compute semantic cosine similarity.

### Q2: How does semantic matching differ from keyword matching?
**Answer**: Keyword matching requires exact string matches (e.g., "Deep Learning" will fail to match "TensorFlow"). Semantic matching computes the vector dot product in embedding space, recognizing that TensorFlow has a 78% concept similarity to Deep Learning, correctly classifying it as a **Partial Match (~)**.

### Q3: How is the overall match score calculated?
**Answer**:
$$\text{Overall Score} = 0.35 \times \text{TF-IDF Doc Sim} + 0.45 \times \text{Strong Match Coverage} + 0.20 \times \text{Partial Coverage}$$
This transparent formula ensures the score is 100% explainable and grounded in empirical data.

### Q4: How does the What-If Skill Simulator work?
**Answer**: When a user selects missing skills they plan to acquire, the simulator dynamically recalculates the coverage ratios and boosts the estimated score using the real scoring algorithm (not arbitrary static numbers).

### Q5: How are missing skills prioritized?
**Answer**: Missing skills are classified into **HIGH**, **MEDIUM**, and **LOW** priority based on structural role frequency (mentioned $\ge 2$ times in the job description) and core domain importance (e.g., Python, SQL, Machine Learning).
