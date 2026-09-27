# AI-Based Resume Skill Gap Analyzer Using NLP

A complete, functional AI/ML college mini-project that analyzes resumes against target job descriptions using Natural Language Processing (NLP), TF-IDF vectorization, Cosine Similarity, and rule-based skill gap extraction.

---

## 📄 Abstract

Resume screening and skill matching are important tasks in modern recruitment. However, manually comparing resumes with job descriptions can be time-consuming and may overlook important skill gaps. This project presents an AI-based Resume Skill Gap Analyzer that uses Natural Language Processing techniques to compare a candidate's resume with a target job description. The system preprocesses textual information, extracts relevant technical skills, converts the documents into TF-IDF vectors, and calculates their similarity using cosine similarity. It identifies matching and missing skills and provides a simple skill-gap report and learning recommendations. The proposed system demonstrates how NLP and machine learning techniques can assist in automated resume analysis and provide an interpretable comparison between candidate skills and job requirements.

---

## 🎯 1. Project Objectives

1. **Automated Resume Matching**: Compute a semantic match score percentage between a resume and job description using TF-IDF and Cosine Similarity.
2. **Skill Extraction & Gap Analysis**: Extract domain-specific technical skills (Programming, AI/ML, Data, Cloud, Web, Tools) and classify them into **Matching**, **Missing**, and **Additional** skills.
3. **Category Breakdown & Visualization**: Provide a detailed tabular matrix and interactive charts comparing matched vs. missing skill distributions.
4. **Actionable Recommendations**: Generate rule-based self-learning suggestions based on identified missing skills.
5. **100% Local & Free Execution**: Run completely offline without relying on third-party paid APIs or OpenAI keys.

---

## 🛠️ 2. Technology Stack

- **Language**: Python 3.11+
- **User Interface**: Streamlit
- **NLP & Machine Learning**: `scikit-learn` (`TfidfVectorizer`, `cosine_similarity`), `NLTK`
- **PDF Extraction**: `pypdf`
- **Data Manipulation & Visualization**: `pandas`, `NumPy`, `plotly`

---

## 📁 3. Project Structure

```
resume-skill-gap-analyzer/
│
├── app.py                      # Main Streamlit dashboard application
├── requirements.txt            # Python dependencies
├── README.md                   # Complete documentation & Viva Q&A
│
├── src/                        # Core NLP & ML modules
│   ├── __init__.py             # Package initializer
│   ├── text_preprocessing.py   # PDF reader & text cleaning pipeline
│   ├── similarity.py           # TF-IDF vectorization & Cosine Similarity score
│   ├── skill_extractor.py      # Regex word-boundary skill extraction & gap analysis
│   └── recommendations.py      # Rule-based learning recommendation engine
│
├── data/                       # Dataset & Knowledge base
│   └── skills.json             # Categorized skill dictionary (60+ technical skills)
│
└── sample/                     # Sample demonstration data
    ├── sample_resume.txt       # Pre-filled sample resume text
    └── sample_job_description.txt # Pre-filled sample job description text
```

---

## ⚙️ 4. AI & NLP Methodology

### A. Text Preprocessing
1. **Lowercase Conversion**: Standardizes text to eliminate case sensitivity.
2. **Punctuation Filtering**: Removes non-alphanumeric noise while preserving programming symbols (e.g. `C++`, `C#`).
3. **Whitespace Normalization**: Strips duplicate line breaks, tabs, and multiple spaces.
4. **Stop-Word Removal**: Filters common English stop-words (`and`, `the`, `is`, `with`) using `NLTK`.

### B. TF-IDF Vectorization
Term Frequency-Inverse Document Frequency (TF-IDF) converts textual documents into numerical feature vectors.

$$\text{TF}(t, d) = \frac{\text{Count of term } t \text{ in document } d}{\text{Total words in document } d}$$

$$\text{IDF}(t, D) = \log\left(\frac{N}{1 + |\{d \in D : t \in d\}|}\right)$$

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

- **Why TF-IDF was selected**: Unlike simple word counts (Bag-of-Words), TF-IDF assigns higher numerical weight to informative technical keywords (e.g., *TensorFlow*, *Kubernetes*) while discounting generic words that appear frequently across all documents.

### C. Cosine Similarity
Cosine Similarity computes the dot product of normalized TF-IDF feature vectors to measure the angle between the resume vector $\vec{A}$ and job description vector $\vec{B}$:

$$\text{Cosine Similarity}(\vec{A}, \vec{B}) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$

- **Why Cosine Similarity was selected**: Cosine similarity measures vector orientation rather than magnitude. This ensures that long resumes are not unfairly advantaged or biased compared to concise resumes.
- **Match Score Formula**: 

$$\text{Resume-Job Match Score (\%)} = \text{Cosine Similarity} \times 100$$

### D. Skill Extraction Engine
Skills are matched using negative lookbehind and lookahead regex assertions (`(?<![\w])skill(?![\w\+\#])`). This prevents false positive substring matches:
- Matches standalone `C` without matching `CSS`, `Cloud`, or `C++`.
- Matches `R` without matching `React` or `Rust`.

---

## 🚀 5. Installation & Run Guide

### Step 1: Clone / Download Repository
Navigate to the project root directory:
```bash
cd "d:/AI MiniProject"
```

### Step 2: Install Requirements
Install all required Python packages:
```bash
pip install -r requirements.txt
```

### Step 3: Launch Streamlit Dashboard
Run the Streamlit application:
```bash
streamlit run app.py
```
The browser will automatically open at `http://localhost:8501`.

---

## 🧪 6. Sample Input & Expected Results

Click the **"📋 Load Sample Demo Data"** button in the sidebar to test:
- **Sample Resume**: Experience in Python, SQL, Pandas, NumPy, Machine Learning, HTML, CSS, Git, GitHub.
- **Sample Job Description**: Junior AI/ML Engineer requiring Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, SQL, AWS, Docker, Git.

### Analysis Output:
- **Match Score**: ~65% - 75%
- **Matching Skills**: Python, SQL, Machine Learning, Git
- **Missing Skills**: Deep Learning, TensorFlow, PyTorch, AWS, Docker
- **Additional Skills**: HTML, CSS, Pandas, NumPy, GitHub
- **Recommendations**: Targeted learning cards for AWS, Docker, TensorFlow, and PyTorch.

---

## 🌟 7. Advantages & Limitations

### Advantages:
1. **100% Local & Privacy Preserved**: No sensitive resume data sent to external cloud APIs.
2. **Fast & Deterministic**: Immediate computation without waiting for LLM API latency.
3. **Interpretable**: Clearly explains match percentages and explicit skill gaps.

### Limitations:
1. **Dictionary Dependent**: Skills outside `skills.json` are evaluated via general TF-IDF similarity but not categorized in the skill matrix.
2. **Syntactic Matching**: Synonyms not in the dictionary may not be automatically resolved (can be improved with word embeddings like Word2Vec/BERT).

---

## 🔮 8. Future Scope

- Integrate pre-trained word embeddings (e.g. Word2Vec, GloVe, Sentence-BERT) for contextual semantic search.
- Add automatic resume parsing for work history duration and experience level calculation.
- Support multi-page batch analysis for HR recruiters.

---

## 🎓 9. Viva Questions and Answers

### Q1: What is the objective of this project?
**Answer**: The objective is to build an automated NLP application that compares a candidate's resume against a target job description, calculates a similarity match percentage, identifies skill gaps (matching, missing, and additional skills), and provides learning recommendations.

### Q2: What is Natural Language Processing (NLP)?
**Answer**: NLP is a branch of Artificial Intelligence (AI) that enables computers to understand, interpret, preprocess, and analyze human textual language.

### Q3: What is TF-IDF?
**Answer**: TF-IDF stands for Term Frequency-Inverse Document Frequency. It is a statistical technique that quantifies the importance of words in a document relative to a corpus. Term Frequency (TF) measures how often a word appears in a document, while Inverse Document Frequency (IDF) penalizes words that appear commonly across all documents.

### Q4: Why did you use TF-IDF vectorization instead of simple word counts?
**Answer**: Simple word counts (Bag-of-Words) give equal weight to common words like "experience", "candidate", or "work". TF-IDF down-weights common non-technical words and highlights domain-specific technical terms (like "PyTorch", "Kubernetes", "PostgreSQL").

### Q5: What is Cosine Similarity?
**Answer**: Cosine Similarity measures the cosine of the angle between two non-zero numerical vectors in a multi-dimensional space. A cosine similarity of 1 means identical orientation (100% match), while 0 means orthogonal/completely dissimilar text vectors.

### Q6: Why did you use Cosine Similarity instead of Euclidean Distance?
**Answer**: Euclidean distance measures the magnitude difference between vectors, meaning longer resumes with more total words would be penalised or separated from shorter job descriptions. Cosine similarity evaluates document orientation/direction regardless of word length.

### Q7: What is skill extraction and how is it implemented?
**Answer**: Skill extraction identifies technical domain skills within raw text. In this project, it is implemented using a categorized dictionary (`skills.json`) and regex word-boundary pattern matching (`\b skill \b`) to accurately detect skills without false partial word matches.

### Q8: How is the final Match Percentage calculated?
**Answer**: The raw text from both the resume and job description undergoes cleaning and stop-word removal. `TfidfVectorizer` converts both texts into TF-IDF vectors, and `cosine_similarity` computes the vector dot product. The decimal score (e.g., 0.724) is multiplied by 100 to yield `72.4%`.

### Q9: What text preprocessing steps are performed?
**Answer**: 
1. Lowercasing.
2. Removal of non-essential punctuation (preserving symbols like `+` in `C++` and `#` in `C#`).
3. Normalization of whitespace.
4. Stop-word removal using `NLTK`.

### Q10: What are the main limitations of this system?
**Answer**: 
1. Relying on a fixed skills dictionary for categorical skill extraction.
2. Lack of semantic understanding for unseen abbreviations or complex sentence contexts without transformer fine-tuning.

### Q11: Is this project supervised or unsupervised machine learning?
**Answer**: This project uses **unsupervised NLP techniques**. TF-IDF vectorization and Cosine Similarity do not require labeled training dataset target labels (ground truth labels) or supervised model training.

### Q12: Why didn't you use a Deep Learning / LLM model like GPT or BERT?
**Answer**: 
1. **Resource Constraints & Privacy**: Deep learning models require high memory/GPU compute or external paid cloud APIs.
2. **Determinism & Speed**: TF-IDF + Cosine Similarity provides instant, 100% reproducible, and transparent results suitable for local execution.

### Q13: How can this project be improved in the future?
**Answer**: By integrating Sentence-BERT embeddings for deep semantic matching, automated experience year extraction from resume headers, and fine-tuning an NER (Named Entity Recognition) model (e.g. spaCy NER) for dynamic skill extraction.

### Q14: What happens if a resume contains a skill not listed in `skills.json`?
**Answer**: Unlisted technical skills are still captured in the overall TF-IDF vectorization and contribute to the Cosine Similarity match percentage, but they will not be categorized under the predefined skill gap summary table.

### Q15: How is this system different from a simple string keyword search?
**Answer**: Simple keyword search only looks for exact string occurrences. This system performs NLTK stop-word removal, TF-IDF statistical weighting (giving higher importance to rare technical terms), Cosine Similarity vector angle calculation, regex boundary assertions, category breakdown analysis, and rule-based recommendation mapping.
