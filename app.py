import os
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.text_preprocessing import extract_text_from_pdf, clean_text
from src.similarity import calculate_similarity
from src.skill_extractor import analyze_skill_gaps, load_skills_dictionary
from src.recommendations import generate_recommendations

# Page configuration
st.set_page_config(
    page_title="AI Resume Skill Gap Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern visual design
st.markdown("""
<style>
    /* Main Theme Variables */
    :root {
        --primary-color: #2E5BFF;
        --success-color: #10B981;
        --danger-color: #EF4444;
        --info-color: #3B82F6;
        --background-card: #F8FAFC;
    }
    
    /* Title and Header styling */
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    
    /* Cards and Containers */
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E293B;
    }
    .metric-label {
        font-size: 0.9rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Skill Tag Chips */
    .chip {
        display: inline-block;
        padding: 5px 12px;
        margin: 4px;
        border-radius: 20px;
        font-size: 0.88rem;
        font-weight: 600;
        line-height: 1.4;
    }
    .chip-match {
        background-color: #DEF7EC;
        color: #03543F;
        border: 1px solid #84E1BC;
    }
    .chip-missing {
        background-color: #FDE8E8;
        color: #9B1C1C;
        border: 1px solid #F8B4B4;
    }
    .chip-additional {
        background-color: #E1EFFE;
        color: #1E429F;
        border: 1px solid #A4CAFE;
    }
    
    /* Recommendation Card */
    .recommendation-card {
        background-color: #FFFFFF;
        border-left: 4px solid #3B82F6;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 10px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .rec-skill {
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 4px;
    }
    .rec-text {
        font-size: 0.92rem;
        color: #475569;
    }
</style>
""", unsafe_allow_html=True)


def load_sample_files():
    """Helper to load predefined sample resume and job description text."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sample_res_path = os.path.join(base_dir, 'sample', 'sample_resume.txt')
    sample_job_path = os.path.join(base_dir, 'sample', 'sample_job_description.txt')
    
    res_text = ""
    job_text = ""
    if os.path.exists(sample_res_path):
        with open(sample_res_path, 'r', encoding='utf-8') as f:
            res_text = f.read()
    if os.path.exists(sample_job_path):
        with open(sample_job_path, 'r', encoding='utf-8') as f:
            job_text = f.read()
            
    return res_text, job_text


def render_chips(skills_list: list, chip_class: str) -> str:
    """Generates HTML tag chips for a list of skills."""
    if not skills_list:
        return "<span style='color: #94A3B8; font-style: italic;'>None detected</span>"
    html_chips = "".join([f'<span class="chip {chip_class}">{skill}</span>' for skill in skills_list])
    return html_chips


def main():
    # Sidebar Info & Instructions
    st.sidebar.title("ℹ️ About Project")
    st.sidebar.info(
        "**AI-Based Resume Skill Gap Analyzer**\n\n"
        "This system uses **NLP (TF-IDF Vectorization)** and **Cosine Similarity** "
        "to calculate the semantic match between your resume and a target job description. "
        "It also performs rule-based skill extraction to highlight matched, missing, and additional technical skills."
    )
    
    st.sidebar.title("🚀 Quick Actions")
    load_sample = st.sidebar.button("📋 Load Sample Demo Data", use_container_width=True)

    # Main Page Header
    st.markdown('<div class="main-title">🎯 AI Resume Skill Gap Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Analyze your resume against a target job description using NLP & Machine Learning</div>', unsafe_allow_html=True)

    # Load sample data if requested via sidebar or state
    default_resume_text = ""
    default_job_text = ""
    
    if load_sample:
        default_resume_text, default_job_text = load_sample_files()
        st.toast("Sample demo data loaded successfully!", icon="✅")

    # Inputs Section
    st.markdown("### 📝 Step 1: Provide Input Documents")
    col1, col2 = st.columns(2)

    resume_input_text = ""

    with col1:
        st.subheader("1. Resume Input")
        resume_mode = st.radio("Choose input method for Resume:", ["Upload PDF", "Paste Raw Text"], horizontal=True)

        if resume_mode == "Upload PDF":
            uploaded_file = st.file_uploader("Upload Resume PDF", type=["pdf"])
            if uploaded_file is not None:
                try:
                    resume_input_text = extract_text_from_pdf(uploaded_file)
                    st.success(f"PDF processed successfully ({len(resume_input_text.split())} words extracted)")
                except Exception as e:
                    st.error(f"Error parsing PDF: {str(e)}")
            elif default_resume_text:
                st.info("Using sample resume text (from Load Sample Demo)")
                resume_input_text = default_resume_text
        else:
            resume_input_text = st.text_area(
                "Paste your resume text here:",
                value=default_resume_text,
                height=250,
                placeholder="Paste the text content of your resume..."
            )

    with col2:
        st.subheader("2. Target Job Description")
        job_input_text = st.text_area(
            "Paste target Job Description (JD) here:",
            value=default_job_text,
            height=295,
            placeholder="Paste the job description or role requirements here..."
        )

    st.markdown("---")
    analyze_button = st.button("⚡ Analyze Resume & Extract Skill Gaps", type="primary", use_container_width=True)

    # Run Analysis when button is clicked or sample data is loaded
    if analyze_button or (load_sample and resume_input_text and job_input_text):
        if not resume_input_text or not resume_input_text.strip():
            st.error("⚠️ Please upload or paste a valid Resume before running analysis.")
            return

        if not job_input_text or not job_input_text.strip():
            st.error("⚠️ Please paste a valid Target Job Description before running analysis.")
            return

        with st.spinner("🔍 Performing NLP text preprocessing, TF-IDF vectorization, and skill extraction..."):
            try:
                # 1. Similarity Engine (TF-IDF + Cosine Similarity)
                similarity_result = calculate_similarity(resume_input_text, job_input_text)
                match_score = similarity_result["match_percentage"]
                cosine_sim = similarity_result["cosine_sim"]
                top_words = similarity_result["top_overlapping_words"]

                # 2. Skill Gap Analysis
                gap_result = analyze_skill_gaps(resume_input_text, job_input_text)
                matching_skills = gap_result["matching_skills"]
                missing_skills = gap_result["missing_skills"]
                additional_skills = gap_result["additional_skills"]
                category_breakdown = gap_result["category_breakdown"]

                # 3. Rule-based Recommendations
                recommendations = generate_recommendations(missing_skills)

            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                return

        # Display Results Dashboard
        st.markdown("## 📊 Step 2: Analysis & Skill Gap Dashboard")
        
        # Row 1: Match Score & Summary Metrics
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        
        with m_col1:
            st.metric(label="🎯 Resume-Job Match Score", value=f"{match_score}%", delta=f"Cosine Sim: {cosine_sim}")
        with m_col2:
            st.metric(label="✅ Matching Skills", value=len(matching_skills))
        with m_col3:
            st.metric(label="❌ Missing Skills", value=len(missing_skills))
        with m_col4:
            st.metric(label="➕ Additional Skills", value=len(additional_skills))

        st.markdown("---")

        # Row 2: Skill Chips (Matching, Missing, Additional)
        sc1, sc2 = st.columns(2)

        with sc1:
            st.subheader("✅ Matching Skills Found")
            st.markdown(render_chips(matching_skills, "chip-match"), unsafe_allow_html=True)
            
            st.subheader("➕ Additional Skills in Resume")
            st.markdown(render_chips(additional_skills, "chip-additional"), unsafe_allow_html=True)

        with sc2:
            st.subheader("❌ Missing Skills Required by Job")
            st.markdown(render_chips(missing_skills, "chip-missing"), unsafe_allow_html=True)

        st.markdown("---")

        # Row 3: Category Breakdown & Visualizations
        c1, c2 = st.columns([1.2, 1])

        with c1:
            st.subheader("📋 Skill Category Breakdown")
            df_cat = pd.DataFrame(category_breakdown)
            st.dataframe(
                df_cat[["Category", "Job Required", "Matched", "Missing", "Additional"]],
                hide_index=True,
                use_container_width=True
            )

        with c2:
            st.subheader("📈 Matched vs Missing Skills")
            if len(matching_skills) == 0 and len(missing_skills) == 0:
                st.info("No domain skills detected for chart generation.")
            else:
                # Plotly Donut Chart
                labels = ['Matching Skills', 'Missing Skills']
                values = [len(matching_skills), len(missing_skills)]
                colors = ['#10B981', '#EF4444']

                fig = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    hole=.4,
                    marker=dict(colors=colors),
                    textinfo='value+percent',
                    hoverinfo='label+value'
                )])
                fig.update_layout(
                    margin=dict(t=20, b=20, l=20, r=20),
                    height=280,
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Row 4: Skill Gap Learning Recommendations
        st.subheader("💡 Rule-Based Skill Gap Recommendations")
        st.caption("Suggested self-learning path based on missing skills detected in the target job description:")

        for skill, rec in recommendations.items():
            st.markdown(f"""
            <div class="recommendation-card">
                <div class="rec-skill">📌 Skill Gap: {skill}</div>
                <div class="rec-text">{rec}</div>
            </div>
            """, unsafe_allow_html=True)

        st.info("ℹ️ *Note: Recommendations are generated dynamically based on rule-based NLP skill gap matching.*")


if __name__ == "__main__":
    main()
