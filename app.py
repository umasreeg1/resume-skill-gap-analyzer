import os
import time
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.text_preprocessing import extract_text_from_pdf, clean_text
from src.similarity import calculate_similarity
from src.skill_extractor import analyze_skill_gaps
from src.recommendations import generate_recommendations

# Page configuration
st.set_page_config(
    page_title="SkillMatch AI - Resume Intelligence & Skill Gap Analysis",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern AI Tech Theme CSS
st.markdown("""
<style>
    /* Dark Navy Glassmorphism Base Theme */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #1E1B4B 0%, #0F172A 50%, #090D16 100%);
        color: #F8FAFC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Hide default Streamlit overhead elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Top Navigation / Branding Bar */
    .nav-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 1.8rem;
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        backdrop-filter: blur(16px);
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .brand-logo {
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .brand-sub {
        font-size: 0.85rem;
        color: #94A3B8;
        font-weight: 500;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 30px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34D399;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #34D399;
        border-radius: 50%;
        box-shadow: 0 0 8px #34D399;
    }

    /* Hero Section */
    .hero-box {
        text-align: center;
        padding: 1rem 1rem 2.2rem 1rem;
    }
    .hero-head {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(180deg, #FFFFFF 0%, #CBD5E1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.8rem;
        letter-spacing: -0.8px;
    }
    .hero-sub {
        font-size: 1.15rem;
        color: #94A3B8;
        max-width: 680px;
        margin: 0 auto;
        line-height: 1.6;
        font-weight: 400;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 1.5rem;
        backdrop-filter: blur(16px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
        height: 100%;
    }
    .card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Metric Cards */
    .metric-grid-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 1.4rem 1rem;
        text-align: center;
        transition: all 0.25s ease;
    }
    .metric-grid-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 8px 24px rgba(56, 189, 248, 0.12);
    }
    .metric-val {
        font-size: 2.3rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .metric-label-text {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94A3B8;
        font-weight: 600;
        margin-top: 6px;
    }

    /* Skill Chip Tags */
    .chip-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 8px;
    }
    .chip {
        display: inline-flex;
        align-items: center;
        padding: 6px 14px;
        border-radius: 30px;
        font-size: 0.88rem;
        font-weight: 600;
        backdrop-filter: blur(8px);
        transition: transform 0.15s ease;
    }
    .chip:hover {
        transform: translateY(-1px);
    }
    .chip-matched {
        background: rgba(16, 185, 129, 0.12);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }
    .chip-missing {
        background: rgba(244, 63, 94, 0.12);
        color: #FB7185;
        border: 1px solid rgba(244, 63, 94, 0.35);
    }
    .chip-additional {
        background: rgba(59, 130, 246, 0.12);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.35);
    }

    /* Recommendation Cards */
    .rec-card {
        background: rgba(30, 41, 59, 0.4);
        border-left: 4px solid #38BDF8;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 14px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 0.9rem;
    }
    .rec-skill-name {
        font-weight: 700;
        font-size: 1.05rem;
        color: #F8FAFC;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .rec-body {
        font-size: 0.92rem;
        color: #CBD5E1;
        line-height: 1.5;
    }

    /* Footer */
    .footer-bar {
        text-align: center;
        color: #64748B;
        font-size: 0.85rem;
        margin-top: 4rem;
        padding-top: 1.8rem;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Custom Input Fields */
    .stTextArea textarea, .stTextInput input {
        background-color: rgba(15, 23, 42, 0.6) !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 12px !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #38BDF8 !important;
        box-shadow: 0 0 0 1px #38BDF8 !important;
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
        return "<span style='color: #64748B; font-style: italic;'>None detected</span>"
    html_chips = "".join([f'<span class="chip {chip_class}">{skill}</span>' for skill in skills_list])
    return f'<div class="chip-container">{html_chips}</div>'


def main():
    # 1. TOP NAVIGATION / BRANDING
    st.markdown("""
    <div class="nav-bar">
        <div>
            <div class="brand-logo">✦ SKILLMATCH AI</div>
            <div class="brand-sub">Resume Intelligence & Skill Gap Analysis</div>
        </div>
        <div class="status-badge">
            <span class="pulse-dot"></span>
            AI ENGINE READY
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. HERO SECTION
    st.markdown("""
    <div class="hero-box">
        <div class="hero-head">Discover your resume's skill gap.</div>
        <div class="hero-sub">Compare your resume with a target role using NLP-powered document similarity and skill analysis.</div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize Session States for demo data pre-fill
    if "resume_text_val" not in st.session_state:
        st.session_state.resume_text_val = ""
    if "job_text_val" not in st.session_state:
        st.session_state.job_text_val = ""

    # Sample Data Action Toolbar
    s_col1, s_col2 = st.columns([4, 1])
    with s_col2:
        if st.button("📋 Load Sample Demo", use_container_width=True):
            r_sample, j_sample = load_sample_files()
            st.session_state.resume_text_val = r_sample
            st.session_state.job_text_val = j_sample
            st.toast("Sample Demo Loaded!", icon="✨")

    # 3. INPUT WORKSPACE (Two Side-by-Side Large Cards)
    col_left, col_right = st.columns(2)

    resume_input_text = ""

    with col_left:
        st.markdown('<div class="card-title">📄 YOUR RESUME</div>', unsafe_allow_html=True)
        resume_mode = st.radio(
            "Choose input mode:",
            ["Upload PDF", "Paste Raw Text"],
            horizontal=True,
            label_visibility="collapsed"
        )

        if resume_mode == "Upload PDF":
            uploaded_file = st.file_uploader("Upload Resume PDF", type=["pdf"])
            if uploaded_file is not None:
                try:
                    resume_input_text = extract_text_from_pdf(uploaded_file)
                    st.success(f"PDF processed ({len(resume_input_text.split())} words extracted)")
                except Exception as e:
                    st.error(f"Error parsing PDF: {str(e)}")
            elif st.session_state.resume_text_val:
                st.info("Using preloaded sample resume text")
                resume_input_text = st.session_state.resume_text_val
        else:
            resume_input_text = st.text_area(
                "Paste Resume Text:",
                value=st.session_state.resume_text_val,
                height=250,
                placeholder="Paste the raw text content of your resume here...",
                label_visibility="collapsed"
            )

    with col_right:
        st.markdown('<div class="card-title">🎯 TARGET JOB</div>', unsafe_allow_html=True)
        job_input_text = st.text_area(
            "Paste Target Job Description:",
            value=st.session_state.job_text_val,
            height=300,
            placeholder="Paste the job description or role requirements here...",
            label_visibility="collapsed"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. ANALYZE BUTTON
    analyze_clicked = st.button("✨ ANALYZE SKILL GAP", type="primary", use_container_width=True)

    if analyze_clicked or (st.session_state.resume_text_val and st.session_state.job_text_val and not resume_input_text):
        if not resume_input_text and st.session_state.resume_text_val:
            resume_input_text = st.session_state.resume_text_val
        if not job_input_text and st.session_state.job_text_val:
            job_input_text = st.session_state.job_text_val

    if analyze_clicked:
        if not resume_input_text or not resume_input_text.strip():
            st.error("⚠️ Please upload or paste a valid Resume before running analysis.")
            return

        if not job_input_text or not job_input_text.strip():
            st.error("⚠️ Please paste a valid Target Job Description before running analysis.")
            return

        # UI Progress / Status sequence
        with st.status("✨ Analyzing Skill Gap...", expanded=True) as status:
            st.write("📄 Extracting resume...")
            time.sleep(0.2)
            st.write("🧹 Preprocessing text...")
            time.sleep(0.2)
            
            # Execute actual existing NLP functions
            similarity_result = calculate_similarity(resume_input_text, job_input_text)
            match_score = similarity_result["match_percentage"]
            cosine_sim = similarity_result["cosine_sim"]
            
            st.write("📐 Calculating similarity...")
            time.sleep(0.2)
            
            gap_result = analyze_skill_gaps(resume_input_text, job_input_text)
            matching_skills = gap_result["matching_skills"]
            missing_skills = gap_result["missing_skills"]
            additional_skills = gap_result["additional_skills"]
            category_breakdown = gap_result["category_breakdown"]

            st.write("🔍 Analyzing skills...")
            time.sleep(0.2)

            recommendations = generate_recommendations(missing_skills)
            st.write("🚀 Generating report...")
            time.sleep(0.2)

            status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

        # 5. RESULTS SECTION
        st.markdown("<br><hr style='border-color: rgba(255,255,255,0.08);'><br>", unsafe_allow_html=True)
        st.markdown("## 📊 Analysis Results")

        # 5. Metric Cards Grid
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.markdown(f"""
            <div class="metric-grid-card">
                <div class="metric-val" style="background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{match_score}%</div>
                <div class="metric-label-text">Resume Match</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-grid-card">
                <div class="metric-val" style="color: #34D399;">{len(matching_skills)}</div>
                <div class="metric-label-text">Matching Skills</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="metric-grid-card">
                <div class="metric-val" style="color: #FB7185;">{len(missing_skills)}</div>
                <div class="metric-label-text">Missing Skills</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            st.markdown(f"""
            <div class="metric-grid-card">
                <div class="metric-val" style="color: #60A5FA;">{len(additional_skills)}</div>
                <div class="metric-label-text">Additional Skills</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 6. MATCH SCORE & 7. SKILL ANALYSIS SECTIONS
        c_score, c_skills = st.columns([1, 1.6])

        with c_score:
            st.markdown('<div class="card-title">🎯 Similarity Gauge</div>', unsafe_allow_html=True)
            # Circular Gauge Indicator Chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=match_score,
                number={'suffix': "%", 'font': {'size': 42, 'color': "#F8FAFC", 'family': "Inter"}},
                title={'text': "Cosine Similarity Score", 'font': {'size': 14, 'color': "#94A3B8"}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#334155"},
                    'bar': {'color': "#38BDF8", 'thickness': 0.3},
                    'bgcolor': "rgba(15, 23, 42, 0.4)",
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 40], 'color': 'rgba(244, 63, 94, 0.15)'},
                        {'range': [40, 70], 'color': 'rgba(251, 191, 36, 0.15)'},
                        {'range': [70, 100], 'color': 'rgba(16, 185, 129, 0.15)'}
                    ],
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "#F8FAFC"},
                height=260,
                margin=dict(l=20, r=20, t=30, b=10)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with c_skills:
            st.markdown('<div class="card-title">✓ MATCHING SKILLS</div>', unsafe_allow_html=True)
            st.markdown(render_chips(matching_skills, "chip-matched"), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown('<div class="card-title">⚠ MISSING SKILLS</div>', unsafe_allow_html=True)
            st.markdown(render_chips(missing_skills, "chip-missing"), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown('<div class="card-title">+ ADDITIONAL SKILLS</div>', unsafe_allow_html=True)
            st.markdown(render_chips(additional_skills, "chip-additional"), unsafe_allow_html=True)

        st.markdown("<br><hr style='border-color: rgba(255,255,255,0.08);'><br>", unsafe_allow_html=True)

        # 8. SKILL CATEGORY BREAKDOWN & CHART
        cat1, cat2 = st.columns([1.2, 1])

        with cat1:
            st.markdown('### 📋 Category Breakdown Table')
            df_cat = pd.DataFrame(category_breakdown)
            st.dataframe(
                df_cat[["Category", "Job Required", "Matched", "Missing", "Additional"]],
                hide_index=True,
                use_container_width=True
            )

        with cat2:
            st.markdown('### 📈 Matched vs Missing by Category')
            categories = [d["Category"] for d in category_breakdown]
            matched_counts = [d["Matched"] for d in category_breakdown]
            missing_counts = [d["Missing"] for d in category_breakdown]

            fig_bar = go.Figure(data=[
                go.Bar(name='Matched', x=categories, y=matched_counts, marker_color='#10B981'),
                go.Bar(name='Missing', x=categories, y=missing_counts, marker_color='#EF4444')
            ])
            fig_bar.update_layout(
                barmode='group',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "#CBD5E1"},
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=290,
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("<br><hr style='border-color: rgba(255,255,255,0.08);'><br>", unsafe_allow_html=True)

        # 9. LEARNING RECOMMENDATIONS
        st.markdown('### 🚀 Your Next Learning Steps')
        st.caption("Tailored learning suggestions based on missing skill gaps detected in target role:")

        for skill, rec in recommendations.items():
            st.markdown(f"""
            <div class="rec-card">
                <div class="rec-skill-name">📌 {skill}</div>
                <div class="rec-body">{rec}</div>
            </div>
            """, unsafe_allow_html=True)

    # 10. FOOTER
    st.markdown("""
    <div class="footer-bar">
        Built with Python • NLP • TF-IDF • Cosine Similarity<br>
        <strong>AI Resume Skill Gap Analyzer</strong>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
