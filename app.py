import os
import json
import time
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Existing core NLP modules
from src.text_preprocessing import extract_text_from_pdf, clean_text
from src.similarity import calculate_similarity
from src.skill_extractor import analyze_skill_gaps, load_skills_dictionary
from src.recommendations import generate_recommendations

# Upgraded AI modules
from src.semantic_matcher import perform_semantic_skill_matching
from src.scoring import calculate_explainable_scores
from src.priority_analyzer import classify_skill_priorities
from src.roadmap_generator import generate_personalized_roadmap
from src.explainable_ai import generate_explainable_insights
from src.simulator import simulate_skill_improvement
from src.resume_feedback import analyze_resume_formatting_and_content
from src.evaluation import run_academic_evaluation_benchmark

# Page configuration
st.set_page_config(
    page_title="SkillMatch AI - Semantic Resume Intelligence & Skill Gap Analyzer",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Dark Navy AI Dashboard
st.markdown("""
<style>
    /* Dark Navy Base Theme */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #1E1B4B 0%, #0F172A 50%, #090D16 100%);
        color: #F8FAFC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Hide default overhead elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        max-width: 1240px;
    }

    /* Top Nav / Branding Bar */
    .nav-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 1.8rem;
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        backdrop-filter: blur(16px);
        margin-bottom: 1.8rem;
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
        padding: 0.8rem 1rem 2rem 1rem;
    }
    .hero-head {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(180deg, #FFFFFF 0%, #CBD5E1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.6rem;
        letter-spacing: -0.8px;
    }
    .hero-sub {
        font-size: 1.1rem;
        color: #94A3B8;
        max-width: 720px;
        margin: 0 auto;
        line-height: 1.6;
        font-weight: 400;
    }

    /* Metric Grid Cards */
    .metric-grid-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 1.3rem 1rem;
        text-align: center;
        transition: all 0.25s ease;
    }
    .metric-grid-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 8px 24px rgba(56, 189, 248, 0.12);
    }
    .metric-val {
        font-size: 2.2rem;
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
    }
    .chip-strong {
        background: rgba(16, 185, 129, 0.14);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .chip-partial {
        background: rgba(245, 158, 11, 0.14);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .chip-missing {
        background: rgba(244, 63, 94, 0.14);
        color: #FB7185;
        border: 1px solid rgba(244, 63, 94, 0.4);
    }

    /* Priority Badges */
    .p-high {
        color: #EF4444;
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .p-med {
        color: #F59E0B;
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .p-low {
        color: #3B82F6;
        background: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.3);
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
    }

    /* Cards */
    .glass-card-inner {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
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
</style>
""", unsafe_allow_html=True)


def load_job_roles_templates():
    """Loads target job role profiles."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    roles_path = os.path.join(base_dir, 'data', 'job_roles.json')
    if os.path.exists(roles_path):
        with open(roles_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def load_sample_files():
    """Loads sample resume and job text."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    res_path = os.path.join(base_dir, 'sample', 'sample_resume.txt')
    job_path = os.path.join(base_dir, 'sample', 'sample_job_description.txt')

    res_text = open(res_path, 'r', encoding='utf-8').read() if os.path.exists(res_path) else ""
    job_text = open(job_path, 'r', encoding='utf-8').read() if os.path.exists(job_path) else ""
    return res_text, job_text


def render_chips(skills_list: list, chip_class: str) -> str:
    """Generates HTML chips."""
    if not skills_list:
        return "<span style='color: #64748B; font-style: italic;'>None detected</span>"
    items = []
    for item in skills_list:
        if isinstance(item, dict):
            text = f"{item['skill']} ({item.get('score', 100)}%)" if 'score' in item else item['skill']
        else:
            text = str(item)
        items.append(f'<span class="chip {chip_class}">{text}</span>')
    return f'<div class="chip-container">{"".join(items)}</div>'


def main():
    # 1. TOP NAVIGATION & BRANDING
    st.markdown("""
    <div class="nav-bar">
        <div>
            <div class="brand-logo">✦ SKILLMATCH AI</div>
            <div class="brand-sub">Semantic Resume Intelligence & Skill Gap Analyzer</div>
        </div>
        <div class="status-badge">
            <span class="pulse-dot"></span>
            SENTENCE EMBEDDINGS ENGINE READY
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. HERO SECTION
    st.markdown("""
    <div class="hero-box">
        <div class="hero-head">Discover your resume's true skill gap.</div>
        <div class="hero-sub">Powered by Sentence Transformers, NLP vector embeddings, and concept similarity matching.</div>
    </div>
    """, unsafe_allow_html=True)

    # Session State setup
    if "resume_text_val" not in st.session_state:
        st.session_state.resume_text_val = ""
    if "job_text_val" not in st.session_state:
        st.session_state.job_text_val = ""

    job_roles_templates = load_job_roles_templates()

    # Toolbar for Template Selection & Sample Demo
    tb_col1, tb_col2 = st.columns([3, 1])
    with tb_col1:
        selected_template = st.selectbox(
            "🎯 Select Target Job Role Profile (Optional Template):",
            ["Custom Job Description"] + list(job_roles_templates.keys())
        )
        if selected_template != "Custom Job Description":
            st.session_state.job_text_val = job_roles_templates[selected_template]["description"]

    with tb_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📋 Load Sample Demo Data", use_container_width=True):
            r_samp, j_samp = load_sample_files()
            st.session_state.resume_text_val = r_samp
            st.session_state.job_text_val = j_samp
            st.toast("Sample Demo Loaded Successfully!", icon="✨")

    # 3. INPUT WORKSPACE
    col_left, col_right = st.columns(2)

    resume_input_text = ""

    with col_left:
        st.subheader("📄 YOUR RESUME")
        resume_mode = st.radio("Input mode:", ["Upload PDF", "Paste Raw Text"], horizontal=True, label_visibility="collapsed")

        if resume_mode == "Upload PDF":
            uploaded_file = st.file_uploader("Upload Resume PDF", type=["pdf"])
            if uploaded_file is not None:
                try:
                    resume_input_text = extract_text_from_pdf(uploaded_file)
                    st.success(f"PDF Parsed ({len(resume_input_text.split())} words)")
                except Exception as e:
                    st.error(f"PDF Parsing Error: {str(e)}")
            elif st.session_state.resume_text_val:
                st.info("Using preloaded sample resume text")
                resume_input_text = st.session_state.resume_text_val
        else:
            resume_input_text = st.text_area(
                "Paste Resume Text:",
                value=st.session_state.resume_text_val,
                height=250,
                placeholder="Paste candidate resume text here...",
                label_visibility="collapsed"
            )

    with col_right:
        st.subheader("🎯 TARGET JOB REQUIREMENT")
        job_input_text = st.text_area(
            "Target Job Description:",
            value=st.session_state.job_text_val,
            height=300,
            placeholder="Paste target job description or select a role template above...",
            label_visibility="collapsed"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. ANALYZE BUTTON
    analyze_clicked = st.button("✨ RUN SEMANTIC AI ANALYSIS", type="primary", use_container_width=True)

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
            st.error("⚠️ Please select a Job Role template or paste a Job Description.")
            return

        with st.status("✨ Executing Semantic AI NLP Pipeline...", expanded=True) as status:
            st.write("📄 Extracting & cleaning text tokens...")
            time.sleep(0.2)

            st.write("📐 Calculating TF-IDF Document Vector Similarity...")
            sim_result = calculate_similarity(resume_input_text, job_input_text)
            tfidf_sim = sim_result["match_percentage"]

            st.write("🧠 Generating Sentence Embeddings & Cosine Concept Match Matrix...")
            semantic_res = perform_semantic_skill_matching(resume_input_text, job_input_text)

            st.write("📊 Computing Explainable Multi-Dimensional Match Scores...")
            scoring_res = calculate_explainable_scores(tfidf_sim, semantic_res, resume_input_text, job_input_text)

            st.write("🎯 Classifying Skill Priorities & Building Personalized Roadmap...")
            prioritized_skills = classify_skill_priorities(semantic_res["missing_skills"], job_input_text)
            roadmap = generate_personalized_roadmap(prioritized_skills)

            st.write("💡 Generating Explainable AI ('Why?') Insights...")
            explainable_insights = generate_explainable_insights(
                semantic_res["strong_matches"],
                semantic_res["partial_matches"],
                semantic_res["missing_skills"],
                prioritized_skills,
                semantic_res["embedding_model_name"]
            )

            st.write("📝 Analyzing Resume Action Verbs & Metric Quality...")
            resume_feedback = analyze_resume_formatting_and_content(
                resume_input_text,
                [s["skill"] for s in semantic_res["missing_skills"]]
            )

            status.update(label="✅ Semantic AI Analysis Complete!", state="complete", expanded=False)

        st.markdown("<br><hr style='border-color: rgba(255,255,255,0.08);'><br>", unsafe_allow_html=True)

        # TABBED DASHBOARD VIEW FOR ACADEMIC & USER PRESENTATION
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 AI Skill Analysis",
            "🎯 Priorities & Roadmap",
            "💡 Explainable AI & Simulator",
            "📝 Resume Feedback",
            "🔬 Evaluation Metrics",
            "📖 AI Methodology & Viva Docs"
        ])

        # TAB 1: AI SKILL ANALYSIS & SEMANTIC MATCHING
        with tab1:
            st.markdown("## 📊 Semantic Skill Match Dashboard")
            st.caption(f"Embedding Engine: **{semantic_res['embedding_model_name']}**")

            # Metric Cards
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""
                <div class="metric-grid-card">
                    <div class="metric-val" style="background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{scoring_res['overall_score']}%</div>
                    <div class="metric-label-text">Overall Match Score</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-grid-card">
                    <div class="metric-val" style="color: #34D399;">{len(semantic_res['strong_matches'])}</div>
                    <div class="metric-label-text">Strong Matches (✓)</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-grid-card">
                    <div class="metric-val" style="color: #FBBF24;">{len(semantic_res['partial_matches'])}</div>
                    <div class="metric-label-text">Partial Matches (~)</div>
                </div>
                """, unsafe_allow_html=True)
            with m4:
                st.markdown(f"""
                <div class="metric-grid-card">
                    <div class="metric-val" style="color: #FB7185;">{len(semantic_res['missing_skills'])}</div>
                    <div class="metric-label-text">Missing Skills (✗)</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.info(f"📐 **Score Breakdown Formula**: `{scoring_res['explanation_formula']}`")
            st.markdown("<br>", unsafe_allow_html=True)

            c_gauge, c_tags = st.columns([1, 1.6])

            with c_gauge:
                st.subheader("🎯 Cosine Similarity Gauge")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=scoring_res['overall_score'],
                    number={'suffix': "%", 'font': {'size': 42, 'color': "#F8FAFC"}},
                    title={'text': "Semantic Match Index", 'font': {'size': 14, 'color': "#94A3B8"}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#334155"},
                        'bar': {'color': "#38BDF8", 'thickness': 0.3},
                        'bgcolor': "rgba(15, 23, 42, 0.4)",
                        'steps': [
                            {'range': [0, 40], 'color': 'rgba(244, 63, 94, 0.15)'},
                            {'range': [40, 70], 'color': 'rgba(245, 158, 11, 0.15)'},
                            {'range': [70, 100], 'color': 'rgba(16, 185, 129, 0.15)'}
                        ]
                    }
                ))
                fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#F8FAFC"}, height=260, margin=dict(l=20,r=20,t=30,b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)

            with c_tags:
                st.subheader("✓ STRONG MATCHES (>=85% Similarity)")
                st.markdown(render_chips(semantic_res['strong_matches'], "chip-strong"), unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                st.subheader("~ PARTIAL MATCHES (50% - 84% Concept Similarity)")
                st.markdown(render_chips(semantic_res['partial_matches'], "chip-partial"), unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                st.subheader("✗ MISSING SKILLS (<50% Similarity)")
                st.markdown(render_chips(semantic_res['missing_skills'], "chip-missing"), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Category Breakdown
            cat_col1, cat_col2 = st.columns([1.2, 1])
            with cat_col1:
                st.subheader("📋 Domain Category Scores")
                cat_data = []
                for cat_name, info in scoring_res['category_scores'].items():
                    cat_data.append({
                        "Category": cat_name,
                        "Match Score": f"{info['score']}%",
                        "Requirement Status": info['status']
                    })
                st.dataframe(pd.DataFrame(cat_data), hide_index=True, use_container_width=True)

            with cat_col2:
                st.subheader("📈 Domain Match Visualization")
                categories = list(scoring_res['category_scores'].keys())
                scores = [info['score'] for info in scoring_res['category_scores'].values()]
                fig_bar = go.Figure(go.Bar(x=categories, y=scores, marker_color='#38BDF8'))
                fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#CBD5E1"}, height=280, margin=dict(l=10,r=10,t=30,b=10))
                st.plotly_chart(fig_bar, use_container_width=True)

        # TAB 2: SKILL PRIORITIES & PERSONALIZED ROADMAP
        with tab2:
            st.markdown("## 🎯 Skill Priority Classification & Personalized Learning Path")

            p_col1, p_col2 = st.columns([1.2, 1.5])

            with p_col1:
                st.subheader("⚠️ Missing Skill Priority Classification")
                for item in prioritized_skills:
                    p_class = "p-high" if item['priority'] == "HIGH" else ("p-med" if item['priority'] == "MEDIUM" else "p-low")
                    st.markdown(f"""
                    <div class="glass-card-inner">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <strong style="font-size:1.05rem;">{item['skill']}</strong>
                            <span class="{p_class}">{item['priority']} PRIORITY</span>
                        </div>
                        <div style="font-size:0.88rem; color:#94A3B8; margin-top:6px;">{item['reason']}</div>
                    </div>
                    """, unsafe_allow_html=True)

            with p_col2:
                st.subheader("🚀 4-Week Dynamic Learning Roadmap")
                for step in roadmap:
                    st.markdown(f"""
                    <div class="glass-card-inner" style="border-left: 4px solid #818CF8;">
                        <div style="font-weight:700; color:#38BDF8;">{step['week']}: {step['title']}</div>
                        <div style="font-size:0.92rem; color:#E2E8F0; margin-top:4px;">{step['tasks']}</div>
                        <div style="font-size:0.8rem; color:#94A3B8; margin-top:6px;">Focus: {', '.join(step['focus_skills'])}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # TAB 3: EXPLAINABLE AI ("WHY?") & WHAT-IF SIMULATOR
        with tab3:
            st.markdown("## 💡 Explainable AI Insights & Skill Improvement Simulator")

            ex_col1, ex_col2 = st.columns([1.3, 1.2])

            with ex_col1:
                st.subheader("🔍 Explainable AI ('Why?') Insights")
                for ins in explainable_insights:
                    st.markdown(f"""
                    <div class="glass-card-inner">
                        <div style="font-weight:700; color:#F8FAFC;">📌 {ins['topic']}</div>
                        <div style="font-size:0.9rem; color:#CBD5E1; margin-top:4px;">{ins['explanation']}</div>
                    </div>
                    """, unsafe_allow_html=True)

            with ex_col2:
                st.subheader("⚡ Skill Improvement Simulator (What-If Analysis)")
                st.caption("Select missing skills you plan to learn to see your estimated match score boost:")

                missing_skill_names = [s["skill"] for s in semantic_res["missing_skills"]]

                selected_to_learn = []
                for sk in missing_skill_names:
                    if st.checkbox(f"Acquire: {sk}", key=f"sim_{sk}"):
                        selected_to_learn.append(sk)

                sim_result = simulate_skill_improvement(
                    scoring_res["overall_score"],
                    semantic_res["strong_matches"],
                    semantic_res["partial_matches"],
                    len(semantic_res["job_skills"]),
                    selected_to_learn
                )

                st.markdown(f"""
                <div class="glass-card-inner" style="text-align:center; border:1px solid #38BDF8;">
                    <div style="font-size:0.9rem; color:#94A3B8;">Current Match: {sim_result['initial_score']}%</div>
                    <div style="font-size:2.2rem; font-weight:800; color:#34D399; margin:6px 0;">{sim_result['simulated_score']}%</div>
                    <div style="font-size:0.95rem; color:#38BDF8;">Estimated Boost: +{sim_result['score_boost']}%</div>
                </div>
                """, unsafe_allow_html=True)

        # TAB 4: RESUME IMPROVEMENT & BULLET FEEDBACK
        with tab4:
            st.markdown("## 📝 Resume Formatting & Content Optimization")

            fb_col1, fb_col2 = st.columns(2)

            with fb_col1:
                st.subheader("📊 Content Quality Check")
                st.write(f"• **Action Verbs Detected**: `{resume_feedback.get('action_verb_count', 0)}` ({', '.join(resume_feedback.get('found_action_verbs', []))})")
                st.write(f"• **Quantifiable Metrics Found**: `{resume_feedback.get('metrics_count', 0)}`")

                st.markdown("### 💡 Actionable Improvements")
                for sug in resume_feedback.get("suggestions", []):
                    st.warning(f"**[{sug['category']}]**: {sug['recommendation']}")

            with fb_col2:
                st.subheader("✏️ Example Bullet Point Rewrite")
                ex_rw = resume_feedback.get("rewrite_example", {})
                st.markdown(f"""
                <div class="glass-card-inner" style="border-left:4px solid #EF4444;">
                    <strong style="color:#FB7185;">Weak Bullet:</strong><br>"{ex_rw.get('weak_bullet')}"
                </div>
                <div class="glass-card-inner" style="border-left:4px solid #10B981;">
                    <strong style="color:#34D399;">Optimized Impact Bullet:</strong><br>"{ex_rw.get('strong_bullet')}"
                </div>
                """, unsafe_allow_html=True)

        # TAB 5: ACADEMIC EVALUATION DASHBOARD
        with tab5:
            st.markdown("## 🔬 Empirical AI Evaluation & Benchmark Dashboard")
            st.caption("Real performance metrics calculated at runtime over a 5-pair annotated benchmark dataset:")

            eval_res = run_academic_evaluation_benchmark()

            e1, e2, e3, e4 = st.columns(4)
            with e1:
                st.metric("Precision", f"{eval_res['precision']}%")
            with e2:
                st.metric("Recall", f"{eval_res['recall']}%")
            with e3:
                st.metric("F1-Score", f"{eval_res['f1_score']}%")
            with e4:
                st.metric("Avg Latency", f"{eval_res['avg_latency_ms']} ms")

            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("📋 Benchmark Test Pair Breakdown")
            st.dataframe(pd.DataFrame(eval_res['pair_results']), hide_index=True, use_container_width=True)

        # TAB 6: AI METHODOLOGY & VIVA DOCS
        with tab6:
            st.markdown("""
            ## 📖 AI Methodology & Technical Viva Guide

            ### 1. What Model is Used?
            - **Model**: `SentenceTransformer ('all-MiniLM-L6-v2')` combined with `TF-IDF Character N-Gram Vector Space`.
            - **Vector Dimensions**: 384-dimensional dense semantic embeddings.

            ### 2. Why Was This Model Selected?
            - It maps textual phrases into a dense vector space where semantically similar concepts (e.g. `TensorFlow` and `Deep Learning`) sit close together in vector space, resolving exact keyword limitations.

            ### 3. Mathematical Similarity Formula
            $$\\text{Cosine Similarity}(\\vec{A}, \\vec{B}) = \\frac{\\vec{A} \\cdot \\vec{B}}{\\|\\vec{A}\\| \\|\\vec{B}\\|}$$

            ### 4. Matching Thresholds
            - **Strong Match (✓)**: Similarity $\\ge 85\\%$ or exact string match.
            - **Partial Match (~)**: Similarity between $50\\%$ and $84\\%$ (e.g., framework-to-concept relationship).
            - **Missing Skill (✗)**: Similarity $< 50\\%$.
            """, unsafe_allow_html=True)

    # 10. FOOTER
    st.markdown("""
    <div class="footer-bar">
        Built with Python • SentenceTransformers • NLP Embeddings • Cosine Similarity<br>
        <strong>SkillMatch AI - Resume Intelligence & Skill Gap Analyzer</strong>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
