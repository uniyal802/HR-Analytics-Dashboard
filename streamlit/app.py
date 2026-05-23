import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    'figure.facecolor': '#0D0D14',
    'axes.facecolor':   '#0D0D14',
    'axes.edgecolor':   '#2A2A3D',
    'axes.labelcolor':  '#A0A0C0',
    'xtick.color':      '#A0A0C0',
    'ytick.color':      '#A0A0C0',
    'text.color':       '#E0E0FF',
    'grid.color':       '#1E1E2E',
    'grid.linestyle':   '--',
    'grid.alpha':       0.5,
})

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title='HR Analytics · Intelligence Suite',
    page_icon='🧬',
    layout='wide',
    initial_sidebar_state='expanded'
)

# =====================================================
# GLOBAL CSS — ANIMATED MESH BACKGROUND + GLASSMORPHISM
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── ROOT VARIABLES ── */
:root {
    --bg-deep:    #07070F;
    --bg-card:    rgba(255,255,255,0.035);
    --border:     rgba(255,255,255,0.07);
    --cyan:       #00F5D4;
    --violet:     #7C3AED;
    --rose:       #F43F5E;
    --amber:      #F59E0B;
    --text-hi:    #F0F0FF;
    --text-mid:   #9090B8;
    --text-lo:    #4A4A6A;
    --glow-cyan:  0 0 40px rgba(0,245,212,0.18);
    --glow-vio:   0 0 40px rgba(124,58,237,0.22);
    --radius:     16px;
    --radius-lg:  24px;
}

/* ── GLOBAL RESET ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-hi) !important;
}

/* ── ANIMATED BACKGROUND MESH ── */
.stApp {
    background: var(--bg-deep) !important;
    min-height: 100vh;
    position: relative;
    overflow-x: hidden;
}

.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 60% at 10% -10%,  rgba(0,245,212,0.10) 0%, transparent 65%),
        radial-gradient(ellipse 60% 50% at 90%  20%,  rgba(124,58,237,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 70% 40% at 50% 110%,  rgba(244,63,94,0.08)  0%, transparent 60%),
        radial-gradient(ellipse 50% 60% at 80% 80%,   rgba(245,158,11,0.06) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
    animation: meshShift 18s ease-in-out infinite alternate;
}

@keyframes meshShift {
    0%   { opacity: 1;   filter: blur(0px);   }
    50%  { opacity: 0.7; filter: blur(2px);   }
    100% { opacity: 1;   filter: blur(0px);   }
}

/* ── NOISE GRAIN OVERLAY ── */
.stApp::after {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.4;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0C0C18 0%, #090912 100%) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 4px 0 40px rgba(0,0,0,0.5);
}

[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--cyan), var(--violet), var(--rose));
}

[data-testid="stSidebar"] .stRadio label {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    padding: 12px 16px !important;
    border-radius: 10px !important;
    margin: 4px 0 !important;
    transition: all 0.25s ease !important;
    color: var(--text-mid) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    font-weight: 400 !important;
    cursor: pointer !important;
    border: 1px solid transparent !important;
}

[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(0,245,212,0.06) !important;
    color: var(--cyan) !important;
    border-color: rgba(0,245,212,0.15) !important;
}

[data-testid="stSidebar"] [aria-checked="true"] + label,
[data-testid="stSidebar"] input[type="radio"]:checked + div label {
    background: rgba(0,245,212,0.08) !important;
    color: var(--cyan) !important;
    border-color: rgba(0,245,212,0.25) !important;
}

/* ── MAIN CONTENT AREA ── */
.main .block-container {
    padding: 1.5rem 2.5rem 4rem !important;
    max-width: 100% !important;
    position: relative;
    z-index: 1;
}

/* ── GLASS CARD ── */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 28px 32px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}

.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent);
}

/* ── NAVBAR ── */
.navbar-wrap {
    background: rgba(255,255,255,0.025);
    backdrop-filter: blur(24px);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 20px 32px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: relative;
    overflow: hidden;
}

.navbar-wrap::before {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--cyan), var(--violet), transparent);
    opacity: 0.4;
}

.navbar-logo {
    font-family: 'Syne', sans-serif !important;
    font-size: 24px !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, var(--cyan) 0%, var(--violet) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 !important;
}

.navbar-sub {
    font-size: 12px;
    color: var(--text-lo);
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight: 300;
    margin-top: 4px;
}

.navbar-badge {
    background: rgba(0,245,212,0.1);
    border: 1px solid rgba(0,245,212,0.25);
    color: var(--cyan);
    padding: 6px 14px;
    border-radius: 100px;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ── SECTION TITLE ── */
.section-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 28px !important;
    font-weight: 700 !important;
    color: var(--text-hi) !important;
    margin: 0 0 6px 0 !important;
    letter-spacing: -0.5px;
}

.section-sub {
    font-size: 13px;
    color: var(--text-lo);
    margin-bottom: 28px;
    letter-spacing: 0.3px;
}

/* ── METRIC CARDS ── */
.metric-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px 28px;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}

.metric-card:hover {
    border-color: rgba(0,245,212,0.3);
    box-shadow: var(--glow-cyan);
    transform: translateY(-2px);
}

.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 0 0 var(--radius) var(--radius);
}

.metric-card.cyan::after  { background: var(--cyan); box-shadow: 0 0 12px var(--cyan); }
.metric-card.violet::after{ background: var(--violet); box-shadow: 0 0 12px var(--violet); }
.metric-card.rose::after  { background: var(--rose); box-shadow: 0 0 12px var(--rose); }
.metric-card.amber::after { background: var(--amber); box-shadow: 0 0 12px var(--amber); }

.metric-icon {
    font-size: 28px;
    margin-bottom: 14px;
    display: block;
}

.metric-label {
    font-size: 11px;
    color: var(--text-lo);
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 500;
    margin-bottom: 8px;
}

.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 36px;
    font-weight: 800;
    color: var(--text-hi);
    line-height: 1;
    letter-spacing: -1px;
}

.metric-delta {
    font-size: 12px;
    margin-top: 8px;
    display: flex;
    align-items: center;
    gap: 4px;
}

.metric-delta.up   { color: var(--cyan); }
.metric-delta.down { color: var(--rose); }

/* ── CHART CONTAINER ── */
.chart-wrap {
    background: rgba(255,255,255,0.025);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    position: relative;
}

.chart-title {
    font-family: 'Syne', sans-serif;
    font-size: 15px;
    font-weight: 600;
    color: var(--text-hi);
    margin-bottom: 4px;
    letter-spacing: -0.2px;
}

.chart-sub {
    font-size: 11px;
    color: var(--text-lo);
    margin-bottom: 16px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

/* ── DIVIDER ── */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 32px 0;
    border: none;
}

/* ── PREDICTION FORM ── */
.input-label {
    font-size: 12px;
    color: var(--text-mid);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 6px;
    display: block;
}

/* override streamlit inputs */
[data-testid="stNumberInput"] input,
[data-testid="stSlider"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-hi) !important;
}

/* ── PREDICT BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, var(--cyan) 0%, #00C4A8 100%) !important;
    color: #07070F !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 36px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    cursor: pointer !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 0 30px rgba(0,245,212,0.25) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 50px rgba(0,245,212,0.4) !important;
}

/* ── RESULT CARD ── */
.result-card {
    border-radius: var(--radius);
    padding: 24px 28px;
    text-align: center;
    font-family: 'Syne', sans-serif;
    font-size: 20px;
    font-weight: 700;
    letter-spacing: -0.3px;
    margin-top: 20px;
    animation: fadeIn 0.5s ease;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

.result-stay {
    background: rgba(0,245,212,0.08);
    border: 1px solid rgba(0,245,212,0.3);
    color: var(--cyan);
    box-shadow: 0 0 40px rgba(0,245,212,0.1);
}

.result-leave {
    background: rgba(244,63,94,0.08);
    border: 1px solid rgba(244,63,94,0.3);
    color: var(--rose);
    box-shadow: 0 0 40px rgba(244,63,94,0.1);
}

/* ── SIDEBAR TITLE ── */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
}

/* ── SELECTBOX ── */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-hi) !important;
}

/* ── FOOTER ── */
.footer-wrap {
    border-top: 1px solid var(--border);
    margin-top: 60px;
    padding: 28px 0 16px;
    text-align: center;
    position: relative;
}

.footer-logo {
    font-family: 'Syne', sans-serif;
    font-size: 16px;
    font-weight: 700;
    background: linear-gradient(135deg, var(--cyan), var(--violet));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px;
}

.footer-text {
    font-size: 12px;
    color: var(--text-lo);
    letter-spacing: 0.5px;
}

.footer-tags {
    display: flex;
    justify-content: center;
    gap: 10px;
    margin-top: 14px;
    flex-wrap: wrap;
}

.footer-tag {
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    color: var(--text-lo);
    padding: 4px 12px;
    border-radius: 100px;
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

/* ── SIDEBAR LOGO BLOCK ── */
.sidebar-logo {
    text-align: center;
    padding: 24px 16px 20px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 20px;
}

.sidebar-logo-icon {
    font-size: 40px;
    display: block;
    margin-bottom: 8px;
}

.sidebar-logo-name {
    font-family: 'Syne', sans-serif;
    font-size: 18px;
    font-weight: 800;
    background: linear-gradient(135deg, var(--cyan), var(--violet));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
}

.sidebar-logo-sub {
    font-size: 10px;
    color: var(--text-lo);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ── HIDE STREAMLIT BRANDING ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── STAT ROW ── */
.stat-row {
    display: flex;
    gap: 10px;
    margin-top: 12px;
}

.stat-pill {
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    border-radius: 100px;
    padding: 5px 14px;
    font-size: 11px;
    color: var(--text-mid);
}

.stat-pill span {
    color: var(--cyan);
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# PATHS
# =====================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(CURRENT_DIR, '..', 'dataset')
MODEL_DIR   = os.path.join(CURRENT_DIR, '..', 'models')

# =====================================================
# LOAD DATASET
# =====================================================

@st.cache_data
def load_dataset():
    csv_files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.csv')]
    if not csv_files:
        st.error('⚠️ No CSV file found in dataset folder.')
        st.stop()
    return pd.read_csv(os.path.join(DATASET_DIR, csv_files[0]))

df = load_dataset()

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    model_path = os.path.join(MODEL_DIR, 'hr_attrition_model.pkl')
    return joblib.load(model_path)

model = load_model()

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("""
    <div class="sidebar-logo">
        <span class="sidebar-logo-icon">🧬</span>
        <div class="sidebar-logo-name">HR · Intelligence</div>
        <div class="sidebar-logo-sub">Analytics Suite v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### Navigation")
    menu = st.radio(
        '',
        ['📊  Dashboard', '👥  Employee Data', '🤖  AI Prediction'],
        label_visibility='collapsed'
    )

    st.markdown('<hr style="border-color:rgba(255,255,255,0.06);margin:20px 0">', unsafe_allow_html=True)
    st.markdown("##### Filters")

    department_filter = st.selectbox(
        'Department',
        ['All Departments'] + list(df['Department'].unique())
    )

    if department_filter == 'All Departments':
        filtered_df = df.copy()
    else:
        filtered_df = df[df['Department'] == department_filter]

    st.markdown('<hr style="border-color:rgba(255,255,255,0.06);margin:20px 0">', unsafe_allow_html=True)

    # Mini stats in sidebar
    total = len(df)
    attr_pct = round(df['Attrition'].value_counts(normalize=True).get('Yes', 0) * 100, 1)

    st.markdown(f"""
    <div style="padding: 0 4px;">
        <div style="margin-bottom:12px;">
            <div style="font-size:10px;color:#4A4A6A;letter-spacing:2px;text-transform:uppercase;margin-bottom:4px;">Workforce</div>
            <div style="font-family:'Syne',sans-serif;font-size:26px;font-weight:800;color:#F0F0FF;">{total:,}</div>
        </div>
        <div>
            <div style="font-size:10px;color:#4A4A6A;letter-spacing:2px;text-transform:uppercase;margin-bottom:4px;">Attrition Rate</div>
            <div style="font-family:'Syne',sans-serif;font-size:26px;font-weight:800;color:#F43F5E;">{attr_pct}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# NAVBAR
# =====================================================

page_map = {
    '📊  Dashboard': ('Dashboard Overview', 'Real-time workforce analytics'),
    '👥  Employee Data': ('Employee Records', 'Browse & filter workforce data'),
    '🤖  AI Prediction': ('Attrition Predictor', 'AI-powered risk assessment')
}
page_title, page_sub = page_map[menu]

st.markdown(f"""
<div class="navbar-wrap">
    <div>
        <div class="navbar-logo">🧬 HR · Intelligence Suite</div>
        <div class="navbar-sub">Workforce Analytics Platform</div>
    </div>
    <div style="display:flex;align-items:center;gap:12px;">
        <div class="navbar-badge">● Live</div>
        <div style="font-size:13px;color:#4A4A6A;">IBM HR Dataset</div>
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# ── DASHBOARD ──
# =====================================================

if '📊' in menu:

    st.markdown(f'<div class="section-title">{page_title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">{page_sub}</div>', unsafe_allow_html=True)

    # ── KPI METRICS ──
    total_employees = len(df)
    attrition_yes   = df['Attrition'].value_counts().get('Yes', 0)
    attrition_rate  = round(attrition_yes / total_employees * 100, 1)
    avg_salary      = int(df['MonthlyIncome'].mean())
    avg_tenure      = round(df['YearsAtCompany'].mean(), 1)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="metric-card cyan">
            <span class="metric-icon">👥</span>
            <div class="metric-label">Total Employees</div>
            <div class="metric-value">{total_employees:,}</div>
            <div class="metric-delta up">↑ Active workforce</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card rose">
            <span class="metric-icon">📉</span>
            <div class="metric-label">Attrition Count</div>
            <div class="metric-value">{attrition_yes}</div>
            <div class="metric-delta down">↓ {attrition_rate}% attrition rate</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card amber">
            <span class="metric-icon">💰</span>
            <div class="metric-label">Avg. Monthly Income</div>
            <div class="metric-value">₹{avg_salary:,}</div>
            <div class="metric-delta up">↑ All departments</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card violet">
            <span class="metric-icon">🏆</span>
            <div class="metric-label">Avg. Tenure</div>
            <div class="metric-value">{avg_tenure}y</div>
            <div class="metric-delta up">↑ Years at company</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ── CHARTS ROW 1 ──
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Headcount by Department</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-sub">Employee distribution</div>', unsafe_allow_html=True)

        dept_counts = df['Department'].value_counts()
        colors = ['#00F5D4', '#7C3AED', '#F43F5E']

        fig, ax = plt.subplots(figsize=(7, 3.8))
        bars = ax.bar(dept_counts.index, dept_counts.values,
                      color=colors[:len(dept_counts)], width=0.5,
                      zorder=3, edgecolor='none')

        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 8,
                    f'{int(bar.get_height())}',
                    ha='center', va='bottom',
                    color='#9090B8', fontsize=10)

        ax.set_xlabel('Department', labelpad=10)
        ax.set_ylabel('Employees', labelpad=10)
        ax.yaxis.grid(True, zorder=0)
        ax.xaxis.grid(False)
        ax.spines[:].set_visible(False)
        ax.tick_params(axis='x', rotation=0)
        fig.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Salary Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-sub">Monthly income spread</div>', unsafe_allow_html=True)

        fig2, ax2 = plt.subplots(figsize=(7, 3.8))
        n, bins, patches = ax2.hist(df['MonthlyIncome'], bins=22, edgecolor='none')

        # Gradient coloring on histogram
        norm = plt.Normalize(n.min(), n.max())
        for val, patch in zip(n, patches):
            patch.set_facecolor(plt.cm.cool(norm(val) * 0.7 + 0.1))

        ax2.set_xlabel('Monthly Income (₹)', labelpad=10)
        ax2.set_ylabel('Employees', labelpad=10)
        ax2.yaxis.grid(True, zorder=0)
        ax2.xaxis.grid(False)
        ax2.spines[:].set_visible(False)
        fig2.tight_layout()
        st.pyplot(fig2)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

    # ── CHARTS ROW 2 ──
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Attrition vs Overtime</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-sub">Overtime impact analysis</div>', unsafe_allow_html=True)

        ot_data = pd.crosstab(df['OverTime'], df['Attrition'])
        fig3, ax3 = plt.subplots(figsize=(7, 3.8))
        x = np.arange(len(ot_data.index))
        w = 0.35
        ax3.bar(x - w/2, ot_data.get('No', [0]*len(x)), w,
                label='Stays', color='#00F5D4', zorder=3, edgecolor='none')
        ax3.bar(x + w/2, ot_data.get('Yes', [0]*len(x)), w,
                label='Leaves', color='#F43F5E', zorder=3, edgecolor='none')
        ax3.set_xticks(x)
        ax3.set_xticklabels(ot_data.index)
        ax3.legend(fontsize=9, framealpha=0,
                   labelcolor='#9090B8')
        ax3.yaxis.grid(True, zorder=0)
        ax3.xaxis.grid(False)
        ax3.spines[:].set_visible(False)
        ax3.set_xlabel('Overtime', labelpad=10)
        ax3.set_ylabel('Employees', labelpad=10)
        fig3.tight_layout()
        st.pyplot(fig3)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_d:
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Job Satisfaction Score</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-sub">Score distribution (1–4)</div>', unsafe_allow_html=True)

        js = df.groupby('JobSatisfaction')['EmployeeNumber'].count()
        fig4, ax4 = plt.subplots(figsize=(7, 3.8))
        ax4.bar(js.index.astype(str), js.values,
                color=['#F43F5E', '#F59E0B', '#00F5D4', '#7C3AED'],
                width=0.5, zorder=3, edgecolor='none')

        labels_map = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}
        ax4.set_xticks(range(len(js)))
        ax4.set_xticklabels([labels_map.get(i, str(i)) for i in js.index])
        ax4.yaxis.grid(True, zorder=0)
        ax4.xaxis.grid(False)
        ax4.spines[:].set_visible(False)
        ax4.set_xlabel('Satisfaction Level', labelpad=10)
        ax4.set_ylabel('Employees', labelpad=10)
        fig4.tight_layout()
        st.pyplot(fig4)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    # ── FEATURE IMPORTANCE ──
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">AI Model — Feature Importance</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-sub">Which factors drive attrition predictions</div>', unsafe_allow_html=True)

    feature_names = ['Age', 'Monthly Income', 'Distance From Home',
                     'Job Satisfaction', 'Work-Life Balance', 'Years at Company']
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)
    palette = ['#7C3AED', '#7C3AED', '#00C4A8', '#00C4A8', '#00F5D4', '#00F5D4']

    fig5, ax5 = plt.subplots(figsize=(12, 3.2))
    bars5 = ax5.barh(
        [feature_names[i] for i in sorted_idx],
        importance[sorted_idx],
        color=[palette[i] for i in sorted_idx],
        height=0.55, edgecolor='none'
    )

    for bar, val in zip(bars5, importance[sorted_idx]):
        ax5.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', color='#9090B8', fontsize=9)

    ax5.xaxis.grid(True, zorder=0)
    ax5.yaxis.grid(False)
    ax5.spines[:].set_visible(False)
    ax5.set_xlabel('Importance Score', labelpad=10)
    fig5.tight_layout()
    st.pyplot(fig5)
    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# ── EMPLOYEE DATA ──
# =====================================================

elif '👥' in menu:

    st.markdown(f'<div class="section-title">{page_title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">{page_sub} · {len(filtered_df):,} records</div>',
                unsafe_allow_html=True)

    # Quick summary pills
    avg_inc = int(filtered_df['MonthlyIncome'].mean())
    avg_age = round(filtered_df['Age'].mean(), 1)
    attr_n  = filtered_df['Attrition'].value_counts().get('Yes', 0)

    st.markdown(f"""
    <div class="stat-row" style="margin-bottom:20px;">
        <div class="stat-pill">Avg Income <span>₹{avg_inc:,}</span></div>
        <div class="stat-pill">Avg Age <span>{avg_age}</span></div>
        <div class="stat-pill">Attritions <span>{attr_n}</span></div>
        <div class="stat-pill">Filtered: <span>{department_filter}</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(
        filtered_df.reset_index(drop=True),
        use_container_width=True,
        height=520
    )

# =====================================================
# ── PREDICTION ──
# =====================================================

elif '🤖' in menu:

    st.markdown(f'<div class="section-title">{page_title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">{page_sub}</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card" style="margin-bottom:28px;">
        <div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:600;
                    color:#F0F0FF;margin-bottom:6px;">How It Works</div>
        <div style="font-size:13px;color:#9090B8;line-height:1.7;">
            Enter the employee's profile details below. Our <strong style="color:#00F5D4;">Random Forest</strong> 
            model — trained on IBM HR Analytics data — will assess attrition risk across 
            6 key workforce dimensions and return an instant prediction.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">👤 Personal</div>', unsafe_allow_html=True)
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        age = st.number_input('Age', 18, 60, 30)
        distance = st.number_input('Distance From Home (km)', 1, 50, 5)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">💼 Career</div>', unsafe_allow_html=True)
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        income = st.number_input('Monthly Income (₹)', 1000, 200000, 50000, step=1000)
        years_company = st.number_input('Years at Company', 0, 40, 5)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">😊 Satisfaction</div>', unsafe_allow_html=True)
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

        sat_map = {1: '1 — Low', 2: '2 — Medium', 3: '3 — High', 4: '4 — Very High'}
        job_sat_label = st.select_slider(
            'Job Satisfaction', options=list(sat_map.values()),
            value='3 — High'
        )
        job_satisfaction = int(job_sat_label[0])

        wlb_label = st.select_slider(
            'Work-Life Balance', options=list(sat_map.values()),
            value='3 — High'
        )
        work_life_balance = int(wlb_label[0])
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        predict_clicked = st.button('⚡ Run Attrition Prediction')

    if predict_clicked:
        prediction = model.predict([[
            age, income, distance,
            job_satisfaction, work_life_balance, years_company
        ]])

        if prediction[0] == 1:
            st.markdown("""
            <div class="result-card result-leave">
                ⚠️ &nbsp; High Attrition Risk — Employee Likely to Leave
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card result-stay">
                ✅ &nbsp; Low Attrition Risk — Employee Likely to Stay
            </div>""", unsafe_allow_html=True)

        # Mini insight
        risk_score = round(model.predict_proba([[
            age, income, distance,
            job_satisfaction, work_life_balance, years_company
        ]])[0][1] * 100, 1)

        st.markdown(f"""
        <div style="text-align:center;margin-top:12px;
                    font-size:12px;color:#4A4A6A;letter-spacing:1px;">
            MODEL CONFIDENCE &nbsp;·&nbsp; Attrition probability: 
            <strong style="color:#F43F5E;">{risk_score}%</strong>
        </div>
        """, unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================

st.markdown("""
<div class="footer-wrap">
    <div class="footer-logo">🧬 HR · Intelligence Suite</div>
    <div class="footer-text">Built with Python, Streamlit & Machine Learning</div>
    <div class="footer-tags">
        <span class="footer-tag">Python</span>
        <span class="footer-tag">Streamlit</span>
        <span class="footer-tag">Random Forest</span>
        <span class="footer-tag">Pandas</span>
        <span class="footer-tag">Matplotlib</span>
        <span class="footer-tag">IBM HR Dataset</span>
    </div>
    <div style="margin-top:16px;font-size:11px;color:#2A2A3A;">
        © 2025 HR Intelligence Suite — Resume Project
    </div>
</div>
""", unsafe_allow_html=True)