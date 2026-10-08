def get_css():
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600&family=Inter:wght@400;500;600&display=swap');
    :root {
        --paper: #FAF8F5;
        --surface: #FFFFFF;
        --soft: #F1EDE6;
        --ink: #1F2933;
        --muted: #6B7280;
        --line: #E7E2DA;
        --teal: #0F5C5C;
        --teal-dark: #094747;
        --brick: #B4452F;
        --green: #2F6B4F;
    }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer { visibility: hidden; }
    .block-container { max-width: 980px; padding: 40px 32px 72px; }
    h1, h2, h3, h4 { font-family: 'Fraunces', serif; color: var(--ink); font-weight: 600; letter-spacing: 0; }
    h1 { font-size: 2.7rem; line-height: 1.08; }
    h2 { font-size: 2rem; line-height: 1.15; margin-top: 2rem; }
    h3 { font-size: 1.35rem; }
    p, label, .stMarkdown { line-height: 1.6; }
    .muted { color: var(--muted); }
    .eyebrow { color: var(--muted); font-size: .76rem; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; }
    .wordmark { color: var(--teal); font-weight: 600; letter-spacing: .02em; }
    .landing-screen { padding: 72px 0 0; max-width: 820px; }
    .landing-screen h1 { font-size: clamp(2.8rem, 6vw, 4.1rem); max-width: 790px; margin: 18px 0; }
    .landing-copy { max-width: 630px; color: var(--muted); font-size: 1.05rem; }
    .step-number { font: 600 2rem 'Fraunces', serif; color: var(--teal); }
    .step-title { font-weight: 600; margin: 8px 0 2px; }
    .step-copy { color: var(--muted); font-size: .9rem; }
    .landing-footer { color: var(--muted); font-size: .82rem; margin-top: 56px; }
    .topbar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 28px; }
    .section-label { color: var(--muted); font-size: .78rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; margin: 28px 0 10px; }
    .surface-card { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 20px 24px; }
    .summary-card { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 16px 20px; }
    .summary-label { color: var(--muted); font-size: .78rem; }
    .summary-value { font-weight: 600; margin-top: 3px; }
    .risk-number { font: 600 3.7rem/1 'Fraunces', serif; }
    .risk-high { color: var(--brick); }
    .risk-low { color: var(--green); }
    .risk-verdict { font-size: 1.1rem; font-weight: 600; margin-top: 12px; }
    .badge { display: inline-block; border: 1px solid var(--line); border-radius: 999px; padding: 3px 9px; font-size: .75rem; font-weight: 600; }
    .badge-applies { color: var(--brick); border-color: #D8A092; }
    .badge-no { color: var(--muted); }
    .flow-line { border-left: 1px solid var(--line); margin-left: 16px; padding-left: 28px; padding-bottom: 24px; }
    .flow-dot { background: var(--teal); border-radius: 50%; color: white; display: inline-grid; height: 32px; place-items: center; width: 32px; margin-left: -45px; margin-right: 12px; }
    .flow-title { font-weight: 600; display: inline; }
    .flow-copy { color: var(--muted); margin: 7px 0 0; }
    .stButton > button { background: var(--teal); color: white; border: 1px solid var(--teal); border-radius: 6px; min-height: 42px; font-weight: 600; }
    .stButton > button:hover { background: var(--teal-dark); border-color: var(--teal-dark); }
    .secondary-action button { background: transparent !important; color: var(--teal) !important; border: 1px solid var(--teal) !important; }
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"], .stTextArea textarea { border-radius: 6px; }
    input:focus, textarea:focus, [data-baseweb="select"] > div:focus-within { border-color: var(--teal) !important; box-shadow: 0 0 0 1px var(--teal) !important; }
    button[data-baseweb="tab"] { color: var(--muted); font-weight: 600; }
    button[data-baseweb="tab"][aria-selected="true"] { color: var(--teal); }
    div[data-baseweb="tab-highlight"] { background: var(--teal); height: 2px; }
    [data-testid="stMetricValue"] { font-family: 'Fraunces', serif; color: var(--ink); }
    [data-testid="stSidebar"] { background: var(--soft); }
    .landing-screen ~ [data-testid="stSidebar"], body:has(.landing-screen) [data-testid="stSidebar"] { display: none; }
    @media (max-width: 700px) {
        .block-container { padding: 24px 18px 56px; }
        .landing-screen { padding-top: 36px; }
        .topbar { align-items: flex-start; }
        .risk-number { font-size: 3rem; }
    }
    </style>
    """
