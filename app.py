import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="AirSense India",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
#  GLOBAL CSS — Dark Luxury Theme
# ─────────────────────────────────────────
st.markdown("""
<style>
/* Google Fonts removed for cloud compatibility */

:root {
    --bg:      #0a0f0d;
    --surface: #111a15;
    --card:    #162019;
    --border:  #1e3025;
    --accent:  #3ddc84;
    --accent2: #00b4d8;
    --text:    #e8f5ee;
    --muted:   #7a9e88;
}

html, body, .stApp {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: -apple-system, 'Segoe UI', sans-serif !important;
}

#MainMenu, footer { visibility: hidden; }

/* Hide header and toolbar */
header { visibility: hidden !important; }
header [data-testid="stToolbar"] { visibility: hidden !important; }

/* Hide default collapsed control - we replace with JS button */
[data-testid="collapsedControl"] { visibility: hidden !important; }
.block-container { padding: 1.5rem 2rem 3rem 2rem !important; max-width: 1400px; }

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
    margin-bottom: 1rem;
}
.metric-card:hover { transform: translateY(-3px); border-color: var(--accent); }
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'Segoe UI', system-ui, sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
}
.metric-sub { font-size: 0.75rem; color: var(--muted); margin-top: 0.3rem; }

.section-title {
    font-family: 'Segoe UI', system-ui, sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text);
    margin: 1.5rem 0 0.8rem 0;
}

.hero-block {
    background: linear-gradient(135deg, #111a15 0%, #0d1f18 60%, #091510 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-block::after {
    content: '';
    position: absolute;
    top: -40%; right: -10%;
    width: 320px; height: 320px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(61,220,132,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Segoe UI', system-ui, sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    color: var(--text);
    margin: 0 0 0.3rem 0;
}
.hero-subtitle { font-size: 0.9rem; color: var(--muted); }

.plant-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s, transform 0.2s;
}
.plant-card:hover { border-color: var(--accent); transform: translateX(4px); }
.plant-name {
    font-family: 'Segoe UI', system-ui, sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--accent);
    margin-bottom: 0.3rem;
}
.plant-detail { font-size: 0.82rem; color: var(--muted); line-height: 1.5; }

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 1.5rem 0;
}

.stSelectbox > div > div, .stMultiSelect > div > div {
    background: var(--card) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
}

[data-testid="stMetric"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"] { color: var(--accent) !important; font-family: 'Segoe UI', system-ui, sans-serif !important; }

div.stButton > button {
    background: linear-gradient(135deg, #1e4d2b, #2d7a45) !important;
    color: var(--accent) !important;
    border: 1px solid var(--accent) !important;
    border-radius: 10px !important;
    font-family: 'Segoe UI', system-ui, sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #2d7a45, #3ddc84) !important;
    color: #0a0f0d !important;
}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Floating hamburger button (works on Streamlit Cloud) ──
st.markdown("""
<script>
(function() {
    function injectHamburger() {
        // Remove existing if any
        var existing = document.getElementById('custom-hamburger');
        if (existing) existing.remove();

        var sidebar = document.querySelector('[data-testid="stSidebar"]');
        var isCollapsed = !sidebar || sidebar.getAttribute('aria-expanded') === 'false' ||
                          getComputedStyle(sidebar).transform.includes('matrix') ||
                          sidebar.style.display === 'none' ||
                          (sidebar.getBoundingClientRect().width < 50);

        if (isCollapsed) {
            var btn = document.createElement('div');
            btn.id = 'custom-hamburger';
            btn.innerHTML = '<div style="width:22px;height:2px;background:#3ddc84;border-radius:2px;margin:5px 0;"></div><div style="width:22px;height:2px;background:#3ddc84;border-radius:2px;margin:5px 0;"></div><div style="width:22px;height:2px;background:#3ddc84;border-radius:2px;margin:5px 0;"></div>';
            btn.style.cssText = 'position:fixed;top:14px;left:14px;width:46px;height:46px;background:#162019;border:1.5px solid #3ddc84;border-radius:10px;display:flex;flex-direction:column;align-items:center;justify-content:center;cursor:pointer;z-index:9999999;box-shadow:0 4px 20px rgba(0,0,0,0.6);transition:background 0.2s;';
            btn.onmouseover = function(){ this.style.background='#1e4d2b'; };
            btn.onmouseout  = function(){ this.style.background='#162019'; };
            btn.onclick = function() {
                // Click Streamlit's own collapsed control button
                var toggle = document.querySelector('[data-testid="collapsedControl"] button') ||
                             document.querySelector('[data-testid="collapsedControl"]');
                if (toggle) { toggle.click(); }
                setTimeout(injectHamburger, 400);
            };
            document.body.appendChild(btn);
        }
    }

    // Run on load and watch for sidebar state changes
    setTimeout(injectHamburger, 800);
    setInterval(injectHamburger, 1000);
})();
</script>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────
POLLUTANT_COLS = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene']

AQI_COLORS = {
    'Good':         '#3ddc84',
    'Satisfactory': '#118ab2',
    'Moderate':     '#06d6a0',
    'Poor':         '#ffd166',
    'Very Poor':    '#ff8c42',
    'Severe':       '#ff4d4d',
}
AQI_ORDER = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']

PLANT_DATA = {
    'PM2.5': {
        'plants': ['Bamboo', 'Neem', 'Areca Palm'],
        'emoji':  ['🎋', '🌿', '🌴'],
        'effect': 'Reduces fine particulate matter via leaf surface absorption',
        'reduction': '15–25% PM2.5 reduction per 100m² canopy',
    },
    'PM10': {
        'plants': ['Peepal', 'Ashoka', 'Holy Basil'],
        'emoji':  ['🌳', '🌲', '🌱'],
        'effect': 'Large leaf surface traps coarse particles effectively',
        'reduction': '10–20% PM10 reduction per 100m² canopy',
    },
    'CO': {
        'plants': ['Snake Plant', 'Money Plant', 'Spider Plant'],
        'emoji':  ['🪴', '🌿', '🌱'],
        'effect': 'Absorbs CO through stomatal gas exchange',
        'reduction': '8–12% CO reduction in enclosed/semi-enclosed areas',
    },
    'NO2': {
        'plants': ['Eucalyptus', 'Silver Oak', 'Gulmohar'],
        'emoji':  ['🌲', '🌳', '🌸'],
        'effect': 'High transpiration rate aids nitrogen oxide uptake',
        'reduction': '12–18% NO2 reduction in urban green belts',
    },
    'NOx': {
        'plants': ['Eucalyptus', 'Neem', 'Silver Oak'],
        'emoji':  ['🌲', '🌿', '🌳'],
        'effect': 'Effective at absorbing mixed nitrogen oxides',
        'reduction': '10–16% NOx reduction in roadside plantations',
    },
    'SO2': {
        'plants': ['Cassia', 'Marigold', 'Bougainvillea'],
        'emoji':  ['🌼', '💐', '🌺'],
        'effect': 'Sulfur-tolerant species absorb SO2 via leaf cuticle',
        'reduction': '10–15% SO2 reduction with dense planting',
    },
    'O3': {
        'plants': ['Banyan', 'Rain Tree', 'Rubber Plant'],
        'emoji':  ['🌳', '🌿', '🪴'],
        'effect': 'Large canopy provides gas exchange and shading',
        'reduction': '5–10% O3 reduction via shading and absorption',
    },
    'NH3': {
        'plants': ['Vetiver Grass', 'Water Hyacinth', 'Moringa'],
        'emoji':  ['🌾', '💧', '🌿'],
        'effect': 'Absorbs ammonia from agricultural and industrial sources',
        'reduction': '8–14% NH3 reduction near source areas',
    },
    'Benzene': {
        'plants': ['Peace Lily', 'Gerbera Daisy', 'Chrysanthemum'],
        'emoji':  ['🌸', '🌼', '💮'],
        'effect': 'Known VOC absorbers proven in NASA clean air studies',
        'reduction': '12–20% benzene reduction in indoor/semi-indoor spaces',
    },
    'Toluene': {
        'plants': ['Dracaena', 'Philodendron', 'English Ivy'],
        'emoji':  ['🪴', '🌿', '🍃'],
        'effect': 'Effective toluene and VOC metabolizers',
        'reduction': '10–18% toluene reduction in confined spaces',
    },
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(22,32,25,0.6)',
    font=dict(family='system-ui, -apple-system, sans-serif', color='#7a9e88', size=12),
    title_font=dict(family='Segoe UI, system-ui, sans-serif', color='#e8f5ee', size=15),
    xaxis=dict(gridcolor='#1e3025', linecolor='#1e3025', tickfont=dict(color='#7a9e88')),
    yaxis=dict(gridcolor='#1e3025', linecolor='#1e3025', tickfont=dict(color='#7a9e88')),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#7a9e88')),
    margin=dict(l=20, r=20, t=50, b=20),
)

# ─────────────────────────────────────────
#  DATA & MODEL
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_data.csv")

@st.cache_data
def get_dominant_pollutant(_df):
    result = {}
    for city in _df['City'].unique():
        city_df = _df[_df['City'] == city]
        valid = [c for c in POLLUTANT_COLS if city_df[c].std() > 0]
        if valid:
            result[city] = city_df[valid].corrwith(city_df['AQI']).idxmax()
        else:
            result[city] = 'PM2.5'
    return result

@st.cache_resource
def train_model(_df):
    import os
    if os.path.exists("model.pkl"):
        bundle = joblib.load("model.pkl")
        return bundle['model'], bundle['label_encoder'], bundle['accuracy']
    # Fallback: retrain if model.pkl missing
    X = _df[POLLUTANT_COLS].fillna(_df[POLLUTANT_COLS].median())
    le = LabelEncoder()
    y = le.fit_transform(_df['AQI_Bucket'])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=120, random_state=42, class_weight='balanced', n_jobs=-1)
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))
    joblib.dump({'model': clf, 'label_encoder': le, 'accuracy': acc}, 'model.pkl')
    return clf, le, acc

def get_bucket(aqi):
    if aqi <= 50: return 'Good'
    elif aqi <= 100: return 'Satisfactory'
    elif aqi <= 200: return 'Moderate'
    elif aqi <= 300: return 'Poor'
    elif aqi <= 400: return 'Very Poor'
    else: return 'Severe'

try:
    df = load_data()
except Exception as e:
    st.error(f"⚠️ Cannot load cleaned_data.csv. Make sure it's in the same folder. Error: {e}")
    st.stop()

dominant_pollutant = get_dominant_pollutant(df)
model, label_encoder, model_accuracy = train_model(df)

# ─────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 1.5rem 0;'>
        <div style='font-family:'Segoe UI',system-ui,sans-serif;font-size:1.5rem;font-weight:800;color:#3ddc84;'>AirSense</div>
        <div style='font-size:0.72rem;color:#7a9e88;letter-spacing:0.1em;text-transform:uppercase;'>India · 2015–2020</div>
    </div>
    <div style='height:1px;background:#1e3025;margin-bottom:1.2rem;'></div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["🏠  Dashboard", "🗺️  India Map", "🏙️  City Analysis", "🤖  AQI Prediction", "🌿  Plant Solutions"],
        label_visibility="collapsed"
    )

    st.markdown("""
    <div style='height:1px;background:#1e3025;margin:1.5rem 0;'></div>
    <div style='font-size:0.72rem;color:#7a9e88;line-height:1.8;'>
        <div style='font-weight:600;color:#3ddc84;margin-bottom:0.4rem;'>AQI Scale</div>
        <div>🟢 Good &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0–50</div>
        <div>🔵 Satisfactory &nbsp; 51–100</div>
        <div>🟡 Moderate &nbsp;&nbsp;&nbsp;&nbsp; 101–200</div>
        <div>🟠 Poor &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 201–300</div>
        <div>🔴 Very Poor &nbsp;&nbsp;&nbsp; 301–400</div>
        <div>⛔ Severe &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 400+</div>
    </div>
    <div style='height:1px;background:#1e3025;margin:1.5rem 0;'></div>
    <div style='font-size:0.72rem;color:#7a9e88;'>
        <b style='color:#3ddc84;'>Dataset</b><br>
        26 cities · 6 years<br>
        {n:,} records · 18 features
    </div>
    """.format(n=len(df)), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════
#  PAGE 1 — DASHBOARD
# ═══════════════════════════════════════════════════
if page == "🏠  Dashboard":

    st.markdown("""
    <div class='hero-block'>
        <div class='hero-title'>🌍 India Air Quality Overview</div>
        <div class='hero-subtitle'>National snapshot of air pollution across 26 Indian cities from 2015 to 2020 — trends, rankings, and seasonal patterns.</div>
    </div>""", unsafe_allow_html=True)

    avg_aqi   = df['AQI'].mean()
    worst     = df.groupby('City')['AQI'].mean()
    worst_city = worst.idxmax()
    worst_val  = worst.max()
    pct_bad    = (df['AQI_Bucket'].isin(['Severe', 'Very Poor'])).mean() * 100
    cleanest   = worst.idxmin()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>National Avg AQI</div>
            <div class='metric-value'>{avg_aqi:.0f}</div>
            <div class='metric-sub'>All cities combined</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Most Polluted City</div>
            <div class='metric-value' style='font-size:1.35rem;color:#ff6b6b;'>{worst_city}</div>
            <div class='metric-sub'>Avg AQI {worst_val:.0f}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Severe / Very Poor Days</div>
            <div class='metric-value' style='color:#ff8c42;'>{pct_bad:.1f}%</div>
            <div class='metric-sub'>Of all recorded days</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Cleanest City</div>
            <div class='metric-value' style='font-size:1.35rem;'>{cleanest}</div>
            <div class='metric-sub'>Avg AQI {worst.min():.0f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Year trend + Donut
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("<div class='section-title'>📈 Year-wise AQI Trend</div>", unsafe_allow_html=True)
        yt = df.groupby('Year')['AQI'].agg(['mean','min','max']).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=yt['Year'], y=yt['max'], mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=yt['Year'], y=yt['min'], fill='tonexty', mode='lines',
                                  line=dict(width=0), fillcolor='rgba(61,220,132,0.07)', showlegend=False))
        fig.add_trace(go.Scatter(x=yt['Year'], y=yt['mean'], mode='lines+markers',
                                  line=dict(color='#3ddc84', width=3),
                                  marker=dict(size=9, color='#3ddc84', line=dict(color='#0a0f0d', width=2)),
                                  name='Avg AQI'))
        fig.update_layout(**PLOTLY_LAYOUT, height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("<div class='section-title'>🗂️ AQI Breakdown</div>", unsafe_allow_html=True)
        bc = df['AQI_Bucket'].value_counts().reindex(AQI_ORDER).dropna()
        fig2 = go.Figure(go.Pie(
            labels=bc.index, values=bc.values, hole=0.58,
            marker=dict(colors=[AQI_COLORS[k] for k in bc.index], line=dict(color='#0a0f0d', width=2)),
            hovertemplate='<b>%{label}</b><br>%{value:,} days · %{percent}<extra></extra>'
        ))
        fig2.update_layout(**PLOTLY_LAYOUT, height=300,
                           annotations=[dict(text='All Cities', x=0.5, y=0.5, showarrow=False,
                                             font=dict(family='Segoe UI, system-ui, sans-serif', size=12, color='#e8f5ee'))])
        st.plotly_chart(fig2, use_container_width=True)

    # Top 10 cities
    st.markdown("<div class='section-title'>🏭 Top 10 Most Polluted Cities</div>", unsafe_allow_html=True)
    tc = df.groupby('City')['AQI'].mean().sort_values(ascending=False).head(10).reset_index()
    tc.columns = ['City', 'AQI']
    tc['color'] = tc['AQI'].apply(lambda x: '#ff4d4d' if x>300 else '#ff8c42' if x>200 else '#ffd166' if x>150 else '#3ddc84')
    fig3 = go.Figure(go.Bar(
        x=tc['City'], y=tc['AQI'],
        marker=dict(color=tc['color'], line=dict(width=0)),
        text=tc['AQI'].round(0).astype(int), textposition='outside',
        textfont=dict(color='#e8f5ee', size=11),
        hovertemplate='<b>%{x}</b><br>Avg AQI: %{y:.1f}<extra></extra>'
    ))
    fig3.update_layout(**PLOTLY_LAYOUT, height=340, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

    # Seasonal + Pollutant avg
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("<div class='section-title'>🍂 AQI by Season</div>", unsafe_allow_html=True)
        sd = df.groupby('Season')['AQI'].mean().sort_values()
        fig4 = go.Figure(go.Bar(
            x=sd.values, y=sd.index, orientation='h',
            marker=dict(color=sd.values,
                        colorscale=[[0,'#3ddc84'],[0.5,'#ffd166'],[1,'#ff4d4d']],
                        line=dict(width=0)),
            text=sd.values.round(0).astype(int), textposition='outside',
            textfont=dict(color='#e8f5ee'),
            hovertemplate='<b>%{y}</b><br>Avg AQI: %{x:.1f}<extra></extra>'
        ))
        fig4.update_layout(**PLOTLY_LAYOUT, height=280, showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    with col_d:
        st.markdown("<div class='section-title'>🧪 Average Pollutant Levels</div>", unsafe_allow_html=True)
        pm = df[POLLUTANT_COLS].mean().sort_values()
        fig5 = go.Figure(go.Bar(
            x=pm.values, y=pm.index, orientation='h',
            marker=dict(color='#00b4d8', line=dict(width=0)),
            hovertemplate='<b>%{y}</b>: %{x:.2f}<extra></extra>'
        ))
        fig5.update_layout(**PLOTLY_LAYOUT, height=280, showlegend=False)
        st.plotly_chart(fig5, use_container_width=True)


# ═══════════════════════════════════════════════════
#  PAGE 2 — CITY ANALYSIS
# ═══════════════════════════════════════════════════
elif page == "🏙️  City Analysis":

    st.markdown("""
    <div class='hero-block'>
        <div class='hero-title'>🏙️ City Deep Dive</div>
        <div class='hero-subtitle'>Full pollution profile for any city — time trends, seasonal patterns, pollutant fingerprint, and AQI distribution.</div>
    </div>""", unsafe_allow_html=True)

    cities = sorted(df['City'].unique())
    selected_city = st.selectbox("Select a City", cities, index=cities.index("Delhi"))
    city_df = df[df['City'] == selected_city]

    city_avg = city_df['AQI'].mean()
    city_max = city_df['AQI'].max()
    dom = dominant_pollutant.get(selected_city, 'PM2.5')
    rank = int(df.groupby('City')['AQI'].mean().rank(ascending=False)[selected_city])

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Average AQI</div>
            <div class='metric-value'>{city_avg:.0f}</div>
            <div class='metric-sub'>{get_bucket(city_avg)}</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Peak AQI</div>
            <div class='metric-value' style='color:#ff6b6b;'>{city_max:.0f}</div>
            <div class='metric-sub'>Recorded max</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>National Rank</div>
            <div class='metric-value'>#{rank}</div>
            <div class='metric-sub'>of 26 cities (1=worst)</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Dominant Pollutant</div>
            <div class='metric-value' style='font-size:1.3rem;'>{dom}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Time series
    st.markdown("<div class='section-title'>📅 Monthly AQI Trend</div>", unsafe_allow_html=True)
    ts = city_df.groupby(['Year','Month'])['AQI'].mean().reset_index()
    ts['Date'] = pd.to_datetime(ts[['Year','Month']].assign(day=1))
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=ts['Date'], y=ts['AQI'], mode='lines',
        line=dict(color='#3ddc84', width=2.5),
        fill='tozeroy', fillcolor='rgba(61,220,132,0.07)',
        hovertemplate='%{x|%b %Y}<br>AQI: %{y:.0f}<extra></extra>'
    ))
    for thresh, col, lbl in [(400,'rgba(255,77,77,0.1)','Severe'),
                              (300,'rgba(255,140,66,0.08)','Very Poor'),
                              (200,'rgba(255,209,102,0.06)','Poor')]:
        fig_ts.add_hrect(y0=thresh, y1=700, fillcolor=col, line_width=0,
                         annotation_text=lbl, annotation_position="right",
                         annotation_font=dict(color='#7a9e88', size=10))
    fig_ts.update_layout(**PLOTLY_LAYOUT, height=300)
    st.plotly_chart(fig_ts, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-title'>📆 Year-wise Avg AQI</div>", unsafe_allow_html=True)
        yr = city_df.groupby('Year')['AQI'].mean().reset_index()
        fig_yr = go.Figure(go.Bar(
            x=yr['Year'], y=yr['AQI'],
            marker=dict(color=yr['AQI'],
                        colorscale=[[0,'#3ddc84'],[0.5,'#ffd166'],[1,'#ff4d4d']],
                        line=dict(width=0)),
            text=yr['AQI'].round(0).astype(int), textposition='outside',
            textfont=dict(color='#e8f5ee'),
            hovertemplate='<b>%{x}</b><br>%{y:.1f}<extra></extra>'
        ))
        fig_yr.update_layout(**PLOTLY_LAYOUT, height=280, showlegend=False)
        st.plotly_chart(fig_yr, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>📊 AQI Category Distribution</div>", unsafe_allow_html=True)
        bkt = city_df['AQI_Bucket'].value_counts().reindex(AQI_ORDER).dropna()
        fig_bkt = go.Figure(go.Bar(
            x=bkt.index, y=bkt.values,
            marker=dict(color=[AQI_COLORS[k] for k in bkt.index], line=dict(width=0)),
            hovertemplate='<b>%{x}</b><br>%{y} days<extra></extra>'
        ))
        fig_bkt.update_layout(**PLOTLY_LAYOUT, height=280, showlegend=False)
        st.plotly_chart(fig_bkt, use_container_width=True)

    # Pollutant radar
    st.markdown("<div class='section-title'>🧪 Pollutant Fingerprint (Radar)</div>", unsafe_allow_html=True)
    pol_means = city_df[POLLUTANT_COLS].mean()
    pol_norm = (pol_means - pol_means.min()) / (pol_means.max() - pol_means.min() + 1e-9)
    cats = pol_norm.index.tolist()
    vals = pol_norm.values.tolist()
    fig_radar = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]], theta=cats + [cats[0]],
        fill='toself', fillcolor='rgba(61,220,132,0.12)',
        line=dict(color='#3ddc84', width=2),
        marker=dict(size=5, color='#3ddc84')
    ))
    fig_radar.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='system-ui, -apple-system, sans-serif', color='#7a9e88'),
        title_font=dict(family='Segoe UI, system-ui, sans-serif', color='#e8f5ee'),
        polar=dict(
            bgcolor='rgba(22,32,25,0.4)',
            radialaxis=dict(visible=True, gridcolor='#1e3025', tickfont=dict(color='#7a9e88', size=9)),
            angularaxis=dict(gridcolor='#1e3025', tickfont=dict(color='#e8f5ee', size=11))
        ),
        margin=dict(l=40, r=40, t=40, b=40),
        height=400
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Pollutant correlation heatmap
    st.markdown("<div class='section-title'>🔗 Pollutant × AQI Correlation</div>", unsafe_allow_html=True)
    corr_data = city_df[POLLUTANT_COLS + ['AQI']].corr()
    fig_heat = go.Figure(go.Heatmap(
        z=corr_data.values,
        x=corr_data.columns.tolist(),
        y=corr_data.index.tolist(),
        colorscale=[[0,'#0a0f0d'],[0.5,'#1e4d2b'],[1,'#3ddc84']],
        text=corr_data.round(2).values,
        texttemplate='%{text}',
        textfont=dict(size=9, color='#e8f5ee'),
        hovertemplate='%{y} × %{x}<br>r = %{z:.3f}<extra></extra>'
    ))
    fig_heat.update_layout(**{k:v for k,v in PLOTLY_LAYOUT.items() if k not in ['xaxis','yaxis']},
                           height=420,
                           xaxis=dict(tickfont=dict(color='#7a9e88', size=10)),
                           yaxis=dict(tickfont=dict(color='#7a9e88', size=10)))
    st.plotly_chart(fig_heat, use_container_width=True)


# ═══════════════════════════════════════════════════
#  PAGE 3 — AQI PREDICTION
# ═══════════════════════════════════════════════════
elif page == "🤖  AQI Prediction":

    st.markdown(f"""
    <div class='hero-block'>
        <div class='hero-title'>🤖 AQI Category Predictor</div>
        <div class='hero-subtitle'>Random Forest model trained on {len(df):,} records. Adjust pollutant sliders below to predict the AQI category.
        &nbsp; <span style='color:#3ddc84; font-weight:600;'>Model Accuracy: {model_accuracy*100:.1f}%</span></div>
    </div>""", unsafe_allow_html=True)

    # Feature importance
    imp = pd.Series(model.feature_importances_, index=POLLUTANT_COLS).sort_values()
    fig_fi = go.Figure(go.Bar(
        x=imp.values, y=imp.index, orientation='h',
        marker=dict(color=imp.values,
                    colorscale=[[0,'#1e3025'],[0.5,'#00b4d8'],[1,'#3ddc84']],
                    line=dict(width=0)),
        hovertemplate='<b>%{y}</b>: %{x:.3f}<extra></extra>'
    ))
    fig_fi.update_layout(**PLOTLY_LAYOUT, height=300, title='Feature Importance (Random Forest)')
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🎚️ Set Pollutant Concentrations</div>", unsafe_allow_html=True)
    st.caption("Defaults are dataset medians. Use city analysis page to look up real values for any city.")

    medians = df[POLLUTANT_COLS].median()
    inputs = {}
    col1, col2 = st.columns(2)
    for i, col in enumerate(POLLUTANT_COLS):
        target = col1 if i % 2 == 0 else col2
        with target:
            inputs[col] = st.slider(
                col,
                min_value=0.0,
                max_value=float(df[col].quantile(0.99)),
                value=float(medians[col]),
                step=0.1
            )

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    if st.button("🔍  Predict AQI Category", use_container_width=True):
        input_df = pd.DataFrame([inputs])
        pred_enc = model.predict(input_df)[0]
        proba    = model.predict_proba(input_df)[0]
        pred_lbl = label_encoder.inverse_transform([pred_enc])[0]
        pred_col = AQI_COLORS.get(pred_lbl, '#3ddc84')

        st.markdown(f"""
        <div style='background:{pred_col}15;border:1px solid {pred_col}55;border-radius:16px;
                    padding:2rem;margin-top:1.5rem;text-align:center;'>
            <div style='font-size:0.75rem;letter-spacing:0.15em;text-transform:uppercase;color:{pred_col};margin-bottom:0.5rem;'>
                Predicted AQI Category
            </div>
            <div style='font-family:Syne;font-size:3rem;font-weight:800;color:{pred_col};line-height:1;'>
                {pred_lbl}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-title' style='margin-top:1.5rem;'>📊 Class Probabilities</div>", unsafe_allow_html=True)
        classes = label_encoder.classes_
        prob_df = pd.DataFrame({'Category': classes, 'Prob': proba}).sort_values('Prob')
        fig_prob = go.Figure(go.Bar(
            x=prob_df['Prob'], y=prob_df['Category'], orientation='h',
            marker=dict(color=[AQI_COLORS.get(c,'#3ddc84') for c in prob_df['Category']], line=dict(width=0)),
            text=[f"{p*100:.1f}%" for p in prob_df['Prob']],
            textposition='outside', textfont=dict(color='#e8f5ee'),
            hovertemplate='<b>%{y}</b>: %{x:.1%}<extra></extra>'
        ))
        prob_layout = {**PLOTLY_LAYOUT}
        prob_layout['xaxis'] = dict(tickformat='.0%', gridcolor='#1e3025', linecolor='#1e3025', tickfont=dict(color='#7a9e88'))
        fig_prob.update_layout(**prob_layout, height=280, showlegend=False)
        st.plotly_chart(fig_prob, use_container_width=True)


# ═══════════════════════════════════════════════════
#  PAGE 4 — PLANT SOLUTIONS
# ═══════════════════════════════════════════════════
elif page == "🌿  Plant Solutions":

    st.markdown("""
    <div class='hero-block'>
        <div class='hero-title'>🌿 Plant-Based Solution System</div>
        <div class='hero-subtitle'>Data-driven plant recommendations based on each city's dominant pollutant, with AQI impact simulation from green cover expansion.</div>
    </div>""", unsafe_allow_html=True)

    cities = sorted(df['City'].unique())
    col_sel, col_sl = st.columns([2, 2])
    with col_sel:
        selected_city = st.selectbox("Select City", cities, index=cities.index("Delhi"))
    with col_sl:
        green_pct = st.slider("Green Cover Increase (%)", 5, 50, 10, 5,
                              help="Assumes 10% green cover → ~3% pollution reduction (conservative estimate from environmental literature)")

    city_df = df[df['City'] == selected_city]
    dom = dominant_pollutant.get(selected_city, 'PM2.5')
    cur_aqi = city_df['AQI'].mean()
    pol_red = green_pct * 0.3
    new_aqi = cur_aqi * (1 - pol_red / 100)
    improvement = cur_aqi - new_aqi
    cur_bkt = get_bucket(cur_aqi)
    new_bkt = get_bucket(new_aqi)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Current Avg AQI</div>
            <div class='metric-value' style='color:{AQI_COLORS.get(cur_bkt,"#ff6b6b")};'>{cur_aqi:.0f}</div>
            <div class='metric-sub'>{cur_bkt}</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Projected AQI</div>
            <div class='metric-value' style='color:{AQI_COLORS.get(new_bkt,"#3ddc84")};'>{new_aqi:.0f}</div>
            <div class='metric-sub'>{new_bkt}</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>AQI Improvement</div>
            <div class='metric-value' style='color:#3ddc84;'>−{improvement:.0f}</div>
            <div class='metric-sub'>{pol_red:.1f}% pollution reduction</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Dominant Pollutant</div>
            <div class='metric-value' style='font-size:1.3rem;'>{dom}</div>
            <div class='metric-sub'>Primary target pollutant</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Gauges
    st.markdown("<div class='section-title'>📉 AQI Impact Simulation</div>", unsafe_allow_html=True)

    def make_gauge(value, title, color):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={'text': title, 'font': {'family': 'Segoe UI, system-ui, sans-serif', 'color': '#e8f5ee', 'size': 13}},
            number={'font': {'family': 'Segoe UI, system-ui, sans-serif', 'color': color, 'size': 34}},
            gauge={
                'axis': {'range': [0, 500], 'tickcolor': '#7a9e88', 'tickfont': {'color': '#7a9e88', 'size': 9}},
                'bar': {'color': color},
                'bgcolor': '#162019',
                'borderwidth': 1, 'bordercolor': '#1e3025',
                'steps': [
                    {'range': [0,50],    'color': 'rgba(61,220,132,0.1)'},
                    {'range': [50,100],  'color': 'rgba(17,138,178,0.1)'},
                    {'range': [100,200], 'color': 'rgba(6,214,160,0.1)'},
                    {'range': [200,300], 'color': 'rgba(255,209,102,0.1)'},
                    {'range': [300,400], 'color': 'rgba(255,140,66,0.1)'},
                    {'range': [400,500], 'color': 'rgba(255,77,77,0.1)'},
                ],
                'threshold': {'line': {'color': color, 'width': 3}, 'thickness': 0.8, 'value': value}
            }
        ))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                          font=dict(family='system-ui, -apple-system, sans-serif', color='#7a9e88'),
                          margin=dict(l=30,r=30,t=50,b=10), height=280)
        return fig

    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(make_gauge(cur_aqi, "Current AQI", AQI_COLORS.get(cur_bkt,'#ff6b6b')), use_container_width=True)
    with g2:
        st.plotly_chart(make_gauge(new_aqi, f"After +{green_pct}% Green Cover", AQI_COLORS.get(new_bkt,'#3ddc84')), use_container_width=True)

    # Before/after pollutant comparison
    pol_cur = city_df[POLLUTANT_COLS].mean()
    pol_new = pol_cur * (1 - pol_red / 100)
    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(name='Current', x=POLLUTANT_COLS, y=pol_cur,
                              marker=dict(color='rgba(255,107,107,0.65)', line=dict(width=0))))
    fig_cmp.add_trace(go.Bar(name='After Plantation', x=POLLUTANT_COLS, y=pol_new,
                              marker=dict(color='rgba(61,220,132,0.65)', line=dict(width=0))))
    fig_cmp.update_layout(**PLOTLY_LAYOUT, barmode='group', height=310,
                           title='Pollutant Levels: Current vs Projected')
    st.plotly_chart(fig_cmp, use_container_width=True)

    # Plant recommendations
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🌱 Recommended Plants for This City</div>", unsafe_allow_html=True)
    pdata = PLANT_DATA.get(dom, PLANT_DATA['PM2.5'])

    st.markdown(f"""
    <div style='background:rgba(61,220,132,0.06);border:1px solid rgba(61,220,132,0.18);
                border-radius:12px;padding:1rem 1.4rem;margin-bottom:1.2rem;'>
        <span style='font-size:0.72rem;color:#7a9e88;text-transform:uppercase;letter-spacing:0.1em;'>
            Targeting dominant pollutant:
        </span>
        <span style='font-family:Syne;font-size:1.05rem;font-weight:700;color:#3ddc84;margin-left:0.6rem;'>{dom}</span>
        <br><span style='font-size:0.82rem;color:#7a9e88;margin-top:0.3rem;display:block;'>{pdata["effect"]}</span>
    </div>
    """, unsafe_allow_html=True)

    p1, p2, p3 = st.columns(3)
    for col, (plant, emoji) in zip([p1, p2, p3], zip(pdata['plants'], pdata['emoji'])):
        with col:
            st.markdown(f"""
            <div class='plant-card'>
                <div style='font-size:2.2rem;margin-bottom:0.5rem;'>{emoji}</div>
                <div class='plant-name'>{plant}</div>
                <div class='plant-detail'>{pdata['reduction']}</div>
                <div class='plant-detail' style='color:#3ddc84;margin-top:0.4rem;font-size:0.78rem;'>
                    Best for {dom} absorption
                </div>
            </div>""", unsafe_allow_html=True)

    # Full reference table
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📋 Pollutant → Plant Reference Guide</div>", unsafe_allow_html=True)
    ref = []
    for pol, d in PLANT_DATA.items():
        ref.append({
            'Pollutant': pol,
            'Plants': '  ·  '.join([f"{e} {p}" for e, p in zip(d['emoji'], d['plants'])]),
            'Est. Reduction': d['reduction'],
            'Mechanism': d['effect']
        })
    ref_df = pd.DataFrame(ref)
    st.dataframe(ref_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════
#  PAGE: INDIA MAP VIEW
# ═══════════════════════════════════════════════════
elif page == "🗺️  India Map":

    # City coordinates (lat, lon)
    CITY_COORDS = {
        'Ahmedabad':          (23.0225, 72.5714),
        'Aizawl':             (23.7307, 92.7173),
        'Amaravati':          (16.5727, 80.3570),
        'Amritsar':           (31.6340, 74.8723),
        'Bengaluru':          (12.9716, 77.5946),
        'Bhopal':             (23.2599, 77.4126),
        'Brajrajnagar':       (21.8167, 83.9167),
        'Chandigarh':         (30.7333, 76.7794),
        'Chennai':            (13.0827, 80.2707),
        'Coimbatore':         (11.0168, 76.9558),
        'Delhi':              (28.6139, 77.2090),
        'Ernakulam':          (9.9816,  76.2999),
        'Gurugram':           (28.4595, 77.0266),
        'Guwahati':           (26.1445, 91.7362),
        'Hyderabad':          (17.3850, 78.4867),
        'Jaipur':             (26.9124, 75.7873),
        'Jorapokhar':         (23.6800, 86.4200),
        'Kochi':              (9.9312,  76.2673),
        'Kolkata':            (22.5726, 88.3639),
        'Lucknow':            (26.8467, 80.9462),
        'Mumbai':             (19.0760, 72.8777),
        'Patna':              (25.5941, 85.1376),
        'Shillong':           (25.5788, 91.8933),
        'Talcher':            (20.9500, 85.2333),
        'Thiruvananthapuram': (8.5241,  76.9366),
        'Visakhapatnam':      (17.6868, 83.2185),
    }

    st.markdown("""
    <div class='hero-block'>
        <div class='hero-title'>🗺️ India AQI Map</div>
        <div class='hero-subtitle'>Interactive map of air quality across 26 Indian cities. Bubble size = average AQI intensity. Color = AQI category. Filter by year and season below.</div>
    </div>""", unsafe_allow_html=True)

    # ── Filters ──
    f1, f2, f3 = st.columns(3)
    with f1:
        year_opts = ['All Years'] + sorted(df['Year'].unique().tolist())
        sel_year  = st.selectbox("Filter by Year", year_opts)
    with f2:
        season_opts = ['All Seasons'] + sorted(df['Season'].unique().tolist())
        sel_season  = st.selectbox("Filter by Season", season_opts)
    with f3:
        metric_choice = st.selectbox("Map Metric", ["Average AQI", "Peak AQI", "% Severe Days"])

    # ── Filter data ──
    map_df = df.copy()
    if sel_year != 'All Years':
        map_df = map_df[map_df['Year'] == int(sel_year)]
    if sel_season != 'All Seasons':
        map_df = map_df[map_df['Season'] == sel_season]

    # ── Aggregate per city ──
    city_agg = map_df.groupby('City').agg(
        avg_aqi   = ('AQI', 'mean'),
        max_aqi   = ('AQI', 'max'),
        min_aqi   = ('AQI', 'min'),
        records   = ('AQI', 'count'),
        dom_bucket= ('AQI_Bucket', lambda x: x.value_counts().index[0]),
    ).reset_index()
    city_agg['pct_severe'] = map_df[map_df['AQI_Bucket'].isin(['Severe','Very Poor'])]\
        .groupby('City').size().reindex(city_agg['City']).fillna(0).values / city_agg['records'] * 100

    # Add coords
    city_agg['lat'] = city_agg['City'].map(lambda c: CITY_COORDS.get(c, (20, 78))[0])
    city_agg['lon'] = city_agg['City'].map(lambda c: CITY_COORDS.get(c, (20, 78))[1])

    # Metric to display
    if metric_choice == "Average AQI":
        city_agg['metric'] = city_agg['avg_aqi'].round(1)
        size_col, color_col, label_suffix = 'metric', 'metric', ' AQI'
    elif metric_choice == "Peak AQI":
        city_agg['metric'] = city_agg['max_aqi'].round(1)
        size_col, color_col, label_suffix = 'metric', 'metric', ' (Peak AQI)'
    else:
        city_agg['metric'] = city_agg['pct_severe'].round(1)
        size_col, color_col, label_suffix = 'metric', 'metric', '% Severe Days'

    city_agg['bucket_color'] = city_agg['dom_bucket'].map(AQI_COLORS)

    # ── Main scatter map ──
    fig_map = go.Figure()

    # Draw bubbles per AQI bucket for legend grouping
    for bucket in AQI_ORDER:
        subset = city_agg[city_agg['dom_bucket'] == bucket]
        if subset.empty:
            continue
        fig_map.add_trace(go.Scattergeo(
            lat=subset['lat'],
            lon=subset['lon'],
            mode='markers+text',
            name=bucket,
            marker=dict(
                size=subset['metric'].clip(lower=20) / 4,
                sizemode='area',
                color=AQI_COLORS[bucket],
                opacity=0.85,
                line=dict(color='#0a0f0d', width=1.5),
            ),
            text=subset['City'],
            textposition='top center',
            textfont=dict(family='system-ui, -apple-system, sans-serif', size=10, color='#e8f5ee'),
            customdata=subset[['avg_aqi','max_aqi','pct_severe','records','dom_bucket']].values,
            hovertemplate=(
                '<b>%{text}</b><br>'
                'Avg AQI: %{customdata[0]:.1f}<br>'
                'Peak AQI: %{customdata[1]:.0f}<br>'
                'Severe Days: %{customdata[2]:.1f}%<br>'
                'Records: %{customdata[3]:,}<br>'
                'Dominant: %{customdata[4]}'
                '<extra></extra>'
            )
        ))

    fig_map.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        geo=dict(
            scope='asia',
            projection_type='mercator',
            center=dict(lat=22, lon=80),
            lonaxis=dict(range=[67, 98]),
            lataxis=dict(range=[6, 38]),
            bgcolor='rgba(10,15,13,0)',
            showland=True,      landcolor='#111a15',
            showocean=True,     oceancolor='#0a0f0d',
            showlakes=True,     lakecolor='#0d1f18',
            showrivers=True,    rivercolor='#1e3025',
            showcountries=True, countrycolor='#2d5c3a',
            showcoastlines=True,coastlinecolor='#2d5c3a',
            showsubunits=True,  subunitcolor='#1e3025',
            subunitwidth=0.5,
        ),
        legend=dict(
            title=dict(text='AQI Category', font=dict(color='#e8f5ee', family='Segoe UI, system-ui, sans-serif', size=12)),
            bgcolor='rgba(17,26,21,0.9)',
            bordercolor='#1e3025',
            borderwidth=1,
            font=dict(color='#7a9e88', family='system-ui, -apple-system, sans-serif', size=11),
            x=0.01, y=0.98,
        ),
        font=dict(family='system-ui, -apple-system, sans-serif', color='#7a9e88'),
        margin=dict(l=0, r=0, t=10, b=0),
        height=620,
    )

    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── KPI strip below map ──
    st.markdown("<div class='section-title'>📊 City Rankings — Worst to Best</div>", unsafe_allow_html=True)

    ranked = city_agg.sort_values('avg_aqi', ascending=False).reset_index(drop=True)
    ranked['rank'] = ranked.index + 1

    # Build a colour-coded ranking table with HTML
    rows_html = ""
    for _, row in ranked.iterrows():
        c  = AQI_COLORS.get(row['dom_bucket'], '#7a9e88')
        bar_w = int(min(row['avg_aqi'] / city_agg['avg_aqi'].max() * 100, 100))
        rank_num  = int(row['rank'])
        city_name = row['City']
        avg_aqi   = round(row['avg_aqi'])
        bucket    = row['dom_bucket']
        pct_sev   = round(row['pct_severe'], 1)
        rows_html += (
            "<div style='display:flex;align-items:center;gap:1rem;padding:0.6rem 1rem;"
            "border-bottom:1px solid #1e3025;'>"
            f"<div style='font-family:Syne;font-size:0.85rem;font-weight:700;color:#7a9e88;width:28px;text-align:right;'>#{rank_num}</div>"
            f"<div style='font-family:-apple-system,'Segoe UI',sans-serif;font-size:0.92rem;color:#e8f5ee;width:180px;'>{city_name}</div>"
            f"<div style='flex:1;background:#1e3025;border-radius:4px;height:8px;overflow:hidden;'>"
            f"<div style='width:{bar_w}%;height:100%;background:{c};border-radius:4px;'></div></div>"
            f"<div style='font-family:Syne;font-size:0.95rem;font-weight:700;color:{c};width:60px;text-align:right;'>{avg_aqi}</div>"
            f"<div style='font-size:0.75rem;color:#7a9e88;width:110px;text-align:center;"
            f"background:rgba(61,220,132,0.06);border:1px solid #1e3025;"
            f"border-radius:20px;padding:0.15rem 0.5rem;'>{bucket}</div>"
            f"<div style='font-size:0.75rem;color:#7a9e88;width:90px;text-align:right;'>{pct_sev}% severe</div>"
            "</div>"
        )

    st.markdown(f"""
    <div style='background:#111a15;border:1px solid #1e3025;border-radius:14px;overflow:hidden;'>
        <div style='display:flex;align-items:center;gap:1rem;padding:0.6rem 1rem;
                    border-bottom:1px solid #2d5c3a;background:#162019;'>
            <div style='font-size:0.7rem;color:#7a9e88;text-transform:uppercase;letter-spacing:0.1em;width:28px;text-align:right;'>Rank</div>
            <div style='font-size:0.7rem;color:#7a9e88;text-transform:uppercase;letter-spacing:0.1em;width:180px;'>City</div>
            <div style='font-size:0.7rem;color:#7a9e88;text-transform:uppercase;letter-spacing:0.1em;flex:1;'>AQI Bar</div>
            <div style='font-size:0.7rem;color:#7a9e88;text-transform:uppercase;letter-spacing:0.1em;width:60px;text-align:right;'>Avg AQI</div>
            <div style='font-size:0.7rem;color:#7a9e88;text-transform:uppercase;letter-spacing:0.1em;width:110px;text-align:center;'>Category</div>
            <div style='font-size:0.7rem;color:#7a9e88;text-transform:uppercase;letter-spacing:0.1em;width:90px;text-align:right;'>Severe %</div>
        </div>
        {rows_html}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    # ── Year-over-year map comparison ──
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📅 How Has Each City's AQI Changed Over the Years?</div>", unsafe_allow_html=True)

    year_city = df.groupby(['City','Year'])['AQI'].mean().reset_index()
    year_city.columns = ['City','Year','AQI']

    fig_line = go.Figure()
    for city in sorted(df['City'].unique()):
        cdata = year_city[year_city['City'] == city]
        dom_bkt = city_agg[city_agg['City'] == city]['dom_bucket'].values
        col = AQI_COLORS.get(dom_bkt[0], '#7a9e88') if len(dom_bkt) else '#7a9e88'
        fig_line.add_trace(go.Scatter(
            x=cdata['Year'], y=cdata['AQI'],
            mode='lines', name=city,
            line=dict(color=col, width=1.5),
            opacity=0.7,
            hovertemplate=f'<b>{city}</b><br>Year: %{{x}}<br>AQI: %{{y:.1f}}<extra></extra>'
        ))

    line_layout = {**PLOTLY_LAYOUT}
    line_layout['legend'] = dict(
        bgcolor='rgba(17,26,21,0.9)', bordercolor='#1e3025', borderwidth=1,
        font=dict(color='#7a9e88', size=9),
        x=1.01, y=1
    )
    line_layout['margin'] = dict(l=20, r=160, t=50, b=20)
    fig_line.update_layout(**line_layout, height=380,
                           title='All Cities — Year-wise AQI Trend',
                           hovermode='x unified')
    st.plotly_chart(fig_line, use_container_width=True)
