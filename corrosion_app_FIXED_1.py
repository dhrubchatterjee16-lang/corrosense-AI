"""
=============================================================================
  CORROSION RATE PREDICTION — STREAMLIT APP
  Deep Learning model with full visualization dashboard
  
  HOW TO RUN:
    pip install streamlit plotly scikit-learn pandas numpy
    streamlit run corrosion_app_FIXED.py
=============================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import re
import warnings
import io

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CorroSense AI",
    page_icon="⚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  —  WILD EDITION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');

/* ── Scrolling particle canvas behind everything ── */
body { background: #020409 !important; }
.main { background: transparent !important; }
.block-container { padding-top: 0.5rem; padding-bottom: 2rem; position: relative; z-index: 1; }

/* ── Animated background grid ── */
.stApp::before {
    content: '';
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background-image:
        linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    animation: gridScroll 20s linear infinite;
    pointer-events: none; z-index: 0;
}
@keyframes gridScroll {
    0%   { background-position: 0 0; }
    100% { background-position: 40px 40px; }
}

/* ── Glowing scanline sweep ── */
.stApp::after {
    content: '';
    position: fixed; top: -100%; left: 0; width: 100%; height: 6px;
    background: linear-gradient(transparent, rgba(0,212,255,0.4), transparent);
    animation: scanline 6s linear infinite;
    pointer-events: none; z-index: 0;
}
@keyframes scanline {
    0%   { top: -2%; }
    100% { top: 102%; }
}

/* ── EPIC TITLE ── */
.epic-title {
    font-family: 'Orbitron', monospace;
    font-size: clamp(28px, 4vw, 56px);
    font-weight: 900;
    text-align: center;
    letter-spacing: 4px;
    text-transform: uppercase;
    background: linear-gradient(90deg, #FF0080, #FF6B35, #FFD700, #39FF14, #00D4FF, #BF5FFF, #FF0080);
    background-size: 400% 100%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: rainbowShift 3s linear infinite, titlePulse 2s ease-in-out infinite alternate;
    text-shadow: none;
    margin: 0; padding: 10px 0 4px;
}
@keyframes rainbowShift {
    0%   { background-position: 0% 50%; }
    100% { background-position: 400% 50%; }
}
@keyframes titlePulse {
    from { filter: brightness(1) drop-shadow(0 0 8px rgba(0,212,255,0.5)); }
    to   { filter: brightness(1.3) drop-shadow(0 0 24px rgba(191,95,255,0.9)); }
}

.title-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 13px;
    text-align: center;
    color: #00D4FF;
    letter-spacing: 8px;
    text-transform: uppercase;
    animation: blink 1.5s step-end infinite;
    margin-bottom: 6px;
}
@keyframes blink {
    0%,100% { opacity: 1; } 50% { opacity: 0.3; }
}

.title-wrapper {
    background: linear-gradient(180deg, rgba(0,0,0,0.8) 0%, transparent 100%);
    border-bottom: 1px solid rgba(0,212,255,0.2);
    padding: 18px 0 14px;
    margin-bottom: 18px;
    position: relative;
    overflow: hidden;
}
.title-wrapper::before {
    content: '';
    position: absolute; bottom: 0; left: -100%; width: 300%; height: 1px;
    background: linear-gradient(90deg, transparent, #00D4FF, #BF5FFF, #FF0080, transparent);
    animation: borderRun 3s linear infinite;
}
@keyframes borderRun {
    0%   { left: -100%; }
    100% { left: 100%; }
}

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #0D1520, #161B22);
    border: 1px solid rgba(0,212,255,0.3);
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
    margin-bottom: 10px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card::after {
    content: '';
    position: absolute; top: -50%; left: -50%; width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(0,212,255,0.05) 0%, transparent 60%);
    animation: cardGlow 4s ease-in-out infinite alternate;
}
@keyframes cardGlow {
    from { transform: scale(0.8); opacity: 0.5; }
    to   { transform: scale(1.2); opacity: 1; }
}
.metric-card .label { font-size: 11px; color: #8B9099; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; }
.metric-card .value { font-size: 26px; font-weight: 700; margin: 6px 0 2px; font-family: 'Orbitron', monospace; }
.metric-card .note  { font-size: 10px; color: #555; }
.metric-card.blue   .value { color: #00D4FF; text-shadow: 0 0 12px rgba(0,212,255,0.8); }
.metric-card.orange .value { color: #FF6B35; text-shadow: 0 0 12px rgba(255,107,53,0.8); }
.metric-card.green  .value { color: #39FF14; text-shadow: 0 0 12px rgba(57,255,20,0.8); }
.metric-card.purple .value { color: #BF5FFF; text-shadow: 0 0 12px rgba(191,95,255,0.8); }
.metric-card.gold   .value { color: #FFD700; text-shadow: 0 0 12px rgba(255,215,0,0.8); }

/* ── Section headers ── */
.section-header {
    background: linear-gradient(90deg, rgba(0,212,255,0.15), rgba(191,95,255,0.1), transparent);
    border-left: 3px solid #00D4FF;
    border-top: 1px solid rgba(0,212,255,0.1);
    padding: 10px 18px;
    border-radius: 0 8px 8px 0;
    margin: 22px 0 14px;
    font-size: 15px;
    font-weight: 700;
    font-family: 'Orbitron', monospace;
    letter-spacing: 2px;
    color: white;
    text-transform: uppercase;
    position: relative;
    overflow: hidden;
}
.section-header::after {
    content: '';
    position: absolute; top: 0; left: -100%; width: 60%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(0,212,255,0.15), transparent);
    animation: headerShine 3s ease-in-out infinite;
}
@keyframes headerShine {
    0%   { left: -100%; }
    100% { left: 200%; }
}

/* ── TRAIN BUTTON — NUCLEAR EDITION ── */
.stButton > button {
    background: linear-gradient(135deg, #FF0080, #FF6B35, #FFD700) !important;
    background-size: 200% 200% !important;
    color: #000 !important;
    font-weight: 900 !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 13px !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 14px 28px !important;
    width: 100% !important;
    cursor: pointer !important;
    position: relative !important;
    overflow: hidden !important;
    animation: btnPulse 2s ease-in-out infinite, btnGrad 3s linear infinite !important;
    box-shadow: 0 0 20px rgba(255,0,128,0.5), 0 0 40px rgba(255,107,53,0.3) !important;
    transition: all 0.3s !important;
}
@keyframes btnPulse {
    0%,100% { box-shadow: 0 0 20px rgba(255,0,128,0.5), 0 0 40px rgba(255,107,53,0.3); transform: scale(1); }
    50%      { box-shadow: 0 0 35px rgba(255,0,128,0.9), 0 0 70px rgba(255,215,0,0.5); transform: scale(1.02); }
}
@keyframes btnGrad {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.stButton > button::before {
    content: '';
    position: absolute; top: -50%; left: -75%; width: 50%; height: 200%;
    background: rgba(255,255,255,0.2);
    transform: skewX(-20deg);
    animation: btnShine 2.5s ease-in-out infinite;
}
@keyframes btnShine {
    0%   { left: -75%; }
    100% { left: 150%; }
}
.stButton > button:hover {
    transform: scale(1.05) !important;
    box-shadow: 0 0 50px rgba(255,0,128,1), 0 0 100px rgba(255,215,0,0.6) !important;
}

/* ── Pred box ── */
.pred-box {
    background: linear-gradient(135deg, #0A0F1A, #161B22);
    border: 2px solid transparent;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    margin-top: 12px;
    position: relative;
    background-clip: padding-box;
    animation: predBorderPulse 2s ease-in-out infinite;
}
@keyframes predBorderPulse {
    0%,100% { box-shadow: 0 0 15px rgba(0,212,255,0.4), inset 0 0 15px rgba(0,212,255,0.05); }
    50%      { box-shadow: 0 0 35px rgba(0,212,255,0.8), inset 0 0 30px rgba(0,212,255,0.1); }
}
.pred-box .rate  { font-size: 52px; font-weight: 800; color: #00D4FF; font-family: 'Orbitron', monospace; text-shadow: 0 0 20px rgba(0,212,255,0.8); }
.pred-box .unit  { font-size: 13px; color: #8B9099; margin-top: -4px; letter-spacing: 3px; }
.pred-box .badge { font-size: 22px; margin-top: 10px; font-weight: 700; }

/* ── Info banner ── */
.info-banner {
    background: linear-gradient(135deg, rgba(0,212,255,0.05), rgba(191,95,255,0.05));
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 10px;
    padding: 14px 18px;
    color: #8B9099;
    font-size: 13px;
    margin-bottom: 16px;
    font-family: 'Share Tech Mono', monospace;
    animation: infoBlink 5s ease-in-out infinite;
}
@keyframes infoBlink {
    0%,100% { border-color: rgba(0,212,255,0.2); }
    50%      { border-color: rgba(191,95,255,0.4); }
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020409 0%, #0D1117 100%) !important;
    border-right: 1px solid rgba(0,212,255,0.15) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 11px !important;
    letter-spacing: 2px !important;
    color: #8B9099 !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.3s !important;
}
.stTabs [aria-selected="true"] {
    color: #00D4FF !important;
    border-bottom: 2px solid #00D4FF !important;
    text-shadow: 0 0 8px rgba(0,212,255,0.6) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0D1117; }
::-webkit-scrollbar-thumb { background: linear-gradient(#00D4FF, #BF5FFF); border-radius: 3px; }

h1,h2,h3 { color: white !important; }
label { color: #CDD9E5 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DEEP NEURAL NETWORK
# ─────────────────────────────────────────────────────────────────────────────
class DeepNeuralNetwork:
    def __init__(self, layer_sizes, dropout_rate=0.2, learning_rate=0.001, use_batch_norm=True):
        self.layer_sizes    = layer_sizes
        self.dropout_rate   = dropout_rate
        self.lr             = learning_rate
        self.use_batch_norm = use_batch_norm
        self.training       = True
        self.weights, self.biases       = [], []
        self.bn_gamma, self.bn_beta     = [], []
        self.bn_running_mean, self.bn_running_var = [], []
        self.m_w, self.v_w = [], []
        self.m_b, self.v_b = [], []
        self.m_g, self.v_g = [], []
        self.m_bt, self.v_bt = [], []
        self.t = 0
        self._init_weights()

    def _init_weights(self):
        for i in range(len(self.layer_sizes) - 1):
            fi, fo = self.layer_sizes[i], self.layer_sizes[i + 1]
            W = np.random.randn(fi, fo) * np.sqrt(2.0 / fi)
            b = np.zeros((1, fo))
            self.weights.append(W); self.biases.append(b)
            self.m_w.append(np.zeros_like(W)); self.v_w.append(np.zeros_like(W))
            self.m_b.append(np.zeros_like(b)); self.v_b.append(np.zeros_like(b))
            if self.use_batch_norm and i < len(self.layer_sizes) - 2:
                g = np.ones((1, fo)); bt = np.zeros((1, fo))
                self.bn_gamma.append(g); self.bn_beta.append(bt)
                self.bn_running_mean.append(np.zeros((1, fo)))
                self.bn_running_var.append(np.ones((1, fo)))
                self.m_g.append(np.zeros_like(g)); self.v_g.append(np.zeros_like(g))
                self.m_bt.append(np.zeros_like(bt)); self.v_bt.append(np.zeros_like(bt))

    def _leaky_relu(self, z, a=0.01): return np.where(z > 0, z, a * z)
    def _leaky_relu_d(self, z, a=0.01): return np.where(z > 0, 1, a)
    def _softplus(self, z): return np.log1p(np.exp(np.clip(z, -500, 500)))

    def _bn_fwd(self, z, idx, eps=1e-8):
        if self.training:
            mu = z.mean(0, keepdims=True); var = z.var(0, keepdims=True)
            self.bn_running_mean[idx] = 0.9 * self.bn_running_mean[idx] + 0.1 * mu
            self.bn_running_var[idx]  = 0.9 * self.bn_running_var[idx]  + 0.1 * var
        else:
            mu = self.bn_running_mean[idx]; var = self.bn_running_var[idx]
        zn = (z - mu) / np.sqrt(var + eps)
        return self.bn_gamma[idx] * zn + self.bn_beta[idx], zn, mu, var

    def forward(self, X):
        self.cache = {"A": [X], "Z": [], "Z_norm": [], "mu": [], "var": [], "dm": []}
        A = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            Z = A @ W + b; self.cache["Z"].append(Z)
            if i < len(self.weights) - 1:
                if self.use_batch_norm and i < len(self.bn_gamma):
                    Z, zn, mu, var = self._bn_fwd(Z, i)
                    self.cache["Z_norm"].append(zn)
                    self.cache["mu"].append(mu); self.cache["var"].append(var)
                A = self._leaky_relu(Z)
                if self.training and self.dropout_rate > 0:
                    mask = (np.random.rand(*A.shape) > self.dropout_rate) / (1 - self.dropout_rate)
                    A *= mask; self.cache["dm"].append(mask)
                else: self.cache["dm"].append(None)
            else:
                A = self._softplus(Z)
            self.cache["A"].append(A)
        return A

    def backward(self, yt):
        yp = self.cache["A"][-1]; m = yt.shape[0]; nl = len(self.weights)
        d = yp - yt.reshape(-1, 1)
        dA = np.where(np.abs(d) <= 1.0, d, np.sign(d)) / m
        sig = 1 / (1 + np.exp(-np.clip(self.cache["Z"][-1], -500, 500)))
        dZ = dA * sig
        gW = [None]*nl; gb = [None]*nl
        gg = [None]*len(self.bn_gamma); gbt = [None]*len(self.bn_gamma)
        for i in range(nl - 1, -1, -1):
            Ap = self.cache["A"][i]
            gW[i] = Ap.T @ dZ; gb[i] = dZ.sum(0, keepdims=True)
            if i > 0:
                dAp = dZ @ self.weights[i].T
                if self.cache["dm"][i-1] is not None: dAp *= self.cache["dm"][i-1]
                dZ = dAp * self._leaky_relu_d(self.cache["Z"][i-1])
                if self.use_batch_norm and i-1 < len(self.bn_gamma):
                    bi = i-1; zn = self.cache["Z_norm"][bi]; var = self.cache["var"][bi]; eps = 1e-8
                    gg[bi]  = (dZ * zn).sum(0, keepdims=True)
                    gbt[bi] = dZ.sum(0, keepdims=True)
                    dzn = dZ * self.bn_gamma[bi]; n = dZ.shape[0]
                    dZ = (1/n)/np.sqrt(var+eps)*(n*dzn - dzn.sum(0,keepdims=True)
                          - zn*(dzn*zn).sum(0,keepdims=True))
        return gW, gb, gg, gbt

    def _adam(self, p, g, m, v, b1=0.9, b2=0.999, eps=1e-8):
        m[:] = b1*m + (1-b1)*g; v[:] = b2*v + (1-b2)*g**2
        p -= self.lr * (m/(1-b1**self.t)) / (np.sqrt(v/(1-b2**self.t)) + eps)

    def update(self, gW, gb, gg, gbt):
        self.t += 1
        for i in range(len(self.weights)):
            self._adam(self.weights[i], gW[i], self.m_w[i], self.v_w[i])
            self._adam(self.biases[i],  gb[i], self.m_b[i], self.v_b[i])
        for i in range(len(self.bn_gamma)):
            if gg[i] is not None:
                self._adam(self.bn_gamma[i], gg[i],  self.m_g[i],  self.v_g[i])
                self._adam(self.bn_beta[i],  gbt[i], self.m_bt[i], self.v_bt[i])

    def huber(self, yt, yp, d=1.0):
        diff = np.abs(yp.flatten() - yt)
        return np.mean(np.where(diff <= d, 0.5*diff**2, d*(diff-0.5*d)))

    def fit(self, Xtr, ytr, Xv, yv, epochs=300, bs=64, patience=30, lr_decay=0.97):
        hist = {"train_loss":[], "val_loss":[], "train_mae":[], "val_mae":[]}
        best_vl = np.inf; pc = 0; bw = None; bb = None; n = Xtr.shape[0]
        prog = st.progress(0, text="Training neural network…")
        status = st.empty()
        for ep in range(epochs):
            if ep > 0 and ep % 20 == 0: self.lr *= lr_decay
            self.training = True
            idx = np.random.permutation(n); Xs, ys = Xtr[idx], ytr[idx]
            tl = []
            for s in range(0, n, bs):
                Xb, yb = Xs[s:s+bs], ys[s:s+bs]
                yp = self.forward(Xb); tl.append(self.huber(yb, yp))
                gW, gb, gg, gbt = self.backward(yb); self.update(gW, gb, gg, gbt)
            self.training = False
            vpred = self.forward(Xv).flatten(); vl = self.huber(yv, vpred)
            tpred = self.forward(Xtr).flatten()
            hist["train_loss"].append(np.mean(tl)); hist["val_loss"].append(vl)
            hist["train_mae"].append(mean_absolute_error(ytr, tpred))
            hist["val_mae"].append(mean_absolute_error(yv, vpred))
            pct = int((ep+1)/epochs*100)
            prog.progress(pct, text=f"Epoch {ep+1}/{epochs}  |  Val Loss: {vl:.5f}  |  Val MAE: {hist['val_mae'][-1]:.4f}")
            if vl < best_vl - 1e-5:
                best_vl = vl; pc = 0
                bw = [w.copy() for w in self.weights]; bb = [b.copy() for b in self.biases]
            else:
                pc += 1
            if pc >= patience:
                status.success(f"✅ Early stopping at epoch {ep+1} — best val loss: {best_vl:.5f}")
                break
        prog.progress(100, text="Training complete!")
        if bw: self.weights = bw; self.biases = bb
        return hist

    def predict(self, X):
        self.training = False
        return self.forward(X).flatten()


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
RATING_MAP = {"a (resistant)": 0.05, "b (good)": 0.25, "c (questionable)": 0.75, "d (poor)": 2.50}

def parse_rate(val):
    val = str(val).strip()
    for k, v in RATING_MAP.items():
        if k in val.lower(): return v
    m = re.match(r'^([\d.]+)\s*(max|min)?$', val, re.IGNORECASE)
    if m: return float(m.group(1))
    nums = re.findall(r'[\d.]+', val)
    return float(nums[0]) if nums else np.nan

def uns_code(val):
    val = str(val).strip()
    if val in ('nan', ''): return -1
    pm = {'A':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'J':8,'K':9,'L':10,
          'M':11,'N':12,'P':13,'R':14,'S':15,'T':16,'W':17,'Z':18}
    return pm.get(val[0].upper(), 0)

@st.cache_data
def load_and_preprocess(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    df["corrosion_rate"] = df["Rate (mm/yr) or Rating"].apply(parse_rate)
    df = df.dropna(subset=["corrosion_rate"])
    df["corrosion_rate"] = df["corrosion_rate"].clip(upper=df["corrosion_rate"].quantile(0.99))
    le = LabelEncoder()
    df["material_enc"] = le.fit_transform(df["Material Family"].fillna("Unknown"))
    df["uns_enc"] = df["UNS"].apply(uns_code)
    for col in ["Temperature (deg C)", "pH", "Salinity (g/L)", "Humidity (%)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(
            {"Temperature (deg C)": 25, "pH": 7, "Salinity (g/L)": 5, "Humidity (%)": 60}[col])
    X = df[["Temperature (deg C)", "pH", "Salinity (g/L)", "Humidity (%)",
            "material_enc", "uns_enc"]].values.astype(float)
    y = df["corrosion_rate"].values
    ml = y * 1.0
    fr = np.clip(0.1*y + 0.3*(ml/10) + 0.2*(df["Temperature (deg C)"].values/95), 0, 1)
    targets = {"corrosion_rate": y, "metal_loss": ml, "failure_risk": fr}
    return X, targets, le, df

FEATURE_NAMES = ["Temperature (°C)", "pH", "Salinity (g/L)", "Humidity (%)", "Material Family", "UNS Code"]


# ─────────────────────────────────────────────────────────────────────────────
# PERMUTATION IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
def permutation_importance(model, X, y, n_repeats=3):
    base = mean_absolute_error(y, model.predict(X))
    imps = []
    for col in range(X.shape[1]):
        sc = []
        for _ in range(n_repeats):
            Xp = X.copy(); Xp[:, col] = np.random.permutation(Xp[:, col])
            sc.append(mean_absolute_error(y, model.predict(Xp)) - base)
        imps.append(np.mean(sc))
    idx = np.argsort(imps)[::-1]
    return [FEATURE_NAMES[i] for i in idx], np.array(imps)[idx]


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {"accent": "#00D4FF", "warn": "#FF6B35", "ok": "#39FF14",
          "purple": "#BF5FFF", "gold": "#FFD700", "bg": "#0D1117", "card": "#161B22"}

def plot_training_curves(history):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Huber Loss", "Mean Absolute Error"))
    ep = list(range(1, len(history["train_loss"]) + 1))
    fig.add_trace(go.Scatter(x=ep, y=history["train_loss"], name="Train Loss",
                             line=dict(color=COLORS["accent"], width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ep, y=history["val_loss"], name="Val Loss",
                             line=dict(color=COLORS["warn"], width=2, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ep, y=history["train_mae"], name="Train MAE",
                             line=dict(color=COLORS["ok"], width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=ep, y=history["val_mae"], name="Val MAE",
                             line=dict(color=COLORS["purple"], width=2, dash="dash")), row=1, col=2)
    fig.update_layout(paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["card"],
                      font_color="white", height=380, margin=dict(t=50, b=40, l=50, r=20),
                      legend=dict(bgcolor="#1C2333", bordercolor="#30363D"))
    fig.update_xaxes(gridcolor="#1C2333", zerolinecolor="#30363D")
    fig.update_yaxes(gridcolor="#1C2333", zerolinecolor="#30363D")
    return fig

def plot_pred_vs_actual(y_test, y_pred):
    err = np.abs(y_pred - y_test)
    lim = [min(y_test.min(), y_pred.min()) * 0.95, max(y_test.max(), y_pred.max()) * 1.05]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers",
                             marker=dict(color=err, colorscale="Plasma", size=6, opacity=0.6,
                                         colorbar=dict(title=dict(text="|Error|", font=dict(color="white")),
                                                       tickfont=dict(color="white"))),
                             name="Predictions"))
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                             line=dict(color="white", dash="dash", width=1.5), name="Perfect fit"))
    r2 = r2_score(y_test, y_pred)
    fig.add_annotation(x=0.05, y=0.92, xref="paper", yref="paper",
                       text=f"R² = {r2:.4f}", font=dict(color=COLORS["ok"], size=14),
                       showarrow=False, bgcolor="#161B22")
    fig.update_layout(title="Predicted vs Actual Corrosion Rate",
                      xaxis_title="Actual (mm/year)", yaxis_title="Predicted (mm/year)",
                      paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["card"],
                      font_color="white", height=420, margin=dict(t=50, b=50, l=60, r=20))
    fig.update_xaxes(gridcolor="#1C2333"); fig.update_yaxes(gridcolor="#1C2333")
    return fig

def plot_residuals(y_test, y_pred):
    res = y_pred - y_test
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Residual Distribution", "Residuals vs Predicted"))
    fig.add_trace(go.Histogram(x=res, nbinsx=60, marker_color=COLORS["accent"],
                               opacity=0.75, name="Residuals"), row=1, col=1)
    fig.add_vline(x=0, line=dict(color=COLORS["warn"], dash="dash", width=2), row=1, col=1)
    fig.add_vline(x=res.mean(), line=dict(color=COLORS["ok"], width=1.5), row=1, col=1)
    fig.add_trace(go.Scatter(x=y_pred, y=res, mode="markers",
                             marker=dict(color=COLORS["accent"], size=4, opacity=0.4),
                             name="Scatter"), row=1, col=2)
    fig.add_hline(y=0, line=dict(color=COLORS["warn"], dash="dash", width=1.5), row=1, col=2)
    fig.update_layout(paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["card"],
                      font_color="white", height=380, margin=dict(t=50, b=40, l=50, r=20),
                      showlegend=False)
    fig.update_xaxes(gridcolor="#1C2333"); fig.update_yaxes(gridcolor="#1C2333")
    return fig

def plot_feature_importance(names, vals):
    colors = px.colors.sequential.Plasma_r[:len(names)]
    fig = go.Figure(go.Bar(x=vals[::-1], y=names[::-1], orientation="h",
                           marker=dict(color=colors),
                           text=[f"{v:.5f}" for v in vals[::-1]],
                           textposition="outside", textfont=dict(color="white", size=10)))
    fig.update_layout(title="Feature Importance (Permutation)",
                      xaxis_title="Increase in MAE when permuted",
                      paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["card"],
                      font_color="white", height=360, margin=dict(t=50, b=50, l=180, r=80))
    fig.update_xaxes(gridcolor="#1C2333"); fig.update_yaxes(gridcolor="#1C2333")
    return fig

def plot_distributions(y_test, y_pred, targets_test):
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("Corrosion Rate Distribution",
                                        "Failure Risk Score", "Metal Loss Distribution"))
    fig.add_trace(go.Histogram(x=targets_test["corrosion_rate"], nbinsx=60,
                               marker_color=COLORS["warn"], opacity=0.7, name="Actual"), row=1, col=1)
    fig.add_trace(go.Histogram(x=y_pred, nbinsx=60,
                               marker_color=COLORS["accent"], opacity=0.5, name="Predicted"), row=1, col=1)
    fig.add_trace(go.Histogram(x=targets_test["failure_risk"], nbinsx=50,
                               marker_color=COLORS["purple"], opacity=0.75, name="Risk"), row=1, col=2)
    fig.add_vline(x=0.5, line=dict(color=COLORS["warn"], dash="dash", width=2), row=1, col=2)
    fig.add_vline(x=0.7, line=dict(color="red", dash="dot", width=2), row=1, col=2)
    fig.add_trace(go.Histogram(x=targets_test["metal_loss"], nbinsx=60,
                               marker_color=COLORS["ok"], opacity=0.75, name="Metal Loss"), row=1, col=3)
    fig.update_layout(paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["card"],
                      font_color="white", height=380, margin=dict(t=50, b=40, l=50, r=20),
                      showlegend=False)
    fig.update_xaxes(gridcolor="#1C2333"); fig.update_yaxes(gridcolor="#1C2333")
    return fig

def plot_corrosion_heatmap(df_raw):
    try:
        pivot = df_raw.groupby(["Material Family", "corrosion_rate_bin"])["corrosion_rate"].mean().unstack()
        fig = px.imshow(pivot, color_continuous_scale="Plasma",
                        title="Avg Corrosion Rate by Material",
                        labels=dict(color="Avg Rate (mm/yr)"))
        fig.update_layout(paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["card"],
                          font_color="white", height=450)
        return fig
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg,#020409 0%,#060D1A 60%,#020409 100%) !important;
        border-right: 1px solid rgba(0,212,255,0.1) !important;
        box-shadow: 4px 0 30px rgba(0,212,255,0.06) !important;
    }
    @keyframes logoPulse {
        0%,100% { filter: drop-shadow(0 0 8px #00D4FF); }
        50%      { filter: drop-shadow(0 0 18px #BF5FFF); }
    }
    @keyframes statusDot {
        0%,100% { opacity:1; } 50% { opacity:0.2; }
    }
    [data-testid="stSlider"] > div > div > div > div {
        background: linear-gradient(90deg,#00D4FF,#BF5FFF) !important;
    }
    </style>

    <div style="text-align:center; padding:20px 0 12px;">
        <div style="font-family:Orbitron,monospace; font-size:28px; font-weight:900;
                    background:linear-gradient(135deg,#00D4FF,#BF5FFF,#FF0080);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    background-clip:text; letter-spacing:2px;
                    animation:logoPulse 3s ease-in-out infinite;">
            CorroSense
        </div>
        <div style="font-family:Orbitron,monospace; font-size:14px; font-weight:700;
                    color:#FFD700; letter-spacing:10px; margin-top:-2px;">
            AI
        </div>
        <div style="margin-top:10px; display:flex; justify-content:center; gap:6px; align-items:center;">
            <span style="width:7px;height:7px;border-radius:50%;background:#39FF14;
                         display:inline-block;animation:statusDot 1.2s ease-in-out infinite;"></span>
            <span style="font-family:'Share Tech Mono',monospace;font-size:10px;
                         color:#39FF14;letter-spacing:3px;">ONLINE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1px;background:linear-gradient(90deg,transparent,#00D4FF55,transparent);margin:6px 0 14px;'></div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("UPLOAD DATASET (.csv)", type=["csv"])

    st.markdown("<div style='height:1px;background:linear-gradient(90deg,transparent,#BF5FFF44,transparent);margin:14px 0 10px;'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:Orbitron,monospace;font-size:9px;letter-spacing:4px;
                color:#BF5FFF;text-transform:uppercase;margin-bottom:14px;text-align:center;">
        NEURAL PARAMETERS
    </div>
    """, unsafe_allow_html=True)

    epochs   = st.slider("MAX EPOCHS",          100, 500, 300, 50)
    batch_sz = st.slider("BATCH SIZE",            32,  128,  64, 16)
    lr       = st.select_slider("LEARNING RATE",
                   options=[0.0001, 0.0005, 0.001, 0.002, 0.005], value=0.001)
    dropout  = st.slider("DROPOUT RATE",         0.0,  0.5, 0.2, 0.05)
    patience = st.slider("EARLY STOP PATIENCE",   10,   50,  30,    5)

    st.markdown("<div style='height:1px;background:linear-gradient(90deg,transparent,#FF008055,transparent);margin:14px 0 12px;'></div>", unsafe_allow_html=True)

    train_btn = st.button("IGNITE NEURAL CORE", use_container_width=True)

    st.markdown("<div style='height:1px;background:linear-gradient(90deg,transparent,#00D4FF22,transparent);margin:12px 0;'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='title-wrapper'>
    <div class='epic-title'>CorroSense AI</div>
    <div class='title-sub'>material degradation prediction engine</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='width:100%;height:3px;margin-bottom:18px;overflow:hidden;
            background:linear-gradient(90deg,transparent,#FF0080,#FFD700,#39FF14,#00D4FF,#BF5FFF,transparent);
            background-size:200% 100%;animation:borderRun 2s linear infinite;'>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "<div class='info-banner'>Upload your dataset, configure parameters in the sidebar, "
    "click <b style='color:#FF0080;'>IGNITE NEURAL CORE</b>, then explore the results across all tabs.</div>",
    unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_train, tab_eval, tab_pred, tab_data, tab_rec = st.tabs(
    ["  TRAINING  ", "  EVALUATION  ", "  PREDICT  ", "  DATASET  ", "  RECOMMEND  "])

st.markdown("""
<style>
/* Tab list background */
.stTabs [data-baseweb="tab-list"] {
    background: #0A0E18 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
}

/* All tabs base */
.stTabs [data-baseweb="tab"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 10px !important;
    letter-spacing: 3px !important;
    font-weight: 700 !important;
    border-radius: 7px !important;
    padding: 10px 18px !important;
    border: none !important;
    transition: all 0.3s ease !important;
    color: #444 !important;
    background: transparent !important;
}

/* Tab 1 — TRAINING — Cyan */
.stTabs [data-baseweb="tab"]:nth-child(1) { border-bottom: none !important; }
.stTabs [data-baseweb="tab"]:nth-child(1)[aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(0,212,255,0.05)) !important;
    color: #00D4FF !important;
    box-shadow: 0 0 16px rgba(0,212,255,0.2), inset 0 1px 0 rgba(0,212,255,0.3) !important;
    text-shadow: 0 0 10px rgba(0,212,255,0.8) !important;
}
.stTabs [data-baseweb="tab"]:nth-child(1):hover { color: #00D4FF99 !important; }

/* Tab 2 — EVALUATION — Purple */
.stTabs [data-baseweb="tab"]:nth-child(2)[aria-selected="true"] {
    background: linear-gradient(135deg, rgba(191,95,255,0.15), rgba(191,95,255,0.05)) !important;
    color: #BF5FFF !important;
    box-shadow: 0 0 16px rgba(191,95,255,0.2), inset 0 1px 0 rgba(191,95,255,0.3) !important;
    text-shadow: 0 0 10px rgba(191,95,255,0.8) !important;
}
.stTabs [data-baseweb="tab"]:nth-child(2):hover { color: #BF5FFF99 !important; }

/* Tab 3 — PREDICT — Green */
.stTabs [data-baseweb="tab"]:nth-child(3)[aria-selected="true"] {
    background: linear-gradient(135deg, rgba(57,255,20,0.12), rgba(57,255,20,0.04)) !important;
    color: #39FF14 !important;
    box-shadow: 0 0 16px rgba(57,255,20,0.15), inset 0 1px 0 rgba(57,255,20,0.25) !important;
    text-shadow: 0 0 10px rgba(57,255,20,0.8) !important;
}
.stTabs [data-baseweb="tab"]:nth-child(3):hover { color: #39FF1499 !important; }

/* Tab 4 — DATASET — Gold */
.stTabs [data-baseweb="tab"]:nth-child(4)[aria-selected="true"] {
    background: linear-gradient(135deg, rgba(255,215,0,0.12), rgba(255,215,0,0.04)) !important;
    color: #FFD700 !important;
    box-shadow: 0 0 16px rgba(255,215,0,0.15), inset 0 1px 0 rgba(255,215,0,0.25) !important;
    text-shadow: 0 0 10px rgba(255,215,0,0.8) !important;
}
.stTabs [data-baseweb="tab"]:nth-child(4):hover { color: #FFD70099 !important; }

/* Tab 5 — RECOMMEND — Orange-Red */
.stTabs [data-baseweb="tab"]:nth-child(5)[aria-selected="true"] {
    background: linear-gradient(135deg, rgba(255,107,53,0.15), rgba(255,0,128,0.05)) !important;
    color: #FF6B35 !important;
    box-shadow: 0 0 16px rgba(255,107,53,0.2), inset 0 1px 0 rgba(255,107,53,0.3) !important;
    text-shadow: 0 0 10px rgba(255,107,53,0.9) !important;
}
.stTabs [data-baseweb="tab"]:nth-child(5):hover { color: #FF6B3599 !important; }

/* Remove default underline indicator */
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"]    { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
for k in ["model", "scaler", "y_scaler", "history", "y_test", "y_pred",
          "targets_test", "feat_imp", "le", "df_raw"]:
    if k not in st.session_state:
        st.session_state[k] = None

# ── TRAIN ─────────────────────────────────────────────────────────────────────
if train_btn:
    if uploaded is None:
        st.error("Please upload a CSV dataset first.")
    else:
        with tab_train:
            file_bytes = uploaded.read()
            with st.spinner("Preprocessing data…"):
                X, targets, le, df_raw = load_and_preprocess(file_bytes)
                st.session_state["le"]     = le
                st.session_state["df_raw"] = df_raw
                y = targets["corrosion_rate"]

            c1, c2, c3 = st.columns(3)
            c1.markdown(f"""<div class='metric-card blue'><div class='label'>TOTAL SAMPLES</div>
                <div class='value'>{len(y):,}</div><div class='note'>rows loaded</div></div>""", unsafe_allow_html=True)
            c2.markdown(f"""<div class='metric-card purple'><div class='label'>FEATURES</div>
                <div class='value'>{X.shape[1]}</div><div class='note'>input dimensions</div></div>""", unsafe_allow_html=True)
            c3.markdown(f"""<div class='metric-card orange'><div class='label'>RATE RANGE</div>
                <div class='value'>{y.min():.2f}–{y.max():.2f}</div><div class='note'>mm/yr</div></div>""", unsafe_allow_html=True)

            # Split
            X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
            X_tr,  X_val,  y_tr,  y_val  = train_test_split(X_tmp, y_tmp, test_size=0.176, random_state=42)
            all_idx = np.arange(len(X))
            _, ti = train_test_split(all_idx, test_size=0.15, random_state=42)
            targets_test = {k: v[ti] for k, v in targets.items()}

            c1b, c2b, c3b = st.columns(3)
            c1b.markdown(f"""<div class='metric-card green'><div class='label'>TRAIN SET</div>
                <div class='value'>{X_tr.shape[0]:,}</div><div class='note'>samples</div></div>""", unsafe_allow_html=True)
            c2b.markdown(f"""<div class='metric-card gold'><div class='label'>VAL SET</div>
                <div class='value'>{X_val.shape[0]:,}</div><div class='note'>samples</div></div>""", unsafe_allow_html=True)
            c3b.markdown(f"""<div class='metric-card orange'><div class='label'>TEST SET</div>
                <div class='value'>{X_test.shape[0]:,}</div><div class='note'>samples</div></div>""", unsafe_allow_html=True)

            # Scale
            sc = StandardScaler();  X_tr_sc = sc.fit_transform(X_tr)
            X_val_sc = sc.transform(X_val); X_test_sc = sc.transform(X_test)
            ysc = StandardScaler()
            y_tr_sc  = ysc.fit_transform(y_tr.reshape(-1,1)).flatten()
            y_val_sc = ysc.transform(y_val.reshape(-1,1)).flatten()

            # Build
            arch = [X.shape[1], 128, 128, 64, 32, 1]
            st.info(f"Architecture: {' → '.join(map(str, arch))}")
            model = DeepNeuralNetwork(layer_sizes=arch, dropout_rate=dropout,
                                      learning_rate=lr, use_batch_norm=True)

            # Train
            st.markdown("<div class='section-header'>LIVE NEURAL TRAINING</div>", unsafe_allow_html=True)
            history = model.fit(X_tr_sc, y_tr_sc, X_val_sc, y_val_sc,
                                epochs=epochs, bs=batch_sz, patience=patience)

            # Predict
            y_pred_sc = model.predict(X_test_sc)
            y_pred    = ysc.inverse_transform(y_pred_sc.reshape(-1,1)).flatten()
            y_pred    = np.maximum(y_pred, 0)

            # Feature importance
            with st.spinner("Computing feature importance…"):
                fn, fi = permutation_importance(model, X_test_sc, y_test)

            # Store state
            st.session_state.update({"model": model, "scaler": sc, "y_scaler": ysc,
                                     "history": history, "y_test": y_test, "y_pred": y_pred,
                                     "targets_test": targets_test, "feat_imp": (fn, fi)})

            st.markdown("<div class='section-header'>TRAINING CURVES</div>", unsafe_allow_html=True)
            st.plotly_chart(plot_training_curves(history), use_container_width=True, key="train_tab_curves")

# ── EVAL TAB ──────────────────────────────────────────────────────────────────
with tab_eval:
    if st.session_state["model"] is None:
        st.markdown("""
        <div style='text-align:center;padding:60px 20px;'>
            <div style='font-family:Orbitron,monospace;font-size:40px;opacity:0.15;'>[ ]</div>
            <div style='font-family:Orbitron,monospace;font-size:14px;color:#333;letter-spacing:4px;margin-top:12px;'>
                AWAITING NEURAL TRAINING
            </div>
            <div style='font-family:Share Tech Mono,monospace;font-size:11px;color:#222;margin-top:8px;'>
                Train a model first → metrics will appear here
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        y_test  = st.session_state["y_test"]
        y_pred  = st.session_state["y_pred"]
        history = st.session_state["history"]
        tt      = st.session_state["targets_test"]
        fn, fi  = st.session_state["feat_imp"]

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
        hr   = 100 * np.mean(tt["failure_risk"] > 0.5)

        st.markdown("""
        <div style='font-family:Orbitron,monospace;font-size:9px;letter-spacing:5px;color:#333;
                    text-align:center;margin-bottom:14px;text-transform:uppercase;'>
            CorroSense AI  —  Model Performance Report
        </div>""", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>PERFORMANCE METRICS</div>", unsafe_allow_html=True)
        cols = st.columns(5)
        cards = [
            ("MAE",  f"{mae:.5f}", "mm/year", "blue"),
            ("RMSE", f"{rmse:.5f}", "mm/year", "orange"),
            ("R²",   f"{r2:.4f}",  "↑ closer to 1", "green"),
            ("MAPE", f"{mape:.1f}%", "↓ lower is better", "purple"),
            ("High-Risk %", f"{hr:.1f}%", "equipment needing attention", "gold"),
        ]
        for col, (label, val, note, cls) in zip(cols, cards):
            col.markdown(f"""
            <div class="metric-card {cls}">
                <div class="label">{label}</div>
                <div class="value">{val}</div>
                <div class="note">{note}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div class='section-header'>LIVE TRAINING CURVES</div>", unsafe_allow_html=True)
        st.plotly_chart(plot_training_curves(history), use_container_width=True, key="eval_curves")

        st.markdown("<div class='section-header'>PREDICTED vs ACTUAL</div>", unsafe_allow_html=True)
        st.plotly_chart(plot_pred_vs_actual(y_test, y_pred), use_container_width=True, key="eval_pred_actual")

        st.markdown("<div class='section-header'>RESIDUAL ANALYSIS</div>", unsafe_allow_html=True)
        st.plotly_chart(plot_residuals(y_test, y_pred), use_container_width=True, key="eval_residuals")

        st.markdown("<div class='section-header'>FEATURE IMPORTANCE</div>", unsafe_allow_html=True)
        st.plotly_chart(plot_feature_importance(fn, fi), use_container_width=True, key="eval_feat_imp")

        st.markdown("<div class='section-header'>DISTRIBUTION INTELLIGENCE</div>", unsafe_allow_html=True)
        st.plotly_chart(plot_distributions(y_test, y_pred, tt), use_container_width=True, key="eval_distributions")

# ── PREDICT TAB ───────────────────────────────────────────────────────────────
with tab_pred:
    st.markdown("<div class='section-header'>CORROSION ORACLE</div>", unsafe_allow_html=True)
    if st.session_state["model"] is None:
        st.markdown("""
        <div style='text-align:center;padding:60px 20px;'>
            <div style='font-family:Orbitron,monospace;font-size:40px;opacity:0.15;'>[ ]</div>
            <div style='font-family:Orbitron,monospace;font-size:14px;color:#333;letter-spacing:4px;margin-top:12px;'>
                ORACLE OFFLINE
            </div>
            <div style='font-family:Share Tech Mono,monospace;font-size:11px;color:#222;margin-top:8px;'>
                Train CorroSense AI first to unlock predictions
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        le     = st.session_state["le"]
        scaler = st.session_state["scaler"]
        ysc    = st.session_state["y_scaler"]
        model  = st.session_state["model"]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div style='font-family:Orbitron,monospace;font-size:9px;letter-spacing:3px;color:#00D4FF44;margin-bottom:8px;'>ENVIRONMENTAL CONDITIONS</div>", unsafe_allow_html=True)
            temp     = st.slider("TEMPERATURE (°C)",  0.0, 100.0, 60.0, 0.5)
            ph       = st.slider("pH LEVEL",           0.0,  14.0,  7.0, 0.1)
            salinity = st.slider("SALINITY (g/L)",     0.0,  35.0, 15.0, 0.5)
        with col2:
            st.markdown("<div style='font-family:Orbitron,monospace;font-size:9px;letter-spacing:3px;color:#BF5FFF44;margin-bottom:8px;'>MATERIAL PROPERTIES</div>", unsafe_allow_html=True)
            humidity = st.slider("HUMIDITY (%)",       0.0, 100.0, 70.0, 1.0)
            material = st.selectbox("MATERIAL FAMILY", sorted(le.classes_))
            uns_raw  = st.text_input("UNS CODE", value="S30400")

        mat_enc = int(le.transform([material])[0])
        uns_enc_val = uns_code(uns_raw)
        sample = np.array([[temp, ph, salinity, humidity, mat_enc, uns_enc_val]])
        sample_sc  = scaler.transform(sample)
        pred_sc    = model.predict(sample_sc)
        pred_rate  = float(ysc.inverse_transform([[pred_sc[0]]])[0][0])
        pred_rate  = max(pred_rate, 0.0)

        if pred_rate < 0.1:
            sev = "LOW"; sev_color = "#39FF14"
        elif pred_rate < 0.5:
            sev = "MEDIUM"; sev_color = "#FFD700"
        elif pred_rate < 1.5:
            sev = "HIGH"; sev_color = "#FF6B35"
        else:
            sev = "CRITICAL"; sev_color = "#FF0040"

        st.markdown(f"""
        <div class="pred-box">
            <div style="font-family:Orbitron,monospace;font-size:9px;letter-spacing:5px;
                        color:#333;margin-bottom:8px;">CorroSense AI ORACLE</div>
            <div class="label" style="font-size:12px;color:#8B9099;letter-spacing:3px;">
                PREDICTED CORROSION RATE
            </div>
            <div class="rate">{pred_rate:.4f}</div>
            <div class="unit">mm / year</div>
            <div class="badge" style="color:{sev_color};font-weight:900;margin-top:14px;
                 font-family:Orbitron,monospace;font-size:16px;letter-spacing:3px;">{sev}</div>
            <div style="margin-top:14px;display:flex;justify-content:center;gap:20px;
                        font-family:Share Tech Mono,monospace;font-size:10px;">
                <span style="color:#333;">TEMP: <span style="color:#00D4FF;">{temp}°C</span></span>
                <span style="color:#333;">pH: <span style="color:#BF5FFF;">{ph}</span></span>
                <span style="color:#333;">SAL: <span style="color:#39FF14;">{salinity}g/L</span></span>
                <span style="color:#333;">HUM: <span style="color:#FFD700;">{humidity}%</span></span>
            </div>
        </div>""", unsafe_allow_html=True)

        # Gauge chart
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_rate,
            number={"suffix": " mm/yr", "font": {"color": "white", "size": 28}},
            gauge={
                "axis": {"range": [0, 5], "tickcolor": "white", "tickfont": {"color": "white"}},
                "bar":  {"color": sev_color},
                "bgcolor": "#161B22",
                "steps": [
                    {"range": [0, 0.1],  "color": "#0D2818"},
                    {"range": [0.1, 0.5], "color": "#2D2A00"},
                    {"range": [0.5, 1.5], "color": "#2D1400"},
                    {"range": [1.5, 5.0], "color": "#2D0010"},
                ],
                "threshold": {"line": {"color": "white", "width": 2}, "thickness": 0.75, "value": pred_rate},
            },
            title={"text": "Corrosion Severity Gauge", "font": {"color": "white"}},
        ))
        fig_g.update_layout(paper_bgcolor=COLORS["bg"], font_color="white", height=300,
                            margin=dict(t=30, b=20, l=30, r=30))
        st.plotly_chart(fig_g, use_container_width=True, key="pred_gauge")

        # Sensitivity: vary temperature
        st.markdown("<div class='section-header'>SENSITIVITY ANALYSIS</div>", unsafe_allow_html=True)
        temps = np.linspace(0, 100, 50)
        sens_preds = []
        for t in temps:
            s = np.array([[t, ph, salinity, humidity, mat_enc, uns_enc_val]])
            ps = model.predict(scaler.transform(s))
            sens_preds.append(max(float(ysc.inverse_transform([[ps[0]]])[0][0]), 0))
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=temps, y=sens_preds, mode="lines+markers",
                                   line=dict(color=COLORS["accent"], width=2),
                                   marker=dict(size=4), name="Corrosion Rate"))
        fig_s.add_vline(x=temp, line=dict(color=COLORS["warn"], dash="dash"), annotation_text="Current")
        fig_s.update_layout(xaxis_title="Temperature (°C)", yaxis_title="Predicted Rate (mm/yr)",
                            paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["card"],
                            font_color="white", height=320, margin=dict(t=20, b=50, l=60, r=20))
        fig_s.update_xaxes(gridcolor="#1C2333"); fig_s.update_yaxes(gridcolor="#1C2333")
        st.plotly_chart(fig_s, use_container_width=True, key="pred_sensitivity")

# ── DATA TAB ──────────────────────────────────────────────────────────────────
with tab_data:
    st.markdown("<div class='section-header'>DATA INTELLIGENCE HUB</div>", unsafe_allow_html=True)
    if st.session_state["df_raw"] is None:
        st.markdown("""
        <div style='text-align:center;padding:60px 20px;'>
            <div style='font-family:Orbitron,monospace;font-size:40px;opacity:0.15;'>[ ]</div>
            <div style='font-family:Orbitron,monospace;font-size:14px;color:#333;letter-spacing:4px;margin-top:12px;'>
                NO DATA STREAM DETECTED
            </div>
            <div style='font-family:Share Tech Mono,monospace;font-size:11px;color:#222;margin-top:8px;'>
                Upload a dataset and train to explore data intelligence
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        df_raw = st.session_state["df_raw"]

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"""<div class='metric-card blue'><div class='label'>TOTAL ROWS</div>
            <div class='value'>{len(df_raw):,}</div><div class='note'>dataset size</div></div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class='metric-card purple'><div class='label'>MATERIAL FAMILIES</div>
            <div class='value'>{df_raw["Material Family"].nunique()}</div><div class='note'>unique types</div></div>""", unsafe_allow_html=True)
        c3.markdown(f"""<div class='metric-card green'><div class='label'>AVG pH</div>
            <div class='value'>{df_raw["pH"].mean():.2f}</div><div class='note'>acidity level</div></div>""", unsafe_allow_html=True)
        c4.markdown(f"""<div class='metric-card orange'><div class='label'>AVG TEMP °C</div>
            <div class='value'>{df_raw["Temperature (deg C)"].mean():.1f}</div><div class='note'>thermal exposure</div></div>""", unsafe_allow_html=True)

        st.markdown("<div class='section-header'>RAW DATA STREAM</div>", unsafe_allow_html=True)
        st.dataframe(df_raw.head(200), use_container_width=True, height=320)

        st.markdown("<div class='section-header'>CORROSION RATE BY MATERIAL</div>", unsafe_allow_html=True)
        mat_stats = df_raw.groupby("Material Family")["corrosion_rate"].mean().sort_values(ascending=False).head(20)
        fig_mat = go.Figure(go.Bar(x=mat_stats.values, y=mat_stats.index, orientation="h",
                                   marker=dict(color=mat_stats.values, colorscale="Plasma"),
                                   text=[f"{v:.3f}" for v in mat_stats.values], textposition="outside",
                                   textfont=dict(color="white")))
        fig_mat.update_layout(xaxis_title="Avg Corrosion Rate (mm/yr)",
                              paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["card"],
                              font_color="white", height=500, margin=dict(t=20, b=50, l=200, r=80))
        fig_mat.update_xaxes(gridcolor="#1C2333"); fig_mat.update_yaxes(gridcolor="#1C2333")
        st.plotly_chart(fig_mat, use_container_width=True, key="data_mat_bar")

        st.markdown("<div class='section-header'>TEMPERATURE vs CORROSION RATE</div>", unsafe_allow_html=True)
        fig_sc = px.scatter(df_raw, x="Temperature (deg C)", y="corrosion_rate", color="Material Family",
                            opacity=0.5, labels={"corrosion_rate": "Corrosion Rate (mm/yr)"},
                            color_discrete_sequence=px.colors.qualitative.Vivid)
        fig_sc.update_layout(paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["card"],
                             font_color="white", height=420, margin=dict(t=20, b=50, l=60, r=20))
        fig_sc.update_xaxes(gridcolor="#1C2333"); fig_sc.update_yaxes(gridcolor="#1C2333")
        st.plotly_chart(fig_sc, use_container_width=True, key="data_scatter")


# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATION HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def get_corrosion_risk_global(temp, ph, sal, hum, mat, env):
    score = 0
    score += min(temp / 100 * 35, 35)
    if ph < 4:    score += 30
    elif ph < 6:  score += 18
    elif ph > 10: score += 20
    elif ph > 8:  score += 8
    score += min(sal / 35 * 20, 20)
    score += min(hum / 100 * 15, 15)
    mat_bonus = {"Carbon Steel": 0, "Cast Iron": 2, "Brass": -3, "Copper Alloy": -4,
                 "Zinc Alloy": -3, "Aluminum Alloy": -8, "Stainless Steel (304)": -15,
                 "Stainless Steel (316)": -20, "Nickel Alloy": -22, "Titanium Alloy": -28}
    score += mat_bonus.get(mat, 0)
    env_bonus = {"Marine / Offshore": 15, "Acidic Chemical": 20, "Alkaline Chemical": 12,
                 "High Temperature Gas": 10, "Soil / Underground": 8,
                 "Industrial Atmosphere": 5, "Humid Indoor": 2, "Freshwater Immersion": 0}
    score += env_bonus.get(env, 0)
    return max(0, min(100, score))

def estimate_lifespan_global(risk_score, mat):
    base = {"Carbon Steel": 15, "Cast Iron": 12, "Brass": 25, "Copper Alloy": 30,
            "Zinc Alloy": 20, "Aluminum Alloy": 35, "Stainless Steel (304)": 50,
            "Stainless Steel (316)": 60, "Nickel Alloy": 70, "Titanium Alloy": 100}
    b = base.get(mat, 20)
    factor = 1.0 - (risk_score / 100) * 0.75
    return max(1, round(b * factor, 1))

def get_best_material_text(temp, ph, sal, hum, env, current_mat):
    materials = ["Carbon Steel", "Cast Iron", "Brass", "Zinc Alloy", "Copper Alloy",
                 "Aluminum Alloy", "Stainless Steel (304)", "Stainless Steel (316)",
                 "Nickel Alloy", "Titanium Alloy"]
    best_mat = min(materials, key=lambda m: get_corrosion_risk_global(temp, ph, sal, hum, m, env))
    second   = sorted(materials, key=lambda m: get_corrosion_risk_global(temp, ph, sal, hum, m, env))[1]
    risk_b   = get_corrosion_risk_global(temp, ph, sal, hum, best_mat, env)
    life_b   = estimate_lifespan_global(risk_b, best_mat)
    current_risk = get_corrosion_risk_global(temp, ph, sal, hum, current_mat, env)
    current_life = estimate_lifespan_global(current_risk, current_mat)
    upgrade_note = ""
    if best_mat != current_mat:
        upgrade_note = (f" Switching from {current_mat} (lifespan {current_life} yrs) to "
                        f"{best_mat} adds {round(life_b - current_life, 1)} years of service life.")
    return (f"For your conditions (T={temp}°C, pH={ph}, Salinity={sal} g/L, "
            f"Humidity={hum}%, Environment: {env}), the optimal material is "
            f"<b style='color:#39FF14;'>{best_mat}</b> — suitability score "
            f"{100-risk_b:.0f}%, estimated lifespan {life_b} years. "
            f"Second best: <b style='color:#FFD700;'>{second}</b>.{upgrade_note} "
            f"If budget is a constraint, apply high-performance coatings to {second} "
            f"to approach {best_mat} performance at lower cost.")

# ── RECOMMEND TAB ─────────────────────────────────────────────────────────────
with tab_rec:
    st.markdown("<div class='section-header'>CORROSION CONTROL INTELLIGENCE</div>", unsafe_allow_html=True)

    # ── CSS for recommendation cards ──────────────────────────────────────────
    st.markdown("""
    <style>
    .rec-card {
        background: linear-gradient(135deg, #0A0F1A, #111827);
        border-radius: 12px;
        padding: 20px 22px;
        margin-bottom: 14px;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s;
    }
    .rec-card::before {
        content:'';
        position:absolute;top:0;left:0;width:4px;height:100%;
        border-radius:12px 0 0 12px;
    }
    .rec-card.cyan::before  { background: linear-gradient(180deg,#00D4FF,#0080CC); box-shadow: 0 0 12px #00D4FF88; }
    .rec-card.green::before { background: linear-gradient(180deg,#39FF14,#00AA00); box-shadow: 0 0 12px #39FF1488; }
    .rec-card.orange::before{ background: linear-gradient(180deg,#FF6B35,#CC3300); box-shadow: 0 0 12px #FF6B3588; }
    .rec-card.purple::before{ background: linear-gradient(180deg,#BF5FFF,#7700CC); box-shadow: 0 0 12px #BF5FFF88; }
    .rec-card.gold::before  { background: linear-gradient(180deg,#FFD700,#CC9900); box-shadow: 0 0 12px #FFD70088; }
    .rec-card.red::before   { background: linear-gradient(180deg,#FF0080,#AA0044); box-shadow: 0 0 12px #FF008088; }

    .rec-card .rec-title {
        font-family: 'Orbitron', monospace;
        font-size: 11px;
        letter-spacing: 3px;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .rec-card.cyan   .rec-title { color: #00D4FF; }
    .rec-card.green  .rec-title { color: #39FF14; }
    .rec-card.orange .rec-title { color: #FF6B35; }
    .rec-card.purple .rec-title { color: #BF5FFF; }
    .rec-card.gold   .rec-title { color: #FFD700; }
    .rec-card.red    .rec-title { color: #FF0080; }

    .rec-card .rec-body {
        font-family: 'Share Tech Mono', monospace;
        font-size: 12px;
        color: #8B9099;
        line-height: 1.8;
    }
    .rec-card .rec-tag {
        display: inline-block;
        font-family: 'Orbitron', monospace;
        font-size: 8px;
        letter-spacing: 2px;
        padding: 3px 10px;
        border-radius: 20px;
        margin: 8px 4px 0 0;
        font-weight: 700;
    }
    .rec-card.cyan   .rec-tag { background: rgba(0,212,255,0.12);  color: #00D4FF;  border: 1px solid rgba(0,212,255,0.3);  }
    .rec-card.green  .rec-tag { background: rgba(57,255,20,0.1);   color: #39FF14;  border: 1px solid rgba(57,255,20,0.25); }
    .rec-card.orange .rec-tag { background: rgba(255,107,53,0.12); color: #FF6B35;  border: 1px solid rgba(255,107,53,0.3); }
    .rec-card.purple .rec-tag { background: rgba(191,95,255,0.12); color: #BF5FFF;  border: 1px solid rgba(191,95,255,0.3); }
    .rec-card.gold   .rec-tag { background: rgba(255,215,0,0.1);   color: #FFD700;  border: 1px solid rgba(255,215,0,0.25); }
    .rec-card.red    .rec-tag { background: rgba(255,0,128,0.1);   color: #FF0080;  border: 1px solid rgba(255,0,128,0.25); }

    .ideal-table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'Share Tech Mono', monospace;
        font-size: 12px;
        margin-top: 8px;
    }
    .ideal-table th {
        font-family: 'Orbitron', monospace;
        font-size: 9px;
        letter-spacing: 2px;
        color: #555;
        padding: 8px 12px;
        border-bottom: 1px solid rgba(0,212,255,0.1);
        text-align: left;
        text-transform: uppercase;
    }
    .ideal-table td {
        padding: 9px 12px;
        border-bottom: 1px solid rgba(255,255,255,0.04);
        color: #8B9099;
    }
    .ideal-table tr:hover td { background: rgba(0,212,255,0.03); }
    .ideal-table .good { color: #39FF14; font-weight: 700; }
    .ideal-table .warn { color: #FFD700; font-weight: 700; }
    .ideal-table .bad  { color: #FF6B35; font-weight: 700; }

    .score-bar-wrap {
        background: #0A0E18;
        border-radius: 4px;
        height: 8px;
        width: 100%;
        margin-top: 6px;
        overflow: hidden;
    }
    .score-bar {
        height: 100%;
        border-radius: 4px;
        transition: width 1s ease;
    }
    .lifespan-card {
        background: linear-gradient(135deg, #0A0F1A, #111827);
        border: 1px solid rgba(57,255,20,0.2);
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 0 20px rgba(57,255,20,0.05);
    }
    .lifespan-card .lval {
        font-family: 'Orbitron', monospace;
        font-size: 32px;
        font-weight: 900;
        color: #39FF14;
        text-shadow: 0 0 16px rgba(57,255,20,0.6);
    }
    .lifespan-card .llabel {
        font-family: 'Share Tech Mono', monospace;
        font-size: 10px;
        color: #555;
        letter-spacing: 3px;
        margin-top: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Input panel for recommendations ──────────────────────────────────────
    st.markdown("""
    <div style='background:linear-gradient(135deg,rgba(255,107,53,0.06),rgba(255,0,128,0.03));
                border:1px solid rgba(255,107,53,0.2);border-radius:12px;
                padding:16px 20px;margin-bottom:20px;
                font-family:Share Tech Mono,monospace;font-size:12px;color:#8B9099;'>
        Configure your operating conditions below. CorroSense AI will generate personalised
        recommendations to maximise metal lifespan and minimise degradation.
    </div>
    """, unsafe_allow_html=True)

    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        r_temp     = st.slider("TEMPERATURE (°C)",   0.0, 100.0, 45.0, 1.0, key="r_temp")
        r_ph       = st.slider("pH LEVEL",            0.0,  14.0,  7.0, 0.1, key="r_ph")
    with rc2:
        r_salinity = st.slider("SALINITY (g/L)",      0.0,  35.0,  5.0, 0.5, key="r_sal")
        r_humidity = st.slider("HUMIDITY (%)",        0.0, 100.0, 55.0, 1.0, key="r_hum")
    with rc3:
        r_material = st.selectbox("MATERIAL", [
            "Carbon Steel", "Stainless Steel (304)", "Stainless Steel (316)",
            "Aluminum Alloy", "Copper Alloy", "Nickel Alloy",
            "Titanium Alloy", "Zinc Alloy", "Cast Iron", "Brass"
        ], key="r_mat")
        r_env = st.selectbox("ENVIRONMENT TYPE", [
            "Marine / Offshore", "Industrial Atmosphere", "Freshwater Immersion",
            "Acidic Chemical", "Alkaline Chemical", "High Temperature Gas",
            "Soil / Underground", "Humid Indoor"
        ], key="r_env")

    # ── Generate recommendations engine ──────────────────────────────────────
    # Using global helper functions defined above

    risk = get_corrosion_risk_global(r_temp, r_ph, r_salinity, r_humidity, r_material, r_env)
    lifespan = estimate_lifespan_global(risk, r_material)

    if risk < 25:
        risk_label = "LOW RISK"; risk_color = "#39FF14"; risk_glow = "rgba(57,255,20,0.4)"
    elif risk < 50:
        risk_label = "MODERATE RISK"; risk_color = "#FFD700"; risk_glow = "rgba(255,215,0,0.4)"
    elif risk < 75:
        risk_label = "HIGH RISK"; risk_color = "#FF6B35"; risk_glow = "rgba(255,107,53,0.4)"
    else:
        risk_label = "CRITICAL RISK"; risk_color = "#FF0080"; risk_glow = "rgba(255,0,128,0.4)"

    # ── Risk score display ────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>RISK ASSESSMENT</div>", unsafe_allow_html=True)
    ra1, ra2, ra3, ra4 = st.columns(4)
    ra1.markdown(f"""
    <div class='lifespan-card' style='border-color:rgba(255,107,53,0.3);'>
        <div class='lval' style='color:{risk_color};text-shadow:0 0 16px {risk_glow};font-size:36px;'>{risk}</div>
        <div class='score-bar-wrap'><div class='score-bar' style='width:{risk}%;background:linear-gradient(90deg,#39FF14,#FFD700,#FF6B35,#FF0080);'></div></div>
        <div class='llabel'>CORROSION RISK SCORE / 100</div>
        <div style='font-family:Orbitron,monospace;font-size:9px;letter-spacing:2px;
                    color:{risk_color};margin-top:8px;'>{risk_label}</div>
    </div>""", unsafe_allow_html=True)

    ra2.markdown(f"""
    <div class='lifespan-card'>
        <div class='lval'>{lifespan}</div>
        <div class='llabel'>ESTIMATED LIFESPAN (YEARS)</div>
        <div style='font-family:Share Tech Mono,monospace;font-size:10px;color:#444;margin-top:6px;'>
            without intervention
        </div>
    </div>""", unsafe_allow_html=True)

    ideal_temp  = 20.0
    ideal_ph    = 7.5
    ideal_sal   = 0.5
    ideal_hum   = 45.0
    ideal_life  = estimate_lifespan_global(
        get_corrosion_risk_global(ideal_temp, ideal_ph, ideal_sal, ideal_hum, r_material, r_env),
        r_material)
    gain = round(ideal_life - lifespan, 1)

    ra3.markdown(f"""
    <div class='lifespan-card' style='border-color:rgba(0,212,255,0.2);'>
        <div class='lval' style='color:#00D4FF;text-shadow:0 0 16px rgba(0,212,255,0.6);'>{ideal_life}</div>
        <div class='llabel'>IDEAL LIFESPAN (YEARS)</div>
        <div style='font-family:Share Tech Mono,monospace;font-size:10px;color:#444;margin-top:6px;'>
            under optimal conditions
        </div>
    </div>""", unsafe_allow_html=True)

    gain_color = "#39FF14" if gain > 0 else "#FF6B35"
    ra4.markdown(f"""
    <div class='lifespan-card' style='border-color:rgba(57,255,20,0.2);'>
        <div class='lval' style='color:{gain_color};text-shadow:0 0 16px rgba(57,255,20,0.4);'>
            +{gain if gain > 0 else abs(gain)}
        </div>
        <div class='llabel'>YEARS GAIN POSSIBLE</div>
        <div style='font-family:Share Tech Mono,monospace;font-size:10px;color:#444;margin-top:6px;'>
            by following recommendations
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Ideal conditions table ────────────────────────────────────────────────
    st.markdown("<div class='section-header'>IDEAL OPERATING CONDITIONS</div>", unsafe_allow_html=True)

    mat_ideal = {
        "Carbon Steel":          (15, 25, "6.5–8.0", "< 5",  "< 50", "Neutral / Dry"),
        "Cast Iron":             (10, 25, "6.5–8.0", "< 3",  "< 55", "Neutral Indoor"),
        "Stainless Steel (304)": (5,  35, "4.0–10.0","< 15", "< 70", "Most Environments"),
        "Stainless Steel (316)": (5,  40, "3.0–11.0","< 25", "< 75", "Marine Safe"),
        "Aluminum Alloy":        (5,  40, "5.5–8.5", "< 20", "< 80", "Atmospheric"),
        "Copper Alloy":          (5,  35, "6.0–9.0", "< 10", "< 65", "Freshwater / Air"),
        "Nickel Alloy":          (5,  60, "2.0–12.0","< 30", "< 85", "Aggressive Chem."),
        "Titanium Alloy":        (5,  80, "1.0–13.0","< 35", "< 90", "All Environments"),
        "Zinc Alloy":            (5,  30, "6.0–8.5", "< 5",  "< 55", "Dry Atmosphere"),
        "Brass":                 (5,  30, "6.5–8.5", "< 8",  "< 60", "Freshwater"),
    }
    mi = mat_ideal.get(r_material, (15, 25, "6.5–8.0", "< 5", "< 50", "Neutral"))

    def classify(val, ideal_max_str, current):
        try:
            ideal_max = float(ideal_max_str.replace("<","").replace(">","").strip())
            if current <= ideal_max * 0.6:  return "good"
            elif current <= ideal_max:       return "warn"
            else:                            return "bad"
        except: return "warn"

    temp_cls = "good" if r_temp <= mi[1] else ("warn" if r_temp <= mi[1]*1.3 else "bad")
    ph_val = r_ph
    ph_low, ph_high = [float(x) for x in mi[2].replace(" ","").split("–")]
    ph_cls = "good" if ph_low <= ph_val <= ph_high else ("warn" if abs(ph_val - 7) < 3 else "bad")
    sal_max = float(mi[3].replace("<","").replace(">","").strip())
    sal_cls = "good" if r_salinity <= sal_max * 0.6 else ("warn" if r_salinity <= sal_max else "bad")
    hum_max = float(mi[4].replace("<","").replace(">","").strip())
    hum_cls = "good" if r_humidity <= hum_max * 0.6 else ("warn" if r_humidity <= hum_max else "bad")

    label_map = {"good": "OPTIMAL", "warn": "ACCEPTABLE", "bad": "EXCEEDS LIMIT"}

    st.markdown(f"""
    <table class='ideal-table'>
        <tr>
            <th>Parameter</th><th>Your Value</th><th>Ideal Range</th><th>Status</th>
        </tr>
        <tr>
            <td>Temperature</td>
            <td>{r_temp} °C</td>
            <td>{mi[0]}–{mi[1]} °C</td>
            <td class='{temp_cls}'>{label_map[temp_cls]}</td>
        </tr>
        <tr>
            <td>pH Level</td>
            <td>{r_ph}</td>
            <td>{mi[2]}</td>
            <td class='{ph_cls}'>{label_map[ph_cls]}</td>
        </tr>
        <tr>
            <td>Salinity</td>
            <td>{r_salinity} g/L</td>
            <td>{mi[3]} g/L</td>
            <td class='{sal_cls}'>{label_map[sal_cls]}</td>
        </tr>
        <tr>
            <td>Humidity</td>
            <td>{r_humidity} %</td>
            <td>{mi[4]} %</td>
            <td class='{hum_cls}'>{label_map[hum_cls]}</td>
        </tr>
        <tr>
            <td>Environment Suitability</td>
            <td>{r_env}</td>
            <td>{mi[5]}</td>
            <td class='warn'>VERIFY</td>
        </tr>
    </table>
    """, unsafe_allow_html=True)

    # ── Recommendations ───────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>ACTIONABLE RECOMMENDATIONS</div>", unsafe_allow_html=True)

    # Build dynamic recommendations
    recs = []

    # Temperature
    if r_temp > mi[1]:
        recs.append(("cyan", "TEMPERATURE CONTROL",
            f"Your operating temperature ({r_temp}°C) exceeds the safe threshold of {mi[1]}°C for {r_material}. "
            f"Install heat exchangers or cooling jackets to bring temperature below {mi[1]}°C. "
            f"Every 10°C reduction approximately halves the electrochemical corrosion rate (Arrhenius law). "
            f"Consider thermal insulation coatings to buffer fluctuations.",
            ["CRITICAL ACTION", "HEAT EXCHANGER", "THERMAL MANAGEMENT"]))
    else:
        recs.append(("cyan", "TEMPERATURE — OPTIMAL",
            f"Temperature ({r_temp}°C) is within the safe operating range for {r_material} (max {mi[1]}°C). "
            f"Maintain consistent thermal conditions. Avoid thermal cycling as repeated expansion/contraction "
            f"accelerates micro-cracking at grain boundaries.",
            ["MAINTAIN", "MONITOR MONTHLY"]))

    # pH
    if ph_cls == "bad":
        if r_ph < ph_low:
            recs.append(("orange", "pH NEUTRALISATION REQUIRED",
                f"pH {r_ph} is highly acidic — significantly below the safe range {mi[2]}. "
                f"Inject alkaline inhibitors (sodium hydroxide, lime dosing) to raise pH above {ph_low}. "
                f"Install inline pH sensors with automatic dosing pumps for continuous correction. "
                f"At pH below 4, hydrogen evolution accelerates anodic dissolution of {r_material} dramatically.",
                ["URGENT", "CHEMICAL DOSING", "pH SENSOR"]))
        else:
            recs.append(("orange", "pH ALKALINITY CONTROL",
                f"pH {r_ph} is alkaline — above the safe range {mi[2]}. "
                f"Dose with weak acids (CO₂ injection or dilute H₂SO₄) to bring pH below {ph_high}. "
                f"High alkalinity causes caustic cracking and stress corrosion, especially in steels.",
                ["URGENT", "ACID DOSING", "pH MONITORING"]))
    else:
        recs.append(("cyan", "pH LEVEL — ACCEPTABLE",
            f"pH {r_ph} is within or near the acceptable range {mi[2]} for {r_material}. "
            f"Install continuous pH monitoring. Log weekly and correct any drift beyond ±0.5 pH units. "
            f"Consider adding pH buffer chemicals to maintain stability during process variations.",
            ["MONITOR WEEKLY", "pH BUFFER"]))

    # Salinity
    if sal_cls == "bad":
        recs.append(("purple", "SALINITY REDUCTION — CRITICAL",
            f"Salinity of {r_salinity} g/L far exceeds the recommended {mi[3]} g/L for {r_material}. "
            f"Chloride ions are the primary driver of pitting corrosion and crevice attack. "
            f"Install reverse osmosis or ion-exchange demineralisation systems. "
            f"Apply cathodic protection (impressed current or sacrificial anodes) as an immediate measure. "
            f"Consider switching to Stainless Steel 316 or Titanium Alloy for high-salinity environments.",
            ["CRITICAL", "CATHODIC PROTECTION", "RO SYSTEM", "MATERIAL UPGRADE"]))
    elif sal_cls == "warn":
        recs.append(("gold", "SALINITY — MONITOR CLOSELY",
            f"Salinity ({r_salinity} g/L) is approaching the safe limit ({mi[3]} g/L). "
            f"Apply a corrosion inhibitor — chromate-based or organic phosphonate types are highly effective. "
            f"Ensure all weld joints and crevices are sealed to prevent chloride concentration. "
            f"Consider a sacrificial zinc or magnesium anode system as additional protection.",
            ["INHIBITOR DOSING", "ANODE SYSTEM", "WELD INSPECTION"]))
    else:
        recs.append(("green", "SALINITY — CONTROLLED",
            f"Salinity ({r_salinity} g/L) is well within safe limits. "
            f"Maintain by monitoring with conductivity sensors every 2 weeks. "
            f"During shutdowns, flush systems with deionised water to prevent salt concentration.",
            ["GOOD", "BI-WEEKLY CHECK"]))

    # Humidity
    if hum_cls == "bad":
        recs.append(("red", "HUMIDITY — DEHUMIDIFICATION NEEDED",
            f"Humidity ({r_humidity}%) exceeds the threshold ({mi[4]}) for {r_material}. "
            f"Above 60% RH, electrolyte films form on metal surfaces enabling electrochemical corrosion. "
            f"Install industrial dehumidifiers or silica gel desiccant systems. "
            f"Apply moisture-barrier coatings (epoxy polyamide or zinc-rich primer) immediately. "
            f"Ensure adequate ventilation with filtered, dry air intake.",
            ["DEHUMIDIFIER", "COATING REQUIRED", "VENTILATION"]))
    else:
        recs.append(("green", "HUMIDITY — WELL CONTROLLED",
            f"Humidity ({r_humidity}%) is within safe operating range. "
            f"Maintain by ensuring HVAC or ventilation systems operate continuously. "
            f"During seasonal humidity spikes, increase inspection frequency to monthly.",
            ["MAINTAIN", "SEASONAL CHECK"]))

    # Material-specific recommendation
    mat_recs = {
        "Carbon Steel": ("gold", "PROTECTIVE COATING ESSENTIAL",
            "Carbon steel has no inherent corrosion resistance — protection is mandatory. "
            "Apply a 3-layer coating system: zinc-rich primer (75µm) + epoxy intermediate (125µm) + "
            "polyurethane topcoat (75µm). Reapply every 5–7 years. "
            "For immersion service, use impressed current cathodic protection (ICCP) at -850 mV vs Cu/CuSO₄. "
            "Apply hot-dip galvanising for structural members — provides 15–25 years protection.",
            ["3-LAYER COATING", "ICCP", "GALVANISING", "5-YEAR INSPECTION"]),
        "Stainless Steel (316)": ("green", "PASSIVATION LAYER MAINTENANCE",
            "SS316 relies on a chromium oxide passive layer — maintain it by avoiding surface damage. "
            "Never use carbon steel tools or wire brushes — contamination causes pitting. "
            "Passivate after any welding or machining using 20–50% nitric acid solution. "
            "Avoid chloride concentration in crevices — ensure full drainage in all geometries.",
            ["PASSIVATION", "CLEAN TOOLS ONLY", "NO CHLORIDE POOLING"]),
        "Aluminum Alloy": ("cyan", "ANODISING AND SEALANT",
            "Anodise the surface to build a 25–50µm oxide layer — dramatically improves corrosion resistance. "
            "Seal anodised layer with hot deionised water or nickel acetate for maximum protection. "
            "Avoid contact with copper or carbon steel components (galvanic corrosion risk). "
            "Apply chromate conversion coating (Alodine) for electrical conductivity requirement zones.",
            ["ANODISING", "GALVANIC ISOLATION", "CHROMATE COATING"]),
        "Titanium Alloy": ("cyan", "MINIMAL INTERVENTION REQUIRED",
            "Titanium is virtually immune to corrosion in most environments — an exceptional choice. "
            "Primary risk is hydrogen embrittlement at high cathodic protection levels — keep above -700 mV. "
            "Avoid contact with pure titanium and fluoride-containing media above 70°C. "
            "No coating required — inspect annually for mechanical damage only.",
            ["INSPECT ANNUALLY", "AVOID FLUORIDES", "CATHODIC LIMIT"]),
        "Copper Alloy": ("gold", "DEZINCIFICATION PROTECTION",
            "Ensure alloy composition contains >1% tin or arsenic to prevent dezincification. "
            "Avoid ammonia-containing environments — stress corrosion cracking risk is high. "
            "Apply inhibitor 1H-benzotriazole (BTA) in water circuits — forms protective film on copper. "
            "Keep velocity below 1.5 m/s in pipes to prevent erosion-corrosion.",
            ["INHIBITOR BTA", "FLOW CONTROL", "ALLOY SELECTION"]),
        "Nickel Alloy": ("purple", "STRESS CORROSION MONITORING",
            "Nickel alloys excel in aggressive environments — ensure correct alloy grade for your medium. "
            "Primary risk is stress corrosion cracking (SCC) above 300°C in chloride + oxygen environments. "
            "Monitor for hydrogen-induced cracking in acidic high-pressure service. "
            "Use solution annealing heat treatment to relieve residual stresses post-fabrication.",
            ["SCC MONITORING", "HEAT TREATMENT", "GRADE VERIFICATION"]),
        "Cast Iron": ("orange", "GRAPHITISATION PREVENTION",
            "Cast iron is susceptible to graphitisation in soils and seawater — the iron matrix dissolves leaving fragile graphite. "
            "Apply coal tar epoxy lining internally + fusion-bonded epoxy externally for buried service. "
            "Install magnesium sacrificial anodes at 10m spacing for underground infrastructure. "
            "Inspect with ultrasonic thickness gauging every 3 years.",
            ["COAL TAR EPOXY", "SACRIFICIAL ANODES", "UT INSPECTION"]),
        "Brass": ("gold", "DEZINCIFICATION INHIBITION",
            "Brass dezincifies in stagnant water — use dezincification-resistant (DZR) brass containing arsenic. "
            "Avoid high-velocity flow exceeding 2 m/s — erosion-corrosion strips protective oxide. "
            "In potable water, ensure pH > 7 and avoid CO₂ saturation. "
            "Apply BTA inhibitor in closed-loop cooling systems.",
            ["DZR BRASS", "FLOW LIMIT", "pH CONTROL"]),
        "Zinc Alloy": ("cyan", "ATMOSPHERIC PROTECTION",
            "Zinc alloys corrode slowly by forming a protective patina of zinc carbonate. "
            "Avoid acid exposure — zinc dissolves rapidly below pH 6. "
            "Apply clear lacquer coating to slow patina formation in aesthetic applications. "
            "In marine atmospheres, lifespan reduces to 30% — consider alternative materials.",
            ["LACQUER COATING", "AVOID ACIDS", "MARINE RISK"]),
        "Stainless Steel (304)": ("green", "CHLORIDE MONITORING",
            "SS304 is susceptible to pitting in chloride concentrations above 200 ppm. "
            "Monitor chloride levels in contact media monthly — upgrade to SS316 if levels rise. "
            "Passivate annually using 10% citric acid or 20% nitric acid solution. "
            "Avoid prolonged contact with cutting oils or machining fluids containing chlorides.",
            ["CHLORIDE LIMIT 200PPM", "ANNUAL PASSIVATION", "UPGRADE TO 316"]),
    }

    if r_material in mat_recs:
        recs.append(mat_recs[r_material])

    # Environment-specific recommendation
    env_recs = {
        "Marine / Offshore": ("red", "MARINE CORROSION STRATEGY",
            "Marine environments demand a multi-barrier defence: "
            "(1) Coating system — thermally sprayed aluminium (TSA) or epoxy/polyurethane; "
            "(2) Cathodic protection — aluminium alloy sacrificial anodes (replace when 50% consumed); "
            "(3) Corrosion allowance — add 3–6 mm extra wall thickness in design; "
            "(4) Splash zone protection — neoprene sheathing or Monel cladding. "
            "Inspect every 6 months using underwater ROV inspection.",
            ["MULTI-BARRIER", "SACRIFICIAL ANODES", "ROV INSPECTION", "6-MONTH CYCLE"]),
        "Acidic Chemical": ("orange", "CHEMICAL RESISTANCE LINING",
            "Apply PTFE, PFA, or rubber lining to all contact surfaces — provides chemical barrier. "
            "If metal is unavoidable, use hastelloy C-276 or titanium for strongest acid resistance. "
            "Monitor wall thickness monthly with ultrasonic gauging — acid corrosion can be rapid. "
            "Ensure emergency shutdown systems flush with neutralising agent on leak detection.",
            ["PTFE LINING", "HASTELLOY", "MONTHLY UT", "EMERGENCY FLUSH"]),
        "Soil / Underground": ("purple", "BURIED ASSET PROTECTION",
            "Apply fusion-bonded epoxy (FBE) + polyethylene wrap coating system. "
            "Install cathodic protection — impressed current at -850 mV (CSE) or magnesium anodes for remote areas. "
            "Conduct soil resistivity survey — resistivity < 500 Ω·cm requires intensive CP. "
            "Install coupons for annual corrosion rate measurement.",
            ["FBE COATING", "CP SYSTEM", "SOIL SURVEY", "ANNUAL COUPONS"]),
        "High Temperature Gas": ("orange", "HIGH-TEMP OXIDATION CONTROL",
            "Apply aluminide or MCrAlY thermal barrier coatings for service above 500°C. "
            "Select high-temperature alloys: Inconel 625, Haynes 230, or silicon-carbide composites. "
            "Monitor oxide scale thickness — spalling above 1 mm accelerates metal loss exponentially. "
            "Implement controlled atmosphere (inert gas purge) to suppress oxidation.",
            ["THERMAL COATING", "INCONEL", "SCALE MONITORING", "INERT ATMOSPHERE"]),
        "Freshwater Immersion": ("cyan", "FRESHWATER PROTECTION",
            "Apply epoxy or vinyl ester coating — 400µm minimum dry film thickness. "
            "Install sacrificial zinc anodes if cathodic protection is needed. "
            "Monitor dissolved oxygen — O₂ > 8 mg/L accelerates corrosion; consider deaeration. "
            "Control biological fouling with biocide dosing — biofilms accelerate MIC (microbiologically induced corrosion).",
            ["EPOXY 400µm", "ZINC ANODES", "DEAERATION", "BIOCIDE"]),
        "Industrial Atmosphere": ("gold", "ATMOSPHERIC CORROSION PROGRAMME",
            "Apply a minimum 3-coat system: zinc silicate primer + epoxy intermediate + aliphatic polyurethane topcoat. "
            "Inspect and touch up every 2 years in polluted environments, every 5 years in clean atmospheres. "
            "Install drip edges and drainage paths — ponding water under coatings causes blistering. "
            "Consider thermal spray zinc for long-maintenance-free service (30+ years).",
            ["3-COAT SYSTEM", "THERMAL SPRAY Zn", "2-5 YEAR CYCLE"]),
        "Humid Indoor": ("green", "HUMIDITY CONTROL PRIORITY",
            "Maintain indoor RH below 50% using HVAC with dehumidification. "
            "Apply VCI (Volatile Corrosion Inhibitor) paper or emitters in enclosed spaces. "
            "Use wax or oil films for stored components. "
            "Inspect for condensation on cold surfaces — insulate cold pipes above dew point.",
            ["VCI EMITTERS", "RH < 50%", "INSULATE COLD PIPES"]),
        "Alkaline Chemical": ("purple", "CAUSTIC ENVIRONMENT PROTOCOL",
            "Avoid carbon steel above 60°C in NaOH — caustic cracking occurs rapidly. "
            "Use nickel alloy 200 or 201 (low carbon) for hot caustic service. "
            "Apply stress-relief heat treatment to all welded components before service. "
            "Monitor for hydrogen blistering — use hydrogen-resistant alloys in high-pressure zones.",
            ["NICKEL ALLOY 201", "STRESS RELIEF", "H₂ MONITORING"]),
    }

    if r_env in env_recs:
        recs.append(env_recs[r_env])

    # General best practices
    recs.append(("purple", "INSPECTION AND MONITORING PROGRAMME",
        f"Establish a structured inspection programme for {r_material} in {r_env}: "
        f"(1) Visual inspection — monthly for high-risk zones, quarterly for low-risk. "
        f"(2) Ultrasonic thickness measurement — biannually at identified corrosion hotspots. "
        f"(3) Electrochemical noise (ECN) monitoring — continuous for critical assets. "
        f"(4) Corrosion coupons — install and retrieve every 90 days to track actual rates. "
        f"(5) Thermographic imaging — annual to detect coating failures and under-insulation corrosion.",
        ["MONTHLY VISUAL", "UT BIANNUAL", "COUPONS 90-DAY", "THERMOGRAPHY"]))

    recs.append(("green", "BEST MATERIAL FOR YOUR CONDITIONS",
        get_best_material_text(r_temp, r_ph, r_salinity, r_humidity, r_env, r_material),
        ["MATERIAL INTEL", "UPGRADE PATH"]))

    # ── Render all recommendation cards ──────────────────────────────────────
    for color, title, body, tags in recs:
        tag_html = "".join([f"<span class='rec-tag'>{t}</span>" for t in tags])
        st.markdown(f"""
        <div class='rec-card {color}'>
            <div class='rec-title'>{title}</div>
            <div class='rec-body'>{body}</div>
            <div style='margin-top:10px;'>{tag_html}</div>
        </div>""", unsafe_allow_html=True)

    # ── Best material chart ───────────────────────────────────────────────────
    st.markdown("<div class='section-header'>MATERIAL SUITABILITY RANKING</div>", unsafe_allow_html=True)

    materials_ranked = [
        "Carbon Steel", "Cast Iron", "Brass", "Zinc Alloy",
        "Copper Alloy", "Aluminum Alloy", "Stainless Steel (304)",
        "Stainless Steel (316)", "Nickel Alloy", "Titanium Alloy"
    ]
    scores = []
    for m in materials_ranked:
        s = get_corrosion_risk_global(r_temp, r_ph, r_salinity, r_humidity, m, r_env)
        scores.append(100 - s)  # suitability = inverse of risk

    colors_bar = ["#39FF14" if s >= 70 else ("#FFD700" if s >= 45 else "#FF6B35") for s in scores]
    sorted_pairs = sorted(zip(scores, materials_ranked, colors_bar), reverse=True)
    scores_s, mats_s, cols_s = zip(*sorted_pairs)

    fig_rank = go.Figure(go.Bar(
        x=list(scores_s), y=list(mats_s), orientation="h",
        marker=dict(color=list(cols_s),
                    line=dict(color="rgba(255,255,255,0.05)", width=0.5)),
        text=[f"{s:.0f}%" for s in scores_s],
        textposition="outside", textfont=dict(color="white", size=11)
    ))
    fig_rank.add_vline(x=70, line=dict(color="#39FF14", dash="dot", width=1.5))
    fig_rank.add_vline(x=45, line=dict(color="#FFD700", dash="dot", width=1.5))
    fig_rank.add_annotation(x=70, y=len(mats_s)-0.5, text="GOOD",
                             font=dict(color="#39FF14", size=10), showarrow=False, xanchor="left")
    fig_rank.add_annotation(x=45, y=len(mats_s)-0.5, text="CAUTION",
                             font=dict(color="#FFD700", size=10), showarrow=False, xanchor="left")
    fig_rank.update_layout(
        xaxis_title="Suitability Score (%)",
        xaxis=dict(range=[0, 115]),
        paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["card"],
        font_color="white", height=420,
        margin=dict(t=20, b=50, l=160, r=80)
    )
    fig_rank.update_xaxes(gridcolor="#1C2333")
    fig_rank.update_yaxes(gridcolor="#1C2333")
    st.plotly_chart(fig_rank, use_container_width=True, key="rec_rank_chart")

    st.markdown("""
    <div style='background:linear-gradient(135deg,rgba(57,255,20,0.04),rgba(0,212,255,0.04));
                border:1px solid rgba(57,255,20,0.1);border-radius:10px;
                padding:14px 18px;margin-top:12px;
                font-family:Share Tech Mono,monospace;font-size:11px;color:#555;line-height:1.9;'>
        DISCLAIMER: Recommendations are generated based on electrochemical corrosion principles and
        industry standards (NACE SP0169, ISO 12944, ASTM G31). Always consult a certified corrosion
        engineer for safety-critical applications. CorroSense AI is a decision-support tool, not a
        substitute for professional engineering assessment.
    </div>
    """, unsafe_allow_html=True)


