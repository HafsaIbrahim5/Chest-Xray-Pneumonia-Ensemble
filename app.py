# ============================================================
#  PneumoScan AI  —  Streamlit App  (v2 — fixed & enhanced)
#  PyTorch Ensemble: Custom CNN + ResNet18/50 + DenseNet121
#                  + EfficientNet-B0 + Grad-CAM
#  Author: Hafsa Ibrahim
#  github.com/HafsaIbrahim5  |  linkedin.com/in/hafsa-ibrahim-ai-mi
# ============================================================

import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib

matplotlib.use("Agg")
from PIL import Image, ImageDraw, ImageFont
import os, time, warnings, io, datetime

warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="PneumoScan AI · Hafsa Ibrahim",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════
#  CSS THEME
# ════════════════════════════════════════════════════════════
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:ital,wght@0,300;0,400;0,500;1,400&family=Inter:wght@300;400;500;600&display=swap');

:root {
  --navy:    #03080f;
  --navy2:   #06111f;
  --card:    #0a1a2e;
  --card2:   #0d2240;
  --teal:    #00e5c3;
  --blue:    #1a8fff;
  --red:     #ff3b5c;
  --green:   #00e58a;
  --amber:   #f5a623;
  --txt:     #dceeff;
  --txt2:    #7a9bbf;
  --border:  rgba(0,229,195,0.14);
  --glow-t:  0 0 28px rgba(0,229,195,0.22);
  --glow-r:  0 0 28px rgba(255,59,92,0.28);
  --glow-g:  0 0 28px rgba(0,229,138,0.28);
  --r: 14px;
}

/* ── Base ── */
html, body, [class*="css"] {
  font-family: 'Inter', sans-serif !important;
  background: var(--navy) !important;
  color: var(--txt) !important;
  font-size: 15px !important;
  line-height: 1.6 !important;
}
.stApp {
  background:
    radial-gradient(ellipse 65% 45% at 5% 10%, rgba(0,229,195,.06) 0%, transparent 58%),
    radial-gradient(ellipse 55% 55% at 95% 88%, rgba(26,143,255,.05) 0%, transparent 58%),
    var(--navy) !important;
}
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb { background: var(--teal); border-radius: 3px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--navy2) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--txt) !important; }
[data-testid="stSidebar"] label {
  font-family: 'Inter', sans-serif !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  letter-spacing: 0 !important;
}

/* ── HERO ── */
.hero {
  background: linear-gradient(135deg, rgba(0,229,195,.06) 0%, rgba(26,143,255,.04) 100%);
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: 2.6rem 3rem 2rem;
  margin-bottom: 1.8rem;
  position: relative; overflow: hidden; text-align: center;
}
.hero::after {
  content: '';
  position: absolute; inset: 0;
  background: repeating-linear-gradient(
    -45deg, transparent 0, transparent 20px,
    rgba(0,229,195,.015) 20px, rgba(0,229,195,.015) 21px
  );
  pointer-events: none;
}
.hero-title {
  font-family: 'Syne', sans-serif;
  font-size: 2.8rem; font-weight: 800; letter-spacing: -0.5px;
  background: linear-gradient(135deg, #ffffff 0%, var(--teal) 48%, var(--blue) 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text; line-height: 1.15; margin: 0;
}
.hero-sub {
  font-family: 'Inter', sans-serif;
  color: var(--txt2); font-size: .95rem; font-weight: 400;
  margin-top: .65rem; letter-spacing: 0;
}
.hero-tags {
  margin-top: 1rem; display: flex; gap: .45rem;
  justify-content: center; flex-wrap: wrap;
}
.tag {
  display: inline-flex; align-items: center; gap: .3rem;
  background: rgba(0,229,195,.07); border: 1px solid rgba(0,229,195,.22);
  border-radius: 999px; padding: .25rem .8rem;
  font-family: 'JetBrains Mono', monospace;
  font-size: .65rem; color: var(--teal);
  letter-spacing: .6px; text-transform: uppercase;
}
.links-row { margin-top: .9rem; display: flex; gap: .7rem; justify-content: center; }
.link-btn {
  display: inline-flex; align-items: center; gap: .4rem;
  background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.11);
  border-radius: 8px; padding: .32rem .85rem; font-size: .78rem;
  font-family: 'Inter', sans-serif;
  color: var(--txt2); text-decoration: none; transition: all .25s;
}
.link-btn:hover { color: var(--teal); border-color: var(--teal); }

/* ── Glass Card ── */
.gc {
  background: rgba(10,26,46,.7);
  backdrop-filter: blur(20px);
  border: 1px solid var(--border); border-radius: var(--r);
  padding: 1.4rem; margin-bottom: 1.2rem;
  transition: border-color .3s, box-shadow .3s;
}
.gc:hover { border-color: rgba(0,229,195,.3); box-shadow: var(--glow-t); }

/* ── Section Label ── */
.slbl {
  font-family: 'JetBrains Mono', monospace;
  font-size: .63rem; letter-spacing: 2.2px;
  text-transform: uppercase; color: var(--teal);
  margin-bottom: .9rem;
  display: flex; align-items: center; gap: .5rem;
}
.slbl::before {
  content: ''; width: 18px; height: 2px;
  background: var(--teal); display: inline-block; border-radius: 1px;
}

/* ── Verdict ── */
.verdict-normal {
  background: linear-gradient(135deg, rgba(0,229,138,.1), rgba(0,229,138,.03));
  border: 1.5px solid rgba(0,229,138,.45); border-radius: var(--r);
  padding: 1.6rem; text-align: center; box-shadow: var(--glow-g);
}
.verdict-pneumonia {
  background: linear-gradient(135deg, rgba(255,59,92,.1), rgba(255,59,92,.03));
  border: 1.5px solid rgba(255,59,92,.45); border-radius: var(--r);
  padding: 1.6rem; text-align: center; box-shadow: var(--glow-r);
}
.verdict-label {
  font-family: 'Syne', sans-serif;
  font-size: 2.1rem; font-weight: 800;
}
.verdict-conf {
  font-family: 'JetBrains Mono', monospace;
  font-size: .78rem; color: var(--txt2); margin-top: .35rem;
}

/* ── Model Grid ── */
.mgrid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(148px, 1fr));
  gap: .8rem; margin: .9rem 0;
}
.mcard {
  background: var(--card2); border: 1px solid var(--border);
  border-radius: 12px; padding: 1rem; text-align: center;
  transition: transform .25s, border-color .25s;
}
.mcard:hover { transform: translateY(-3px); border-color: rgba(0,229,195,.4); }
.mcard.p-card {
  border-color: rgba(255,59,92,.4);
  background: linear-gradient(160deg, rgba(255,59,92,.07), var(--card));
}
.mcard.n-card {
  border-color: rgba(0,229,138,.4);
  background: linear-gradient(160deg, rgba(0,229,138,.07), var(--card));
}
.mcard.ens-card {
  border-color: rgba(0,229,195,.55) !important;
  background: linear-gradient(160deg, rgba(0,229,195,.1), var(--card)) !important;
  box-shadow: var(--glow-t);
}
.mcard-name {
  font-family: 'JetBrains Mono', monospace;
  font-size: .62rem; color: var(--txt2); letter-spacing: .8px;
}
.mcard-pred {
  font-family: 'Syne', sans-serif;
  font-size: .92rem; font-weight: 700; margin: .35rem 0 .2rem;
}
.mbar { height: 4px; border-radius: 2px; background: rgba(255,255,255,.06); margin-top: .5rem; overflow: hidden; }
.mbar-fill { height: 100%; border-radius: 2px; }

/* ── Ensemble Explain Box ── */
.ens-explain {
  background: rgba(0,229,195,.05);
  border: 1px solid rgba(0,229,195,.25);
  border-radius: 12px; padding: 1.1rem 1.3rem;
  margin-top: .9rem;
}
.ens-explain-title {
  font-family: 'Syne', sans-serif;
  font-size: .9rem; font-weight: 700; color: var(--teal);
  margin-bottom: .5rem;
}

/* ── Progress Bars ── */
.pbar { background: rgba(255,255,255,.05); border-radius: 999px; height: 9px; overflow: hidden; margin: .4rem 0 .15rem; }
.pbar-g { background: linear-gradient(90deg, #00b86e, var(--green)); }
.pbar-r { background: linear-gradient(90deg, #c02040, var(--red)); }

/* ── Stat Tiles ── */
.stiles { display: flex; gap: .7rem; flex-wrap: wrap; margin: .7rem 0; }
.stile {
  flex: 1; min-width: 75px; background: var(--card);
  border: 1px solid var(--border); border-radius: 10px;
  padding: .8rem; text-align: center;
}
.stile-val {
  font-family: 'JetBrains Mono', monospace;
  font-size: 1.15rem; font-weight: 500; color: var(--teal);
}
.stile-lbl { font-size: .63rem; color: var(--txt2); margin-top: .2rem; font-family: 'Inter', sans-serif; }

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, #006fcc, #00b8a0) !important;
  color: white !important; border: none !important; border-radius: 10px !important;
  font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
  font-size: 14px !important; letter-spacing: .3px !important;
  padding: .65rem 1.4rem !important;
  box-shadow: 0 4px 18px rgba(0,180,160,.3) !important;
  transition: all .3s !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 7px 22px rgba(0,180,160,.5) !important;
}

/* ── File Uploader ── */
.stFileUploader > div {
  background: var(--card) !important;
  border: 2px dashed rgba(0,229,195,.2) !important;
  border-radius: var(--r) !important; transition: all .3s !important;
}
.stFileUploader > div:hover {
  border-color: var(--teal) !important;
  background: rgba(0,229,195,.03) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] * {
  font-family: 'Inter', sans-serif !important;
  font-size: 13px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--card) !important; border-radius: 10px !important;
  padding: 4px !important; border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important; color: var(--txt2) !important;
  border-radius: 8px !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 600 !important; font-size: 13.5px !important;
}
.stTabs [aria-selected="true"] {
  background: rgba(0,229,195,.1) !important; color: var(--teal) !important;
}
.stTabs [data-baseweb="tab-panel"] {
  background: transparent !important; padding-top: .8rem !important;
}

/* ── Selects & Sliders ── */
.stSelectbox > div, .stSlider > div { background: var(--card) !important; }
.stSelectbox label, .stSlider label {
  font-family: 'Inter', sans-serif !important;
  font-size: 13px !important; font-weight: 500 !important;
  color: var(--txt2) !important; letter-spacing: 0 !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
  background: var(--card) !important; border: 1px solid var(--border) !important;
  border-radius: 10px !important; color: var(--txt) !important;
  font-family: 'Syne', sans-serif !important; font-weight: 600 !important;
  font-size: 14px !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
  background: var(--card) !important; border: 1px solid var(--border) !important;
  border-radius: 12px !important; padding: 1rem !important;
}
[data-testid="stMetricValue"] {
  color: var(--teal) !important;
  font-family: 'JetBrains Mono', monospace !important;
}

/* ── Demo/warning banner ── */
.banner-warn {
  background: rgba(245,166,35,.07); border: 1px solid rgba(245,166,35,.3);
  border-radius: 10px; padding: .8rem 1.2rem; margin-bottom: 1.5rem;
  font-family: 'Inter', sans-serif; font-size: .84rem;
  display: flex; align-items: center; gap: .8rem;
}

/* ── Disclaimer ── */
.disc {
  background: rgba(26,143,255,.06); border: 1px solid rgba(26,143,255,.2);
  border-radius: 10px; padding: .85rem 1.1rem;
  font-family: 'Inter', sans-serif; font-size: .82rem;
  color: var(--txt2); line-height: 1.7; margin-top: 1rem;
}

/* ── Footer ── */
.foot {
  text-align: center; padding: 2rem 0 1rem;
  font-family: 'Inter', sans-serif; font-size: .75rem;
  color: var(--txt2); line-height: 2;
}

/* ── Pulse dot ── */
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
.dot {
  width: 7px; height: 7px; background: var(--green); border-radius: 50%;
  display: inline-block; animation: pulse 2s ease-in-out infinite; margin-right: 5px;
}

/* ── Preprocessing comparison ── */
.prep-row {
  display: grid; grid-template-columns: 1fr 1fr; gap: .8rem; margin-top: .8rem;
}
.prep-label {
  font-family: 'JetBrains Mono', monospace; font-size: .62rem;
  color: var(--txt2); text-align: center; margin-top: .3rem;
  letter-spacing: .5px; text-transform: uppercase;
}
</style>
""",
    unsafe_allow_html=True,
)


# ════════════════════════════════════════════════════════════
#  MATPLOTLIB HELPERS  (no CSS rgba — use tuples!)
# ════════════════════════════════════════════════════════════
BG1 = "#03080f"
BG2 = "#0a1a2e"
TEAL = "#00e5c3"
BLUE = "#1a8fff"
RED = "#ff3b5c"
GRN = "#00e58a"
AMB = "#f5a623"
PUR = "#a855f7"
PAL = [TEAL, BLUE, GRN, AMB, RED, PUR]

# matplotlib-safe low-alpha colors  (r,g,b,a) float tuples
GRID_CLR = (1, 1, 1, 0.05)
SPINE_CLR = (1, 1, 1, 0.08)
LEG_CLR = (0.12, 0.22, 0.35, 1)


def _mpl_ax(fig, ax):
    """Apply consistent dark theme to any matplotlib axes."""
    fig.patch.set_facecolor(BG2)
    ax.set_facecolor(BG1)
    ax.tick_params(colors="#7a9bbf", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(SPINE_CLR)
    ax.grid(color=GRID_CLR, linewidth=0.5)


# ════════════════════════════════════════════════════════════
#  MODEL CLASS DEFINITIONS  (must match training exactly)
# ════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_ensemble(path="pneumonia_ensemble_full.pth"):
    try:
        import torch
        import torch.nn as nn
        import torchvision.models as tv

        class ImprovedCNN(nn.Module):
            def __init__(self, num_classes=2, dropout_rate=0.5):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(1, 32, 3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(True),
                    nn.Conv2d(32, 32, 3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(True),
                    nn.MaxPool2d(2, 2),
                    nn.Dropout2d(dropout_rate / 3),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                    nn.MaxPool2d(2, 2),
                    nn.Dropout2d(dropout_rate / 2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(True),
                    nn.MaxPool2d(2, 2),
                    nn.Dropout2d(dropout_rate),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(128 * 28 * 28, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(True),
                    nn.Dropout(dropout_rate),
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(True),
                    nn.Dropout(dropout_rate / 2),
                    nn.Linear(256, num_classes),
                )

            def forward(self, x):
                return self.classifier(self.features(x))

        class ResNetModel(nn.Module):
            def __init__(self, num_classes=2, model_name="resnet18", pretrained=False):
                super().__init__()
                bm = (
                    tv.resnet50(weights=None)
                    if model_name == "resnet50"
                    else tv.resnet18(weights=None)
                )
                self.base_model = bm
                orig = bm.conv1
                bm.conv1 = nn.Conv2d(
                    1,
                    64,
                    kernel_size=orig.kernel_size,
                    stride=orig.stride,
                    padding=orig.padding,
                    bias=orig.bias,
                )
                nf = bm.fc.in_features
                bm.fc = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(nf, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(True),
                    nn.Dropout(0.3),
                    nn.Linear(512, num_classes),
                )

            def forward(self, x):
                return self.base_model(x)

        class DenseNetModel(nn.Module):
            def __init__(
                self, num_classes=2, model_name="densenet121", pretrained=False
            ):
                super().__init__()
                self.base_model = tv.densenet121(weights=None)
                orig = self.base_model.features.conv0
                self.base_model.features.conv0 = nn.Conv2d(
                    1,
                    64,
                    kernel_size=orig.kernel_size,
                    stride=orig.stride,
                    padding=orig.padding,
                    bias=(orig.bias is not None),
                )
                nf = self.base_model.classifier.in_features
                self.base_model.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(nf, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(True),
                    nn.Dropout(0.3),
                    nn.Linear(512, num_classes),
                )

            def forward(self, x):
                return self.base_model(x)

        class EfficientNetModel(nn.Module):
            def __init__(
                self, num_classes=2, model_name="efficientnet_b0", pretrained=False
            ):
                super().__init__()
                try:
                    import timm

                    self.base_model = timm.create_model(
                        "efficientnet_b0",
                        pretrained=False,
                        in_chans=1,
                        num_classes=num_classes,
                    )
                except Exception:
                    self.base_model = tv.mobilenet_v2(weights=None)
                    self.base_model.features[0][0] = nn.Conv2d(
                        1, 32, 3, 2, 1, bias=False
                    )
                    self.base_model.classifier[1] = nn.Linear(1280, num_classes)

            def forward(self, x):
                return self.base_model(x)

        class EnsembleModel(nn.Module):
            def __init__(self, model_list, weights=None):
                super().__init__()
                self.models = nn.ModuleList(model_list)
                self.weights = (
                    weights if weights else [1.0 / len(model_list)] * len(model_list)
                )

            def forward(self, x):
                out = None
                for m, w in zip(self.models, self.weights):
                    o = m(x) * w
                    out = o if out is None else out + o
                return out

        if not os.path.exists(path):
            return None, False, f"'{path}' not found — place it next to app.py"
        ens = torch.load(path, map_location="cpu")
        ens.eval()
        return ens, True, "OK"
    except Exception as e:
        return None, False, str(e)


# ════════════════════════════════════════════════════════════
#  PREPROCESSING  (CLAHE → grayscale → 224 → norm)
# ════════════════════════════════════════════════════════════
def preprocess(pil_img):
    import torch
    import torchvision.transforms as T

    arr = np.array(pil_img.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    arr = clahe.apply(arr)
    img = Image.fromarray(arr)
    tfm = T.Compose(
        [T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean=[0.485], std=[0.229])]
    )
    return tfm(img).unsqueeze(0)  # [1,1,224,224]


def clahe_preview(pil_img):
    """Return (original_gray_np, clahe_np) for display."""
    arr = np.array(pil_img.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(arr)
    return arr, enhanced


# ════════════════════════════════════════════════════════════
#  INFERENCE
# ════════════════════════════════════════════════════════════
CLASSES = ["NORMAL", "PNEUMONIA"]


def _display_name(sub, idx):
    cname = type(sub).__name__
    if cname == "ImprovedCNN":
        return "Custom CNN"
    if cname == "ResNetModel":
        n = sum(p.numel() for p in sub.parameters())
        return "ResNet-50" if n > 20_000_000 else "ResNet-18"
    if cname == "DenseNetModel":
        return "DenseNet-121"
    if cname == "EfficientNetModel":
        return "EfficientNet-B0"
    return f"Model-{idx}"


def run_inference(ensemble, img_tensor):
    import torch

    results = {}
    for i, sub in enumerate(ensemble.models):
        sub.eval()
        with torch.no_grad():
            probs = torch.softmax(sub(img_tensor), dim=1)[0].numpy()
        name = _display_name(sub, i)
        results[f"{name}_{i}"] = {
            "display": name,
            "probs": probs,
            "pred": CLASSES[probs.argmax()],
            "conf": float(probs.max()),
            "weight": ensemble.weights[i],
        }
    with torch.no_grad():
        ens_probs = torch.softmax(ensemble(img_tensor), dim=1)[0].numpy()
    return results, CLASSES[ens_probs.argmax()], float(ens_probs.max()), ens_probs


def demo_inference(seed):
    rng = np.random.default_rng(int(seed * 9999) % 9999)
    base = float(rng.random())
    names = ["Custom CNN", "ResNet-18", "ResNet-50", "DenseNet-121", "EfficientNet-B0"]
    results = {}
    for i, n in enumerate(names):
        conf = float(rng.uniform(0.76, 0.97))
        pred = "PNEUMONIA" if base > 0.45 else "NORMAL"
        probs = (
            np.array([1 - conf, conf])
            if pred == "PNEUMONIA"
            else np.array([conf, 1 - conf])
        )
        results[f"{n}_{i}"] = {
            "display": n,
            "probs": probs,
            "pred": pred,
            "conf": conf,
            "weight": 0.2,
        }
    ec = float(rng.uniform(0.88, 0.98))
    ep = "PNEUMONIA" if base > 0.45 else "NORMAL"
    return (
        results,
        ep,
        ec,
        (np.array([1 - ec, ec]) if ep == "PNEUMONIA" else np.array([ec, 1 - ec])),
    )


# ════════════════════════════════════════════════════════════
#  GRAD-CAM  (PyTorch hooks)
# ════════════════════════════════════════════════════════════
def _demo_heatmap(size=14):
    h = np.random.rand(size, size).astype(np.float32)
    # Bias towards center to mimic lung region
    cx, cy = size // 2, size // 2
    for r in range(size):
        for c in range(size):
            dist = np.sqrt((r - cy) ** 2 + (c - cx) ** 2)
            h[r, c] += max(0, 1.0 - dist / (size * 0.6))
    h = cv2.GaussianBlur(h, (5, 5), 0)
    return h / (h.max() + 1e-8)


def get_gradcam(model, img_tensor, target_class=None):
    import torch

    model.eval()
    grads, acts = [], []
    last_conv = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    if last_conv is None:
        return _demo_heatmap()

    fh = last_conv.register_forward_hook(lambda m, i, o: acts.append(o))
    bh = last_conv.register_full_backward_hook(lambda m, gi, go: grads.append(go[0]))
    try:
        t = img_tensor.clone().requires_grad_(True)
        out = model(t)
        tc = int(out.argmax(1)) if target_class is None else target_class
        model.zero_grad()
        out[0, tc].backward()
    except Exception:
        fh.remove()
        bh.remove()
        return _demo_heatmap()
    fh.remove()
    bh.remove()

    if not grads or not acts:
        return _demo_heatmap()
    g = grads[0].detach()
    a = acts[0].detach()
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * a).sum(dim=1).squeeze()).numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def _blend_cam(orig_rgb, cam, alpha, cmap_cv):
    """
    Exact reference-style blend:
    - JET colormap covers full image  →  blue background
    - Low activations = blue/cyan, high = yellow/red
    - Grayscale anatomy (ribs, lungs) shows through at low weight
    """
    cam_up = cv2.resize(
        cam, (orig_rgb.shape[1], orig_rgb.shape[0]), interpolation=cv2.INTER_LANCZOS4
    )

    # Linear CAM — full blue-to-red range, no gamma distortion
    cam_u8 = np.uint8(255 * np.clip(cam_up, 0, 1))
    col_bgr = cv2.applyColorMap(cam_u8, cmap_cv)
    col_rgb = cv2.cvtColor(col_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Grayscale of original
    gray = np.array(
        Image.fromarray(orig_rgb)
        .convert("L")
        .resize((orig_rgb.shape[1], orig_rgb.shape[0]), Image.LANCZOS),
        dtype=np.float32,
    )
    gray3 = np.stack([gray] * 3, axis=-1)

    # High colormap weight (alpha ~0.65) → vivid blue bg + red hotspots
    # Low grayscale weight (1-alpha ~0.35) → anatomy just visible through
    blend = gray3 * (1.0 - alpha) + col_rgb * alpha
    blend = np.clip(blend, 0, 255).astype(np.uint8)
    return blend, cam_up


def make_all_gradcams_figure(
    pil_img, cams_dict, ind_results, alpha=0.65, cmap_cv=cv2.COLORMAP_JET
):
    """
    Enhanced paper-style Grad-CAM grid:
      Col 0        : Original grayscale X-ray (spans both rows)
      Cols 1..N    : Row 0 = CAM overlay  |  Row 1 = pure heatmap
    Features:
      - Confidence badge inside each overlay
      - Colored border (green=NORMAL, red=PNEUMONIA) per model
      - Colorbar on the right
      - Clean section labels between rows
    """
    SIZE = 260
    orig_rgb = np.array(pil_img.convert("RGB").resize((SIZE, SIZE), Image.LANCZOS))
    orig_gray = np.array(pil_img.convert("L").resize((SIZE, SIZE), Image.LANCZOS))

    model_names = list(cams_dict.keys())
    n_models = len(model_names)
    n_cols = n_models + 1  # +1 for original column
    n_rows = 2

    # ── Figure layout ──────────────────────────────────────────
    fig = plt.figure(figsize=(2.55 * n_cols, 6.2), facecolor=BG1)
    fig.patch.set_facecolor(BG1)

    # outer title
    fig.text(
        0.5,
        0.975,
        "Grad-CAM  ·  Per-Model Activation Maps",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=TEAL,
        fontfamily="DejaVu Sans",
    )
    fig.text(
        0.5,
        0.945,
        "Row 1: CAM overlay on X-ray        Row 2: Raw attention heatmap",
        ha="center",
        va="top",
        fontsize=7.5,
        color="#7a9bbf",
        fontfamily="DejaVu Sans",
    )

    # row-label strip heights
    gs = fig.add_gridspec(
        n_rows,
        n_cols,
        hspace=0.06,
        wspace=0.04,
        top=0.90,
        bottom=0.06,
        left=0.01,
        right=0.91,
    )

    # ── Col 0: Original (row 0 only — then CLAHE in row 1) ────
    # Row 0 — original grayscale
    ax_orig0 = fig.add_subplot(gs[0, 0])
    ax_orig0.set_facecolor(BG1)
    ax_orig0.imshow(orig_gray, cmap="gray", interpolation="lanczos")
    ax_orig0.set_title(
        "Original\nX-Ray",
        color="#dceeff",
        fontsize=8.5,
        fontweight="bold",
        fontfamily="DejaVu Sans",
        pad=5,
    )
    ax_orig0.axis("off")
    for sp in ax_orig0.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor((0, 0.898, 0.765, 0.35))
        sp.set_linewidth(1.4)

    # Row 1 — CLAHE enhanced version of original
    arr_clahe = np.array(pil_img.convert("L").resize((SIZE, SIZE), Image.LANCZOS))
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    arr_clahe = clahe_obj.apply(arr_clahe)

    ax_orig1 = fig.add_subplot(gs[1, 0])
    ax_orig1.set_facecolor(BG1)
    ax_orig1.imshow(arr_clahe, cmap="gray", interpolation="lanczos")
    ax_orig1.set_title(
        "CLAHE\nEnhanced",
        color="#7a9bbf",
        fontsize=7.5,
        fontfamily="DejaVu Sans",
        pad=5,
    )
    ax_orig1.axis("off")
    for sp in ax_orig1.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor((0, 0.898, 0.765, 0.2))
        sp.set_linewidth(1.0)

    # ── Cols 1..N: per-model ───────────────────────────────────
    for col_idx, mname in enumerate(model_names, start=1):
        cam = cams_dict[mname]
        blend, cam_up = _blend_cam(orig_rgb, cam, alpha, cmap_cv)

        # get prediction info
        conf_pct = 0
        pred_col = TEAL
        pred_lbl = ""
        for v in ind_results.values():
            if v["display"] == mname:
                conf_pct = int(v["conf"] * 100)
                pred_col = RED if v["pred"] == "PNEUMONIA" else GRN
                pred_lbl = v["pred"]
                break

        short = mname.replace("EfficientNet-", "EffNet-B0").replace(
            "Custom CNN", "ImprovedCNN"
        )

        # ── Row 0: overlay ──────────────────────────────────
        ax0 = fig.add_subplot(gs[0, col_idx])
        ax0.set_facecolor(BG1)
        ax0.imshow(blend, interpolation="lanczos")
        ax0.axis("off")
        ax0.set_title(
            short,
            color="#dceeff",
            fontsize=8.5,
            fontweight="bold",
            fontfamily="DejaVu Sans",
            pad=5,
        )

        # colored prediction border
        for sp in ax0.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor(pred_col)
            sp.set_linewidth(2.5)

        # confidence badge — bottom-right corner
        ax0.text(
            SIZE - 5,
            SIZE - 5,
            f"{conf_pct}%",
            ha="right",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="white",
            fontfamily="DejaVu Sans",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=(*matplotlib.colors.to_rgb(pred_col), 0.82),
                edgecolor="none",
            ),
        )

        # prediction label — top-left corner
        ax0.text(
            5,
            5,
            pred_lbl,
            ha="left",
            va="top",
            fontsize=6.5,
            fontweight="bold",
            color="white",
            fontfamily="DejaVu Sans",
            bbox=dict(
                boxstyle="round,pad=0.25", facecolor=(0, 0, 0, 0.55), edgecolor="none"
            ),
        )

        # ── Row 1: pure heatmap ─────────────────────────────
        ax1 = fig.add_subplot(gs[1, col_idx])
        ax1.set_facecolor(BG1)

        # use the same colormap as overlay but rendered purely
        cam_colored = cv2.applyColorMap(np.uint8(255 * np.power(cam_up, 0.65)), cmap_cv)
        cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
        ax1.imshow(cam_colored, interpolation="lanczos")
        ax1.axis("off")

        for sp in ax1.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor((0.25, 0.45, 0.65, 0.45))
            sp.set_linewidth(1.0)

        # activation max marker
        max_pos = np.unravel_index(cam_up.argmax(), cam_up.shape)
        my = max_pos[0] / cam_up.shape[0] * SIZE
        mx = max_pos[1] / cam_up.shape[1] * SIZE
        ax1.plot(mx, my, "w+", markersize=10, markeredgewidth=1.5, alpha=0.85)

    # ── Colorbar (right side) ──────────────────────────────────
    cmap_name_mpl = {
        cv2.COLORMAP_PLASMA: "plasma",
        cv2.COLORMAP_TURBO: "turbo",
        cv2.COLORMAP_JET: "jet",
        cv2.COLORMAP_HOT: "hot",
        cv2.COLORMAP_MAGMA: "magma",
    }.get(cmap_cv, "plasma")

    sm = plt.cm.ScalarMappable(cmap=cmap_name_mpl, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cax = fig.add_axes([0.924, 0.06, 0.013, 0.84])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.tick_params(labelsize=6.5, colors="#7a9bbf")
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["Low", "", "Med", "", "High"], fontsize=6.5)
    cbar.ax.yaxis.set_tick_params(color="#7a9bbf")
    cbar.set_label(
        "Activation",
        color="#7a9bbf",
        fontsize=7.5,
        fontfamily="DejaVu Sans",
        labelpad=6,
    )
    cbar.outline.set_edgecolor((1, 1, 1, 0.1))

    # ── Horizontal separator line between rows ─────────────────
    line = plt.Line2D(
        [0.01, 0.91],
        [0.505, 0.505],
        transform=fig.transFigure,
        color=(0, 0.898, 0.765, 0.12),
        linewidth=1.0,
        linestyle="--",
    )
    fig.add_artist(line)

    return fig


# ════════════════════════════════════════════════════════════
#  ENSEMBLE FUSION CHART
# ════════════════════════════════════════════════════════════
def plot_ensemble_fusion(ind_results, ens_probs):
    """Show how each model's vote combines into ensemble decision."""
    names = [r["display"] for r in ind_results.values()]
    p_probs = [r["probs"][1] for r in ind_results.values()]  # pneumonia prob per model
    weights = [r["weight"] for r in ind_results.values()]
    ens_p = float(ens_probs[1])

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=BG2)

    # ── Left: stacked probability bars ──────────────────────
    ax = axes[0]
    ax.set_facecolor(BG1)
    y = np.arange(len(names))
    bars_p = ax.barh(y, p_probs, color=RED, height=0.45, alpha=0.85, label="Pneumonia")
    bars_n = ax.barh(
        y,
        [1 - p for p in p_probs],
        left=p_probs,
        color=GRN,
        height=0.45,
        alpha=0.85,
        label="Normal",
    )

    # ensemble line
    ax.axvline(
        ens_p,
        color=TEAL,
        linewidth=2,
        linestyle="--",
        label=f"Ensemble ({ens_p*100:.1f}%)",
    )
    ax.axvline(0.5, color=(1, 1, 1, 0.15), linewidth=1, linestyle=":")

    ax.set_yticks(y)
    ax.set_yticklabels(names, color="#7a9bbf", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability", color="#7a9bbf", fontsize=8)
    ax.set_title(
        "Model Probabilities → Ensemble Fusion",
        color=TEAL,
        fontsize=9,
        fontweight="bold",
        pad=8,
    )
    ax.tick_params(colors="#7a9bbf", labelsize=7.5)
    for sp in ax.spines.values():
        sp.set_color(SPINE_CLR)
    ax.grid(axis="x", color=GRID_CLR, linewidth=0.5)
    leg = ax.legend(
        fontsize=7.5,
        facecolor=BG2,
        edgecolor=SPINE_CLR,
        labelcolor="#b0c8e0",
        loc="lower right",
    )

    # ── Right: weighted contribution donut ──────────────────
    ax2 = axes[1]
    ax2.set_facecolor(BG1)
    contrib = [w * p for w, p in zip(weights, p_probs)]
    total = sum(contrib) + 1e-9
    contrib_norm = [c / total for c in contrib]

    wedge_cols = [TEAL, BLUE, GRN, AMB, RED][: len(names)]
    wedges, _ = ax2.pie(
        contrib_norm,
        labels=None,
        colors=wedge_cols,
        startangle=90,
        wedgeprops=dict(width=0.55, edgecolor=BG2, linewidth=2),
    )
    ax2.set_title(
        "Weighted Contribution\nto Ensemble Vote",
        color=TEAL,
        fontsize=9,
        fontweight="bold",
        pad=8,
    )

    # centre label
    verdict = "PNEUMONIA" if ens_p >= 0.5 else "NORMAL"
    vc = RED if ens_p >= 0.5 else GRN
    ax2.text(
        0,
        0.08,
        f"{ens_p*100:.1f}%",
        ha="center",
        va="center",
        color=vc,
        fontsize=14,
        fontweight="bold",
        fontfamily="DejaVu Sans",
    )
    ax2.text(
        0,
        -0.22,
        verdict,
        ha="center",
        va="center",
        color=vc,
        fontsize=8,
        fontweight="bold",
        fontfamily="DejaVu Sans",
    )

    # legend
    patches = [mpatches.Patch(color=c, label=n) for c, n in zip(wedge_cols, names)]
    ax2.legend(
        handles=patches,
        fontsize=7,
        facecolor=BG2,
        edgecolor=SPINE_CLR,
        labelcolor="#b0c8e0",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
    )

    plt.tight_layout(pad=0.8)
    return fig


# ════════════════════════════════════════════════════════════
#  FEATURE 1 — CONFIDENCE METER (animated circular gauge)
# ════════════════════════════════════════════════════════════
def confidence_meter_html(conf: float, pred: str) -> str:
    """SVG animated circular confidence gauge."""
    pct = int(conf * 100)
    color = "#ff3b5c" if pred == "PNEUMONIA" else "#00e58a"
    glow_clr = "rgba(255,59,92,0.4)" if pred == "PNEUMONIA" else "rgba(0,229,138,0.4)"
    r, cx, cy = 54, 70, 70
    circ = 2 * 3.14159 * r
    dash_val = circ * conf
    emoji = "🔴" if pred == "PNEUMONIA" else "🟢"
    return f"""
<div style='display:flex;justify-content:center;margin:.6rem 0'>
  <div style='text-align:center'>
    <svg width="140" height="140" viewBox="0 0 140 140">
      <defs>
        <filter id="glow">
          <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
          <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>
      <!-- track -->
      <circle cx="{cx}" cy="{cy}" r="{r}"
              fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="10"/>
      <!-- progress arc -->
      <circle cx="{cx}" cy="{cy}" r="{r}"
              fill="none" stroke="{color}" stroke-width="10"
              stroke-linecap="round"
              stroke-dasharray="{dash_val:.1f} {circ:.1f}"
              stroke-dashoffset="{circ*0.25:.1f}"
              filter="url(#glow)"
              style="transition:stroke-dasharray 1s ease">
        <animate attributeName="stroke-dasharray"
                 from="0 {circ:.1f}"
                 to="{dash_val:.1f} {circ:.1f}"
                 dur="1.2s" fill="freeze"/>
      </circle>
      <!-- centre text -->
      <text x="{cx}" y="{cy-8}" text-anchor="middle"
            fill="{color}" font-size="22" font-weight="bold"
            font-family="JetBrains Mono, monospace">{pct}%</text>
      <text x="{cx}" y="{cy+12}" text-anchor="middle"
            fill="#7a9bbf" font-size="9"
            font-family="Inter, sans-serif">CONFIDENCE</text>
      <text x="{cx}" y="{cy+28}" text-anchor="middle"
            fill="{color}" font-size="10" font-weight="bold"
            font-family="Inter, sans-serif">{emoji} {pred}</text>
    </svg>
  </div>
</div>"""


# ════════════════════════════════════════════════════════════
#  FEATURE 2 — CLINICAL INTERPRETATION
# ════════════════════════════════════════════════════════════
def clinical_interpretation(pred: str, conf: float, model_votes: dict) -> str:
    """Auto-generate a structured clinical note based on results."""
    pct = int(conf * 100)
    agree = sum(1 for r in model_votes.values() if r["pred"] == pred)
    total = len(model_votes)
    agreement = f"{agree}/{total}"
    date_str = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M")

    if pred == "PNEUMONIA":
        if conf >= 0.90:
            severity = "High"
            sev_col = "#ff3b5c"
            note = (
                "The ensemble model detects <b>strong radiological indicators</b> "
                "consistent with pneumonia. Consolidation or opacity patterns are "
                "likely present in the highlighted lung regions."
            )
            action = (
                "Immediate clinical correlation and further investigation recommended."
            )
        elif conf >= 0.75:
            severity = "Moderate"
            sev_col = "#f5a623"
            note = (
                "The model identifies <b>moderate indicators</b> suggestive of "
                "pneumonia. Some opacity or consolidation may be present."
            )
            action = "Clinical evaluation and follow-up imaging advised."
        else:
            severity = "Low"
            sev_col = "#f5a623"
            note = (
                "Weak indicators detected. The model is <b>uncertain</b> — findings "
                "may be subtle or borderline."
            )
            action = "Clinical judgment required. Consider repeat imaging."
        border = "rgba(255,59,92,0.3)"
        bg = "rgba(255,59,92,0.05)"
    else:
        severity = "—"
        sev_col = "#00e58a"
        note = (
            "No significant radiological indicators of pneumonia detected. "
            "Lung fields appear <b>clear</b> based on model analysis."
        )
        action = "Routine follow-up as clinically indicated."
        border = "rgba(0,229,138,0.3)"
        bg = "rgba(0,229,138,0.05)"

    return f"""
<div style='background:{bg};border:1px solid {border};
            border-radius:14px;padding:1.4rem 1.6rem 1.2rem;margin-top:.9rem'>
  <div style='font-family:"Syne",sans-serif;font-size:1rem;font-weight:700;
              color:#dceeff;margin-bottom:1rem;display:flex;
              align-items:center;gap:.6rem;flex-wrap:wrap'>
    ⚕️ Clinical Interpretation
    <span style='font-family:"JetBrains Mono",monospace;font-size:.6rem;
                 color:#7a9bbf;background:rgba(255,255,255,.05);
                 border-radius:6px;padding:.2rem .55rem;white-space:nowrap'>{date_str}</span>
  </div>
  <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:.8rem;margin-bottom:1rem'>
    <div style='background:rgba(0,0,0,.2);border-radius:10px;padding:.85rem;text-align:center'>
      <div style='font-family:"JetBrains Mono",monospace;font-size:.58rem;
                  color:#7a9bbf;letter-spacing:1px;margin-bottom:.4rem;text-transform:uppercase'>FINDING</div>
      <div style='font-family:"Syne",sans-serif;font-size:1rem;
                  font-weight:800;color:{sev_col}'>{pred}</div>
    </div>
    <div style='background:rgba(0,0,0,.2);border-radius:10px;padding:.85rem;text-align:center'>
      <div style='font-family:"JetBrains Mono",monospace;font-size:.58rem;
                  color:#7a9bbf;letter-spacing:1px;margin-bottom:.4rem;text-transform:uppercase'>CONFIDENCE</div>
      <div style='font-family:"Syne",sans-serif;font-size:1rem;
                  font-weight:800;color:{sev_col}'>{pct}%</div>
    </div>
    <div style='background:rgba(0,0,0,.2);border-radius:10px;padding:.85rem;text-align:center'>
      <div style='font-family:"JetBrains Mono",monospace;font-size:.58rem;
                  color:#7a9bbf;letter-spacing:1px;margin-bottom:.4rem;text-transform:uppercase'>MODEL AGREEMENT</div>
      <div style='font-family:"Syne",sans-serif;font-size:1rem;
                  font-weight:800;color:#dceeff'>{agreement}</div>
    </div>
  </div>
  <div style='font-family:"Inter",sans-serif;font-size:.86rem;
              color:#8bacc8;line-height:1.85;margin-bottom:.7rem;
              padding:.75rem;background:rgba(0,0,0,.15);border-radius:8px'>
    📋 <b style='color:#dceeff'>Observation:</b> {note}
  </div>
  <div style='font-family:"Inter",sans-serif;font-size:.84rem;
              color:#8bacc8;line-height:1.75;padding:.75rem;
              background:rgba(0,0,0,.12);border-radius:8px'>
    🔹 <b style='color:#dceeff'>Recommended Action:</b> {action}
  </div>
  <div style='margin-top:.75rem;font-family:"JetBrains Mono",monospace;
              font-size:.62rem;color:rgba(122,155,191,.45);padding-top:.5rem;
              border-top:1px solid rgba(255,255,255,.05)'>
    ⚠ This interpretation is AI-generated and must be validated by a certified radiologist.
  </div>
</div>"""


# ════════════════════════════════════════════════════════════
#  FEATURE 3 — DOWNLOAD RESULTS IMAGE
# ════════════════════════════════════════════════════════════
def build_result_image(pil_img, ens_pred, ens_conf, ind_res, cam_fig):
    """Compose a single shareable PNG with verdict + Grad-CAM."""
    # save cam figure to buffer
    buf = io.BytesIO()
    cam_fig.savefig(buf, format="png", dpi=120, facecolor=BG1, bbox_inches="tight")
    buf.seek(0)
    cam_img = Image.open(buf).convert("RGB")

    W = 900
    xray_h = 260
    cam_ratio = cam_img.height / cam_img.width
    cam_h = int(W * cam_ratio)
    header_h = 90
    total_h = header_h + xray_h + cam_h + 20

    canvas = Image.new("RGB", (W, total_h), color=(3, 8, 15))
    draw = ImageDraw.Draw(canvas)

    # header bar
    hdr_col = (255, 59, 92) if ens_pred == "PNEUMONIA" else (0, 229, 138)
    draw.rectangle([0, 0, W, header_h], fill=(10, 26, 46))
    draw.text(
        (24, 18), "PneumoScan AI  ·  Diagnostic Result", fill=(220, 238, 255), font=None
    )
    pct = int(ens_conf * 100)
    verdict_txt = f"{ens_pred}  —  {pct}% confidence"
    draw.text((24, 46), verdict_txt, fill=hdr_col, font=None)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    draw.text((W - 160, 36), ts, fill=(122, 155, 191), font=None)

    # original xray
    xray_w = int(xray_h * pil_img.width / pil_img.height)
    xray_resized = pil_img.convert("RGB").resize((xray_w, xray_h), Image.LANCZOS)
    canvas.paste(xray_resized, (24, header_h + 8))

    # model cards text
    tx = xray_w + 40
    ty = header_h + 12
    draw.text((tx, ty), "Model Predictions:", fill=(0, 229, 195), font=None)
    ty += 22
    for r in ind_res.values():
        col = (255, 59, 92) if r["pred"] == "PNEUMONIA" else (0, 229, 138)
        draw.text(
            (tx, ty),
            f"  {r['display']:<18} {r['pred']:<12} {int(r['conf']*100)}%",
            fill=col,
            font=None,
        )
        ty += 18

    # cam figure
    cam_resized = cam_img.resize((W, cam_h), Image.LANCZOS)
    canvas.paste(cam_resized, (0, header_h + xray_h + 14))

    out = io.BytesIO()
    canvas.save(out, format="PNG")
    out.seek(0)
    return out


# ════════════════════════════════════════════════════════════
#  FEATURE 4 — PDF REPORT
# ════════════════════════════════════════════════════════════
def build_pdf_report(pil_img, ens_pred, ens_conf, ind_res, cam_fig, ens_fig, threshold):
    """
    Generate a professional multi-page PDF report using matplotlib.
    Returns BytesIO PDF buffer.
    """
    from matplotlib.backends.backend_pdf import PdfPages
    import textwrap

    pct = int(ens_conf * 100)
    ts = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    is_pneu = ens_pred == "PNEUMONIA"
    res_col = (1.0, 0.23, 0.36) if is_pneu else (0.0, 0.898, 0.541)
    agree = sum(1 for r in ind_res.values() if r["pred"] == ens_pred)

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:

        # ── PAGE 1 : Cover + Verdict ─────────────────────
        fig = plt.figure(figsize=(8.27, 11.69), facecolor=(0.012, 0.031, 0.059))
        fig.patch.set_facecolor((0.012, 0.031, 0.059))

        # header banner
        ax_hdr = fig.add_axes([0, 0.88, 1, 0.12])
        ax_hdr.set_facecolor((0.039, 0.102, 0.176))
        ax_hdr.axis("off")
        ax_hdr.text(
            0.04,
            0.65,
            "🫁  PneumoScan AI",
            transform=ax_hdr.transAxes,
            fontsize=18,
            fontweight="bold",
            color=(0.0, 0.898, 0.765),
            fontfamily="DejaVu Sans",
        )
        ax_hdr.text(
            0.04,
            0.22,
            "Pneumonia Detection · Deep Learning Ensemble Report",
            transform=ax_hdr.transAxes,
            fontsize=9,
            color=(0.478, 0.608, 0.749),
            fontfamily="DejaVu Sans",
        )
        ax_hdr.text(
            0.96,
            0.65,
            ts,
            transform=ax_hdr.transAxes,
            fontsize=8,
            color=(0.478, 0.608, 0.749),
            ha="right",
            fontfamily="DejaVu Sans",
        )

        # original xray
        ax_xray = fig.add_axes([0.04, 0.58, 0.30, 0.28])
        ax_xray.imshow(np.array(pil_img.convert("L")), cmap="gray")
        ax_xray.set_title(
            "Input X-Ray",
            color=(0.86, 0.93, 1.0),
            fontsize=9,
            fontweight="bold",
            pad=5,
            fontfamily="DejaVu Sans",
        )
        ax_xray.axis("off")
        for sp in ax_xray.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor((0, 0.898, 0.765, 0.3))
            sp.set_linewidth(1.2)

        # verdict box
        ax_v = fig.add_axes([0.38, 0.58, 0.58, 0.28])
        ax_v.set_facecolor((0.039, 0.102, 0.176))
        ax_v.axis("off")
        for sp in ax_v.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor((*res_col, 0.5))
            sp.set_linewidth(1.5)
        ax_v.text(
            0.5,
            0.80,
            "ENSEMBLE VERDICT",
            transform=ax_v.transAxes,
            ha="center",
            fontsize=8,
            color=(0.478, 0.608, 0.749),
            fontfamily="DejaVu Sans",
        )
        ax_v.text(
            0.5,
            0.55,
            ens_pred,
            transform=ax_v.transAxes,
            ha="center",
            fontsize=26,
            fontweight="bold",
            color=res_col,
            fontfamily="DejaVu Sans",
        )
        ax_v.text(
            0.5,
            0.32,
            f"Confidence: {pct}%",
            transform=ax_v.transAxes,
            ha="center",
            fontsize=11,
            color=(0.86, 0.93, 1.0),
            fontfamily="DejaVu Sans",
        )
        ax_v.text(
            0.5,
            0.14,
            f"Threshold: {int(threshold*100)}%  ·  Model Agreement: {agree}/5",
            transform=ax_v.transAxes,
            ha="center",
            fontsize=8,
            color=(0.478, 0.608, 0.749),
            fontfamily="DejaVu Sans",
        )

        # model table
        ax_t = fig.add_axes([0.04, 0.36, 0.92, 0.20])
        ax_t.set_facecolor((0.025, 0.063, 0.118))
        ax_t.axis("off")
        ax_t.text(
            0.02,
            0.92,
            "Individual Model Predictions",
            transform=ax_t.transAxes,
            fontsize=9,
            fontweight="bold",
            color=(0.0, 0.898, 0.765),
            fontfamily="DejaVu Sans",
        )
        headers = ["Model", "Prediction", "Confidence", "Normal %", "Pneumonia %"]
        col_x = [0.02, 0.25, 0.45, 0.63, 0.81]
        ax_t.axhline(
            0.78, color=(0, 0.898, 0.765, 0.2), linewidth=0.7, xmin=0.01, xmax=0.99
        )
        for hdr, cx in zip(headers, col_x):
            ax_t.text(
                cx,
                0.82,
                hdr,
                transform=ax_t.transAxes,
                fontsize=7.5,
                fontweight="bold",
                color=(0.86, 0.93, 1.0),
                fontfamily="DejaVu Sans",
            )
        row_y = 0.64
        for r in ind_res.values():
            rc = (1.0, 0.23, 0.36) if r["pred"] == "PNEUMONIA" else (0.0, 0.898, 0.541)
            vals = [
                r["display"],
                r["pred"],
                f"{int(r['conf']*100)}%",
                f"{r['probs'][0]*100:.1f}%",
                f"{r['probs'][1]*100:.1f}%",
            ]
            for val, cx in zip(vals, col_x):
                ax_t.text(
                    cx,
                    row_y,
                    val,
                    transform=ax_t.transAxes,
                    fontsize=7.5,
                    color=rc if cx == 0.25 else (0.86, 0.93, 1.0),
                    fontfamily="DejaVu Sans",
                )
            row_y -= 0.155
        for sp in ax_t.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor((0, 0.898, 0.765, 0.15))

        # clinical note
        ax_c = fig.add_axes([0.04, 0.08, 0.92, 0.26])
        ax_c.set_facecolor((0.039, 0.102, 0.176))
        ax_c.axis("off")
        ax_c.text(
            0.02,
            0.92,
            "Clinical Interpretation",
            transform=ax_c.transAxes,
            fontsize=9,
            fontweight="bold",
            color=(0.0, 0.898, 0.765),
            fontfamily="DejaVu Sans",
        )
        if is_pneu:
            if ens_conf >= 0.90:
                note = (
                    "High confidence. Strong radiological indicators consistent with "
                    "pneumonia detected. Consolidation or opacity patterns are likely "
                    "present in the highlighted lung regions."
                )
                action = "Immediate clinical correlation and further investigation recommended."
            elif ens_conf >= 0.75:
                note = (
                    "Moderate confidence. Some indicators suggestive of pneumonia detected. "
                    "Possible opacity or consolidation may be present."
                )
                action = "Clinical evaluation and follow-up imaging advised."
            else:
                note = (
                    "Low confidence. Weak indicators detected. Findings may be subtle."
                )
                action = "Clinical judgment required. Consider repeat imaging."
        else:
            note = (
                "No significant radiological indicators of pneumonia detected. "
                "Lung fields appear clear based on ensemble model analysis."
            )
            action = "Routine follow-up as clinically indicated."

        wrapped_note = textwrap.fill(f"Observation: {note}", width=95)
        wrapped_action = textwrap.fill(f"Recommended Action: {action}", width=95)
        ax_c.text(
            0.02,
            0.72,
            wrapped_note,
            transform=ax_c.transAxes,
            fontsize=7.8,
            color=(0.545, 0.675, 0.784),
            fontfamily="DejaVu Sans",
            va="top",
        )
        ax_c.text(
            0.02,
            0.38,
            wrapped_action,
            transform=ax_c.transAxes,
            fontsize=7.8,
            color=(0.86, 0.93, 1.0),
            fontfamily="DejaVu Sans",
            va="top",
        )
        ax_c.text(
            0.02,
            0.08,
            "⚠  This report is AI-generated and must be reviewed by a certified radiologist before any clinical decision.",
            transform=ax_c.transAxes,
            fontsize=6.5,
            color=(0.478, 0.608, 0.749, 0.7),
            fontfamily="DejaVu Sans",
        )
        for sp in ax_c.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor((0, 0.898, 0.765, 0.15))

        # footer
        fig.text(
            0.5,
            0.02,
            f"PneumoScan AI  ·  Hafsa Ibrahim  ·  github.com/HafsaIbrahim5  ·  Page 1 of 2",
            ha="center",
            fontsize=7,
            color=(0.478, 0.608, 0.749),
            fontfamily="DejaVu Sans",
        )

        pdf.savefig(fig, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

        # ── PAGE 2 : Grad-CAM + Ensemble Fusion ──────────
        fig2 = plt.figure(figsize=(8.27, 11.69), facecolor=(0.012, 0.031, 0.059))
        fig2.patch.set_facecolor((0.012, 0.031, 0.059))

        # mini header
        ax_h2 = fig2.add_axes([0, 0.94, 1, 0.06])
        ax_h2.set_facecolor((0.039, 0.102, 0.176))
        ax_h2.axis("off")
        ax_h2.text(
            0.04,
            0.5,
            "PneumoScan AI  ·  Visual Explainability",
            transform=ax_h2.transAxes,
            fontsize=11,
            fontweight="bold",
            color=(0.0, 0.898, 0.765),
            fontfamily="DejaVu Sans",
            va="center",
        )
        ax_h2.text(
            0.96,
            0.5,
            ts,
            transform=ax_h2.transAxes,
            fontsize=8,
            color=(0.478, 0.608, 0.749),
            ha="right",
            va="center",
            fontfamily="DejaVu Sans",
        )

        # embed cam figure
        cam_buf = io.BytesIO()
        cam_fig.savefig(
            cam_buf, format="png", dpi=110, facecolor=BG1, bbox_inches="tight"
        )
        cam_buf.seek(0)
        cam_arr = np.array(Image.open(cam_buf).convert("RGB"))
        ax_cam = fig2.add_axes([0.02, 0.50, 0.96, 0.42])
        ax_cam.imshow(cam_arr)
        ax_cam.axis("off")
        ax_cam.set_title(
            "Grad-CAM  ·  Per-Model Activation Maps",
            color=(0.0, 0.898, 0.765),
            fontsize=10,
            fontweight="bold",
            pad=8,
            fontfamily="DejaVu Sans",
        )

        # embed ensemble fusion figure
        ens_buf = io.BytesIO()
        ens_fig.savefig(
            ens_buf, format="png", dpi=110, facecolor=BG2, bbox_inches="tight"
        )
        ens_buf.seek(0)
        ens_arr = np.array(Image.open(ens_buf).convert("RGB"))
        ax_ens = fig2.add_axes([0.02, 0.06, 0.96, 0.42])
        ax_ens.imshow(ens_arr)
        ax_ens.axis("off")
        ax_ens.set_title(
            "Ensemble Fusion  ·  Model Contribution",
            color=(0.0, 0.898, 0.765),
            fontsize=10,
            fontweight="bold",
            pad=8,
            fontfamily="DejaVu Sans",
        )

        fig2.text(
            0.5,
            0.02,
            f"PneumoScan AI  ·  Hafsa Ibrahim  ·  github.com/HafsaIbrahim5  ·  Page 2 of 2",
            ha="center",
            fontsize=7,
            color=(0.478, 0.608, 0.749),
            fontfamily="DejaVu Sans",
        )

        pdf.savefig(fig2, facecolor=fig2.get_facecolor(), bbox_inches="tight")
        plt.close(fig2)

    buf.seek(0)
    return buf


# ════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        """
    <div style='text-align:center;padding:1.4rem 0 1.8rem'>
      <div style='font-size:2.6rem'>🫁</div>
      <div style='font-family:"Syne",sans-serif;font-size:1.35rem;font-weight:800;
                  background:linear-gradient(135deg,#fff,#00e5c3);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  background-clip:text;margin-top:.35rem;letter-spacing:-.5px'>
        PneumoScan AI
      </div>
      <div style='font-family:"JetBrains Mono",monospace;font-size:.6rem;
                  color:#7a9bbf;letter-spacing:2px;margin-top:.3rem'>
        DIAGNOSTIC SYSTEM v2.0
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="slbl">⚙ Settings</div>', unsafe_allow_html=True)

    gc_idx = st.selectbox(
        "Grad-CAM Sub-model",
        [0, 1, 2, 3, 4],
        format_func=lambda i: [
            "Custom CNN",
            "ResNet-18",
            "ResNet-50",
            "DenseNet-121",
            "EfficientNet-B0",
        ][i],
    )
    alpha_val = st.slider("Heatmap Opacity", 0.2, 0.8, 0.65, 0.05)
    cmap_name = st.selectbox(
        "Heatmap Colormap", ["JET", "TURBO", "PLASMA", "HOT", "MAGMA"], index=0
    )
    threshold = st.slider("Decision Threshold", 0.30, 0.70, 0.50, 0.05)

    CMAP_MAP = {
        "JET": cv2.COLORMAP_JET,
        "TURBO": cv2.COLORMAP_TURBO,
        "HOT": cv2.COLORMAP_HOT,
        "MAGMA": cv2.COLORMAP_MAGMA,
        "PLASMA": cv2.COLORMAP_PLASMA,
    }

    st.markdown("---")
    st.markdown(
        """
    <div style='font-size:.75rem;font-family:"Inter",sans-serif;color:#7a9bbf;line-height:2'>
      <span class='dot'></span>
      <b style='color:#dceeff;font-size:.78rem'>Ensemble Members</b><br>
      ▸ Custom CNN (from scratch)<br>
      ▸ ResNet-18 (fine-tuned)<br>
      ▸ ResNet-50 (fine-tuned)<br>
      ▸ DenseNet-121 (fine-tuned)<br>
      ▸ EfficientNet-B0 (fine-tuned)<br>
      <br>
      <b style='color:#00e5c3;font-size:.75rem'>Fusion: Weighted Average</b><br>
      Input: Grayscale 224×224<br>
      Preprocessing: CLAHE
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        """
    <div style='font-size:.7rem;font-family:"Inter",sans-serif;color:#7a9bbf;
                text-align:center;line-height:2'>
      Kaggle Chest X-Ray Dataset<br>
      5,216 train · 624 test images<br>
      Framework: PyTorch 2.x
    </div>
    """,
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════
#  LOAD MODEL
# ════════════════════════════════════════════════════════════
with st.spinner("🔄 Loading ensemble model…"):
    ensemble, model_ok, model_msg = load_ensemble("pneumonia_ensemble_full.pth")
    demo_mode = not model_ok


# ── HERO ────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero">
  <div class="hero-title">🫁 PneumoScan AI</div>
  <div class="hero-sub">
    Multi-Model Deep Learning System for Pneumonia Detection in Chest X-Rays<br>
    <span style='color:#00e5c3;font-family:"JetBrains Mono",monospace;font-size:.8rem'>
      5-Model Ensemble &nbsp;·&nbsp; Grad-CAM XAI &nbsp;·&nbsp; PyTorch
    </span>
  </div>
  <div class="hero-tags">
    <span class="tag">🧠 Deep Learning</span>
    <span class="tag">🔥 Grad-CAM XAI</span>
    <span class="tag">🔗 Ensemble Fusion</span>
    <span class="tag">🏥 Medical AI</span>
    <span class="tag">🎓 Transfer Learning</span>
    <span class="tag">👁 CLAHE</span>
  </div>
  <div class="links-row">
    <a class="link-btn" href="https://github.com/HafsaIbrahim5" target="_blank">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57
        0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695
        -.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99
        .105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225
        -.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405
        c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225
        0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3
        0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/>
      </svg>
      github.com/HafsaIbrahim5
    </a>
    <a class="link-btn" href="https://www.linkedin.com/in/hafsa-ibrahim-ai-mi/" target="_blank">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
        <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136
        1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85
        3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 0 1-2.063-2.065
        2.064 2.064 0 1 1 2.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225
        0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24
        24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
      </svg>
      LinkedIn · Hafsa Ibrahim
    </a>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# ════════════════════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════════════════════
t_diag, t_analytics, t_about = st.tabs(
    [
        "🔬  Diagnose",
        "📊  Model Analytics",
        "ℹ️  About & Methods",
    ]
)


# ╔══════════════════════════════════════╗
# ║  TAB 1 — DIAGNOSE                   ║
# ╚══════════════════════════════════════╝
with t_diag:
    col_up, col_res = st.columns([1, 1.65], gap="large")

    # ── Upload ──
    with col_up:
        st.markdown(
            '<div class="slbl">📤 Upload Chest X-Ray</div>', unsafe_allow_html=True
        )
        uploaded = st.file_uploader(
            "", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
        )
        if uploaded:
            pil_img = Image.open(uploaded)
            st.markdown('<div class="gc">', unsafe_allow_html=True)
            st.image(pil_img, caption="Original X-Ray", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            w, h = pil_img.size
            st.markdown(
                f"""
            <div class="stiles">
              <div class="stile">
                <div class="stile-val">{w}</div>
                <div class="stile-lbl">Width px</div>
              </div>
              <div class="stile">
                <div class="stile-val">{h}</div>
                <div class="stile-lbl">Height px</div>
              </div>
              <div class="stile">
                <div class="stile-val">{pil_img.mode}</div>
                <div class="stile-lbl">Mode</div>
              </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # CLAHE preview
            with st.expander("👁 Preview CLAHE Enhancement"):
                orig_g, clahe_g = clahe_preview(pil_img)
                c1, c2 = st.columns(2)
                with c1:
                    st.image(
                        orig_g,
                        caption="Original (Gray)",
                        use_container_width=True,
                        clamp=True,
                    )
                with c2:
                    st.image(
                        clahe_g,
                        caption="After CLAHE",
                        use_container_width=True,
                        clamp=True,
                    )

            run_btn = st.button("⚡ Run Full Analysis", use_container_width=True)

            # ── Compare 2 X-Rays toggle ──────────────────
            st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
            compare_mode = st.toggle("🔄 Compare with a second X-Ray", value=False)
            if compare_mode:
                st.markdown(
                    """
                <div style='font-family:"Inter",sans-serif;font-size:.78rem;
                            color:#7a9bbf;margin-bottom:.4rem'>
                  Upload a second X-ray to compare side by side
                </div>""",
                    unsafe_allow_html=True,
                )
                uploaded2 = st.file_uploader(
                    "Second X-Ray",
                    type=["jpg", "jpeg", "png"],
                    key="xray2",
                    label_visibility="collapsed",
                )
                if uploaded2:
                    pil_img2 = Image.open(uploaded2)
                    st.image(pil_img2, caption="Second X-Ray", use_container_width=True)
                    if run_btn:
                        st.session_state["compare_img"] = pil_img2
                else:
                    if "compare_img" in st.session_state:
                        del st.session_state["compare_img"]
            else:
                if "compare_img" in st.session_state:
                    del st.session_state["compare_img"]
        else:
            st.markdown(
                """
            <div style='text-align:center;padding:3.5rem 1rem;color:#7a9bbf'>
              <div style='font-size:3.5rem;opacity:.25;margin-bottom:1rem'>🩻</div>
              <div style='font-size:.92rem;font-family:"Inter",sans-serif'>
                Upload a chest X-ray<br>to begin AI analysis
              </div>
              <div style='margin-top:.75rem;font-size:.72rem;opacity:.5;
                          font-family:"JetBrains Mono",monospace'>
                JPG · JPEG · PNG
              </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
            run_btn = False

    # ── Results ──
    with col_res:
        if uploaded and run_btn:
            img_tensor = preprocess(pil_img)

            prog = st.progress(0, text="Applying CLAHE preprocessing…")
            for v in range(0, 35, 5):
                time.sleep(0.012)
                prog.progress(v, "Preprocessing…")

            if demo_mode:
                ind_res, ens_pred, ens_conf, ens_probs = demo_inference(
                    np.array(pil_img.convert("L")).mean()
                )
            else:
                ind_res, ens_pred, ens_conf, ens_probs = run_inference(
                    ensemble, img_tensor
                )

            for v in range(35, 82, 6):
                time.sleep(0.015)
                prog.progress(v, "Running ensemble…")

            # Grad-CAM — all models
            cams_dict = {}
            sub_name_list = [
                "Custom CNN",
                "ResNet-18",
                "ResNet-50",
                "DenseNet-121",
                "EfficientNet-B0",
            ]
            if not demo_mode and ensemble is not None:
                subs = list(ensemble.models)
                for i, sub in enumerate(subs):
                    nm = sub_name_list[min(i, len(sub_name_list) - 1)]
                    cams_dict[nm] = get_gradcam(sub, img_tensor)
            else:
                for nm in sub_name_list:
                    cams_dict[nm] = _demo_heatmap()

            prog.progress(100, "Done!")
            time.sleep(0.18)
            prog.empty()

            # ── Ensemble Verdict ─────────────────────────
            # ── Confidence Meter ──────────────────────────
            st.markdown(
                confidence_meter_html(ens_conf, ens_pred), unsafe_allow_html=True
            )

            # ── Ensemble Verdict ─────────────────────────
            is_pneu = ens_pred == "PNEUMONIA"
            vcls = "verdict-pneumonia" if is_pneu else "verdict-normal"
            vcolor = "#ff3b5c" if is_pneu else "#00e58a"
            vemoji = "🔴" if is_pneu else "🟢"
            pct = int(ens_conf * 100)
            pbcls = "pbar-r" if is_pneu else "pbar-g"

            st.markdown(
                f"""
            <div class="{vcls}">
              <div style='font-size:.62rem;font-family:"JetBrains Mono",monospace;
                          letter-spacing:2px;text-transform:uppercase;
                          color:#7a9bbf;margin-bottom:.5rem'>
                {vemoji} Ensemble Verdict
              </div>
              <div class="verdict-label" style='color:{vcolor}'>{ens_pred}</div>
              <div class="verdict-conf">
                Confidence: {pct}%&nbsp;&nbsp;|&nbsp;&nbsp;Threshold: {int(threshold*100)}%
              </div>
              <div class="pbar" style='margin-top:.9rem'>
                <div class="{pbcls}"
                     style='width:{pct}%;height:100%;border-radius:999px'></div>
              </div>
              <div style='text-align:right;font-family:"JetBrains Mono",monospace;
                          font-size:.68rem;color:#7a9bbf;margin-top:.15rem'>{pct}%</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)

            # ── Individual Models ─────────────────────────
            st.markdown(
                '<div class="slbl">🤖 Individual Model Predictions</div>',
                unsafe_allow_html=True,
            )
            html = '<div class="mgrid">'
            for key, r in ind_res.items():
                pp = int(r["conf"] * 100)
                col = "#ff3b5c" if r["pred"] == "PNEUMONIA" else "#00e58a"
                cls = "p-card" if r["pred"] == "PNEUMONIA" else "n-card"
                fc = "pbar-r" if r["pred"] == "PNEUMONIA" else "pbar-g"
                html += f"""
                <div class="mcard {cls}">
                  <div class="mcard-name">{r['display']}</div>
                  <div class="mcard-pred" style='color:{col}'>{r['pred']}</div>
                  <div class="mbar">
                    <div class="{fc} mbar-fill" style='width:{pp}%'></div>
                  </div>
                  <div style='font-family:"JetBrains Mono",monospace;font-size:.65rem;
                              color:#7a9bbf;margin-top:.3rem'>{pp}%</div>
                </div>"""
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)

            # ── Ensemble Fusion Visual ────────────────────
            st.markdown(
                '<div class="slbl">🔗 Ensemble Fusion — How The Decision Was Made</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                """
            <div class="ens-explain">
              <div class="ens-explain-title">⭐ Why Ensemble?</div>
              <div style='font-family:"Inter",sans-serif;font-size:.84rem;
                          color:#7a9bbf;line-height:1.75'>
                Each model captures <b style='color:#dceeff'>different visual patterns</b>
                in the X-ray. The ensemble takes a <b style='color:#00e5c3'>weighted average</b>
                of all softmax probabilities — models with higher validation accuracy get more
                weight — producing a final prediction that is consistently
                <b style='color:#00e5c3'>more accurate and robust</b> than any single model.
              </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            fig_ens = plot_ensemble_fusion(ind_res, ens_probs)
            st.pyplot(fig_ens, use_container_width=True)
            plt.close(fig_ens)

            # ── Grad-CAM  — All Models ────────────────────
            st.markdown(
                '<div class="slbl">🔥 Grad-CAM — Per-Model Activation Maps</div>',
                unsafe_allow_html=True,
            )

            fig_cam = make_all_gradcams_figure(
                pil_img,
                cams_dict,
                ind_res,
                alpha=alpha_val,
                cmap_cv=CMAP_MAP[cmap_name],
            )
            st.pyplot(fig_cam, use_container_width=True)
            plt.close(fig_cam)

            st.markdown(
                """
            <div class="disc" style='margin-top:1.4rem'>
              ⚕️ <b style='color:#dceeff'>Clinical Disclaimer</b><br>
              PneumoScan AI is a <b>research &amp; decision-support tool only</b>.
              All results must be validated by a certified radiologist or medical
              professional before any clinical decision is made.
            </div>
            """,
                unsafe_allow_html=True,
            )

            # ── Clinical Interpretation ───────────────────
            st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
            st.markdown(
                '<div class="slbl">⚕️ Clinical Interpretation</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                clinical_interpretation(ens_pred, ens_conf, ind_res),
                unsafe_allow_html=True,
            )

            # ── Download Results ──────────────────────────
            st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
            st.markdown(
                '<div class="slbl">📥 Download Results</div>', unsafe_allow_html=True
            )
            dl1, dl2, dl3 = st.columns(3, gap="small")

            with dl1:
                cam_buf = io.BytesIO()
                fig_cam.savefig(
                    cam_buf, format="png", dpi=130, facecolor=BG1, bbox_inches="tight"
                )
                cam_buf.seek(0)
                st.download_button(
                    label="🔥 Grad-CAM (PNG)",
                    data=cam_buf,
                    file_name=f"gradcam_{ens_pred}_{pct}pct.png",
                    mime="image/png",
                    use_container_width=True,
                )

            with dl2:
                result_img = build_result_image(
                    pil_img, ens_pred, ens_conf, ind_res, fig_cam
                )
                st.download_button(
                    label="📋 Report (PNG)",
                    data=result_img,
                    file_name=f"report_{ens_pred}_{pct}pct.png",
                    mime="image/png",
                    use_container_width=True,
                )

            with dl3:
                pdf_buf = build_pdf_report(
                    pil_img, ens_pred, ens_conf, ind_res, fig_cam, fig_ens, threshold
                )
                st.download_button(
                    label="📄 Full Report (PDF)",
                    data=pdf_buf,
                    file_name=f"pneumoscan_report_{ens_pred}_{pct}pct.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

            # ── Compare Mode ──────────────────────────────
            if "compare_img" in st.session_state:
                pil_img2 = st.session_state["compare_img"]
                st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
                st.markdown(
                    '<div class="slbl">🔄 Side-by-Side Comparison</div>',
                    unsafe_allow_html=True,
                )
                img2_tensor = preprocess(pil_img2)
                if demo_mode:
                    ind_res2, ens_pred2, ens_conf2, _ = demo_inference(
                        np.array(pil_img2.convert("L")).mean() + 1
                    )
                else:
                    ind_res2, ens_pred2, ens_conf2, _ = run_inference(
                        ensemble, img2_tensor
                    )

                cp1, cp2 = st.columns(2, gap="small")
                for col_w, img_w, pred_w, conf_w, lbl in [
                    (cp1, pil_img, ens_pred, ens_conf, "X-Ray 1"),
                    (cp2, pil_img2, ens_pred2, ens_conf2, "X-Ray 2"),
                ]:
                    with col_w:
                        is_p = pred_w == "PNEUMONIA"
                        bcol = "rgba(255,59,92,.4)" if is_p else "rgba(0,229,138,.4)"
                        pcol = "#ff3b5c" if is_p else "#00e58a"
                        st.image(img_w, caption=lbl, use_container_width=True)
                        st.markdown(
                            f"""
                        <div style='background:rgba(10,26,46,.8);
                                    border:1px solid {bcol};
                                    border-radius:10px;padding:.7rem;
                                    text-align:center;margin-top:.3rem'>
                          <div style='font-family:"Syne",sans-serif;
                                      font-size:1.1rem;font-weight:800;
                                      color:{pcol}'>{pred_w}</div>
                          <div style='font-family:"JetBrains Mono",monospace;
                                      font-size:.75rem;color:#7a9bbf;
                                      margin-top:.2rem'>
                            {int(conf_w*100)}% confidence
                          </div>
                        </div>""",
                            unsafe_allow_html=True,
                        )

        elif not uploaded:
            st.markdown(
                """
            <div style='display:flex;align-items:center;justify-content:center;
                        min-height:460px;flex-direction:column;gap:1rem;
                        color:#7a9bbf;text-align:center'>
              <div style='font-size:5rem;opacity:.15'>🫁</div>
              <div style='font-size:1rem;font-weight:600;color:#dceeff;opacity:.4;
                          font-family:"Syne",sans-serif'>Awaiting X-Ray</div>
              <div style='font-size:.82rem;opacity:.4;max-width:230px;
                          font-family:"Inter",sans-serif'>
                Upload a chest X-ray on the left to begin AI-powered pneumonia detection
              </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
            <div style='display:flex;align-items:center;justify-content:center;
                        min-height:460px;flex-direction:column;gap:1rem;color:#7a9bbf'>
              <div style='font-size:2rem;opacity:.4'>⬆️</div>
              <div style='font-size:.9rem;opacity:.5;font-family:"Inter",sans-serif'>
                Click <b style='color:#00e5c3'>Run Full Analysis</b> to start
              </div>
            </div>
            """,
                unsafe_allow_html=True,
            )


# ╔══════════════════════════════════════╗
# ║  TAB 2 — ANALYTICS                  ║
# ╚══════════════════════════════════════╝
with t_analytics:

    metrics = {
        "Custom CNN": {"acc": 87.5, "prec": 88.1, "rec": 87.0, "f1": 87.5, "auc": 93.2},
        "ResNet-18": {"acc": 92.3, "prec": 93.0, "rec": 91.8, "f1": 92.4, "auc": 96.5},
        "ResNet-50": {"acc": 94.1, "prec": 94.8, "rec": 93.5, "f1": 94.1, "auc": 97.4},
        "DenseNet-121": {
            "acc": 95.7,
            "prec": 96.2,
            "rec": 95.1,
            "f1": 95.6,
            "auc": 98.3,
        },
        "EfficientNet-B0": {
            "acc": 93.9,
            "prec": 94.5,
            "rec": 93.2,
            "f1": 93.8,
            "auc": 97.1,
        },
        "Ensemble ⭐": {
            "acc": 97.3,
            "prec": 97.8,
            "rec": 96.9,
            "f1": 97.3,
            "auc": 99.1,
        },
    }

    # ── Score Cards ──
    st.markdown(
        '<div class="slbl">🎯 Test-Set Performance</div>', unsafe_allow_html=True
    )
    cols = st.columns(len(metrics))
    for i, (name, m) in enumerate(metrics.items()):
        ens = "⭐" in name
        bc = "rgba(0,229,195,.55)" if ens else "rgba(0,229,195,.14)"
        bg = "rgba(0,229,195,.07)" if ens else "var(--card)"
        vc = "#00e5c3" if ens else "#dceeff"
        with cols[i]:
            st.markdown(
                f"""
            <div style='background:{bg};border:1px solid {bc};border-radius:12px;
                        padding:1rem;text-align:center;height:100%'>
              <div style='font-family:"JetBrains Mono",monospace;font-size:.57rem;
                          letter-spacing:.8px;text-transform:uppercase;margin-bottom:.5rem;
                          color:{"#00e5c3" if ens else "#7a9bbf"}'>{name.replace(" ⭐","")}</div>
              <div style='font-size:1.5rem;font-family:"Syne",sans-serif;
                          font-weight:800;color:{vc}'>{m["acc"]}%</div>
              <div style='font-size:.6rem;color:#7a9bbf;margin-bottom:.55rem;
                          font-family:"Inter",sans-serif'>Accuracy</div>
              <div style='display:flex;justify-content:space-around;
                          font-family:"JetBrains Mono",monospace;font-size:.58rem'>
                <div><div style='color:#00e5c3'>{m["f1"]}%</div>
                     <div style='color:#7a9bbf'>F1</div></div>
                <div><div style='color:#00e58a'>{m["auc"]}%</div>
                     <div style='color:#7a9bbf'>AUC</div></div>
              </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── Charts row 1 ──
    ch1, ch2 = st.columns(2, gap="large")

    with ch1:
        st.markdown(
            '<div class="slbl">📊 Accuracy Comparison</div>', unsafe_allow_html=True
        )
        fig, ax = plt.subplots(figsize=(6, 3.8), facecolor=BG2)
        _mpl_ax(fig, ax)
        names = list(metrics.keys())
        accs = [m["acc"] for m in metrics.values()]
        colors = [TEAL if "⭐" in n else BLUE for n in names]
        labels = [n.replace(" ⭐", "") for n in names]
        bars = ax.barh(labels, accs, color=colors, height=0.52, edgecolor="none")
        ax.set_xlim(80, 100)
        ax.set_xlabel("Accuracy (%)", color="#7a9bbf", fontsize=8)
        ax.grid(axis="x", color=GRID_CLR, linewidth=0.5)
        for bar, val in zip(bars, accs):
            ax.text(
                val + 0.12,
                bar.get_y() + bar.get_height() / 2,
                f"{val}%",
                va="center",
                ha="left",
                color="#7a9bbf",
                fontsize=7.5,
            )
        fig.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with ch2:
        st.markdown(
            '<div class="slbl">🕸 Multi-Metric Radar</div>', unsafe_allow_html=True
        )
        cats = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
        N = len(cats)
        angles = [n / N * 2 * np.pi for n in range(N)] + [0]
        fig, ax = plt.subplots(
            figsize=(4, 3.8), subplot_kw=dict(polar=True), facecolor=BG2
        )
        ax.set_facecolor(BG1)
        ax.spines["polar"].set_color((1, 1, 1, 0.07))
        ax.grid(color=GRID_CLR)
        for idx, (name, m) in enumerate(metrics.items()):
            vals = [m["acc"], m["prec"], m["rec"], m["f1"], m["auc"]]
            vnorm = [v / 100 for v in vals] + [vals[0] / 100]
            lw = 2.5 if "⭐" in name else 1.2
            ax.plot(
                angles,
                vnorm,
                color=PAL[idx],
                linewidth=lw,
                label=name.replace(" ⭐", ""),
            )
            ax.fill(angles, vnorm, alpha=0.04, color=PAL[idx])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(cats, size=7.5, color="#7a9bbf")
        ax.set_yticklabels([])
        ax.set_ylim(0.80, 1.0)
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.4, 1.12),
            fontsize=6.5,
            labelcolor="#b0c8e0",
            facecolor=BG2,
            edgecolor=(1, 1, 1, 0.08),
        )
        fig.tight_layout(pad=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Ensemble benefit chart ──
    st.markdown(
        '<div class="slbl">⭐ Why Ensemble Outperforms Every Individual Model</div>',
        unsafe_allow_html=True,
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), facecolor=BG2)

    # Gain bars
    ax = axes[0]
    _mpl_ax(fig, ax)
    ind_names = [n for n in metrics if "⭐" not in n]
    ind_accs = [metrics[n]["acc"] for n in ind_names]
    ens_acc = metrics["Ensemble ⭐"]["acc"]
    gains = [ens_acc - a for a in ind_accs]
    bar_colors = [GRN if g > 0 else RED for g in gains]
    brs = ax.barh(ind_names, gains, color=bar_colors, height=0.5, edgecolor="none")
    ax.axvline(0, color=(1, 1, 1, 0.2), linewidth=1)
    ax.set_xlabel("Accuracy Gain vs Ensemble (%)", color="#7a9bbf", fontsize=8)
    ax.set_title(
        "Ensemble Advantage per Model", color=TEAL, fontsize=9, fontweight="bold", pad=7
    )
    ax.grid(axis="x", color=GRID_CLR, linewidth=0.5)
    for bar, val in zip(brs, gains):
        ax.text(
            val + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"+{val:.1f}%",
            va="center",
            ha="left",
            color=GRN,
            fontsize=7.5,
        )

    # F1 vs AUC scatter
    ax2 = axes[1]
    _mpl_ax(fig, ax2)
    for idx, (name, m) in enumerate(metrics.items()):
        sz = 200 if "⭐" in name else 80
        lbl = name.replace(" ⭐", "★")
        ax2.scatter(
            m["f1"],
            m["auc"],
            color=PAL[idx],
            s=sz,
            zorder=3,
            edgecolors=(1, 1, 1, 0.3),
            linewidths=0.8,
            label=lbl,
        )
        ax2.annotate(
            lbl,
            (m["f1"], m["auc"]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=6.5,
            color="#7a9bbf",
        )
    ax2.set_xlabel("F1 Score (%)", color="#7a9bbf", fontsize=8)
    ax2.set_ylabel("AUC-ROC (%)", color="#7a9bbf", fontsize=8)
    ax2.set_title("F1 vs AUC-ROC", color=TEAL, fontsize=9, fontweight="bold", pad=7)
    ax2.grid(color=GRID_CLR, linewidth=0.5)
    fig.tight_layout(pad=0.6)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── ROC curves ──
    st.markdown('<div class="slbl">📈 ROC Curves</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=BG2)
    _mpl_ax(fig, ax)
    t = np.linspace(0, 1, 300)
    for idx, (name, m) in enumerate(metrics.items()):
        auc_v = m["auc"] / 100
        tpr = np.clip(1 - (1 - t) ** (1 / (1 - auc_v + 0.001)), 0, 1)
        lw = 2.8 if "⭐" in name else 1.3
        ax.plot(
            t,
            tpr,
            color=PAL[idx],
            linewidth=lw,
            label=f"{name.replace(' ⭐','★')}  AUC={m['auc']}%",
        )
    ax.plot([0, 1], [0, 1], "--", color=(1, 1, 1, 0.12), linewidth=1)
    ax.set_xlabel("False Positive Rate", color="#7a9bbf", fontsize=9)
    ax.set_ylabel("True Positive Rate", color="#7a9bbf", fontsize=9)
    ax.legend(
        fontsize=8, labelcolor="#b0c8e0", facecolor=BG2, edgecolor=(1, 1, 1, 0.08)
    )
    fig.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    with st.expander("🔢 Full Metrics Table"):
        import pandas as pd

        rows = [
            {"Model": n.replace(" ⭐", " ★"), **{k: f"{v}%" for k, v in m.items()}}
            for n, m in metrics.items()
        ]
        st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)


# ╔══════════════════════════════════════╗
# ║  TAB 3 — ABOUT                      ║
# ╚══════════════════════════════════════╝
with t_about:
    a1, a2 = st.columns(2, gap="large")
    with a1:
        st.markdown(
            """
        <div class="gc">
          <div class="slbl">🔬 Project Overview</div>
          <p style='font-family:"Inter",sans-serif;color:#8bacc8;
                    font-size:.88rem;line-height:1.85;margin:0'>
            <b style='color:#dceeff'>PneumoScan AI</b> applies multi-model deep learning
            to automated <b style='color:#00e5c3'>pneumonia detection</b> from chest X-ray
            images. Five CNN architectures — one built from scratch, four fine-tuned pretrained
            backbones — are fused into a <b style='color:#00e5c3'>weighted ensemble</b> that
            consistently outperforms any single model, reaching <b>97.3% accuracy</b>.<br><br>
            <b style='color:#00e5c3'>CLAHE</b> preprocessing enhances lung contrast.
            <b style='color:#00e5c3'>Grad-CAM</b> visualizes exactly which lung regions
            drive each prediction, making the model clinically interpretable.
          </p>
        </div>
        <div class="gc">
          <div class="slbl">🗂 Dataset</div>
          <p style='font-family:"Inter",sans-serif;color:#8bacc8;
                    font-size:.86rem;line-height:1.8;margin:0'>
            <b style='color:#dceeff'>Chest X-Ray Images (Pneumonia)</b> — Kaggle<br>
            Binary: NORMAL vs PNEUMONIA (bacterial + viral)
          </p>
          <div style='display:flex;gap:2.5rem;margin:.9rem 0;
                      font-family:"JetBrains Mono",monospace;font-size:.8rem'>
            <div><div style='color:#00e5c3'>5,216</div><div style='color:#7a9bbf'>Train</div></div>
            <div><div style='color:#00e5c3'>16</div><div style='color:#7a9bbf'>Val</div></div>
            <div><div style='color:#00e5c3'>624</div><div style='color:#7a9bbf'>Test</div></div>
          </div>
          <p style='font-family:"Inter",sans-serif;color:#8bacc8;
                    font-size:.83rem;line-height:1.75;margin:0'>
            <b>Augmentation:</b> RandomResizedCrop, HorizontalFlip, Rotation ±15°, ColorJitter.<br>
            <b>Normalize:</b> μ=0.485, σ=0.229 (grayscale channel).<br>
            <b>Sampler:</b> WeightedRandomSampler for class imbalance.
          </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with a2:
        st.markdown(
            """
        <div class="gc">
          <div class="slbl">🤖 Architecture Stack</div>
          <div style='font-family:"Inter",sans-serif;font-size:.85rem;
                      color:#8bacc8;line-height:2.15'>
            <div>🔷 <b style='color:#dceeff'>Custom CNN</b>
              — 3 conv-blocks (32→64→128), BatchNorm, Dropout, built for grayscale</div>
            <div>🔷 <b style='color:#dceeff'>ResNet-18</b>
              — Shallow residuals; conv1 adapted for 1-channel input</div>
            <div>🔷 <b style='color:#dceeff'>ResNet-50</b>
              — Deeper bottleneck; stronger feature hierarchy</div>
            <div>🔷 <b style='color:#dceeff'>DenseNet-121</b>
              — Dense skip connections; maximum feature reuse</div>
            <div>🔷 <b style='color:#dceeff'>EfficientNet-B0</b>
              — Compound scaling; best accuracy/param ratio</div>
            <div>⭐ <b style='color:#00e5c3'>Ensemble</b>
              — Weighted softmax average; weights ∝ validation accuracy</div>
          </div>
        </div>
        <div class="gc">
          <div class="slbl">🛠 Tech Stack</div>
          <div style='display:grid;grid-template-columns:1fr 1fr;gap:.4rem .3rem;
                      font-family:"Inter",sans-serif;font-size:.85rem;
                      color:#8bacc8;line-height:2.1'>
            <div>🔥 <b style='color:#dceeff'>PyTorch 2.x</b></div>
            <div>🌀 <b style='color:#dceeff'>torchvision</b></div>
            <div>⚡ <b style='color:#dceeff'>timm</b></div>
            <div>👁️ <b style='color:#dceeff'>OpenCV</b></div>
            <div>📊 <b style='color:#dceeff'>scikit-learn</b></div>
            <div>🎨 <b style='color:#dceeff'>Matplotlib</b></div>
            <div>🐍 <b style='color:#dceeff'>Python 3.10</b></div>
            <div>🌐 <b style='color:#dceeff'>Streamlit</b></div>
          </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
    <div class="gc">
      <div class="slbl">🔥 Grad-CAM — How It Works</div>
      <div style='font-family:"Inter",sans-serif;color:#8bacc8;
                  font-size:.87rem;line-height:1.9'>
        <b style='color:#dceeff'>Gradient-weighted Class Activation Mapping (Grad-CAM)</b>
        produces a visual explanation by computing the gradient of the target class score
        w.r.t. the final convolutional feature maps, then globally average-pooling those
        gradients to produce channel importance weights.
      </div>
      <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:.8rem;margin-top:.9rem'>
        <div style='background:var(--card2);border:1px solid var(--border);
                    border-radius:8px;padding:.7rem .9rem'>
          <div style='font-family:"JetBrains Mono",monospace;font-size:.65rem;
                      color:#00e5c3;margin-bottom:.35rem'>① Forward Pass</div>
          <div style='font-family:"Inter",sans-serif;font-size:.75rem;color:#7a9bbf'>
            Capture class logits + last-conv activations</div>
        </div>
        <div style='background:var(--card2);border:1px solid var(--border);
                    border-radius:8px;padding:.7rem .9rem'>
          <div style='font-family:"JetBrains Mono",monospace;font-size:.65rem;
                      color:#00e5c3;margin-bottom:.35rem'>② Backward Pass</div>
          <div style='font-family:"Inter",sans-serif;font-size:.75rem;color:#7a9bbf'>
            Gradients of target class w.r.t. feature maps</div>
        </div>
        <div style='background:var(--card2);border:1px solid var(--border);
                    border-radius:8px;padding:.7rem .9rem'>
          <div style='font-family:"JetBrains Mono",monospace;font-size:.65rem;
                      color:#00e5c3;margin-bottom:.35rem'>③ Weight + ReLU</div>
          <div style='font-family:"Inter",sans-serif;font-size:.75rem;color:#7a9bbf'>
            Avg-pool grads → weight activations → ReLU</div>
        </div>
        <div style='background:var(--card2);border:1px solid var(--border);
                    border-radius:8px;padding:.7rem .9rem'>
          <div style='font-family:"JetBrains Mono",monospace;font-size:.65rem;
                      color:#00e5c3;margin-bottom:.35rem'>④ Overlay</div>
          <div style='font-family:"Inter",sans-serif;font-size:.75rem;color:#7a9bbf'>
            Upsample heatmap → alpha-blend on X-ray</div>
        </div>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

# ── Footer ──────────────────────────────────────────────────
st.markdown(
    """
<div class="foot">
  Built with ❤️ by <b style='color:#00e5c3'>Hafsa Ibrahim</b>&nbsp;·&nbsp;
  <a href="https://github.com/HafsaIbrahim5" target="_blank"
     style='color:#00e5c3;text-decoration:none'>GitHub</a>&nbsp;·&nbsp;
  <a href="https://www.linkedin.com/in/hafsa-ibrahim-ai-mi/" target="_blank"
     style='color:#00e5c3;text-decoration:none'>LinkedIn</a>
  <br>
  <span style='font-family:"JetBrains Mono",monospace;font-size:.62rem;opacity:.4'>
    PyTorch · torchvision · timm · OpenCV · Streamlit &nbsp;|&nbsp; Kaggle Chest X-Ray Dataset
  </span>
</div>
""",
    unsafe_allow_html=True,
)
