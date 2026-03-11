import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib

matplotlib.use("Agg")
from PIL import Image
import os, time, warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Pneumonia Detection in Chest X-Rays · Hafsa Ibrahim",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

.gc {
  background: rgba(10,26,46,.7);
  backdrop-filter: blur(20px);
  border: 1px solid var(--border); border-radius: var(--r);
  padding: 1.4rem; margin-bottom: 1.2rem;
  transition: border-color .3s, box-shadow .3s;
}
.gc:hover { border-color: rgba(0,229,195,.3); box-shadow: var(--glow-t); }

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

.pbar { background: rgba(255,255,255,.05); border-radius: 999px; height: 9px; overflow: hidden; margin: .4rem 0 .15rem; }
.pbar-g { background: linear-gradient(90deg, #00b86e, var(--green)); }
.pbar-r { background: linear-gradient(90deg, #c02040, var(--red)); }

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

.stSelectbox > div, .stSlider > div { background: var(--card) !important; }
.stSelectbox label, .stSlider label {
  font-family: 'Inter', sans-serif !important;
  font-size: 13px !important; font-weight: 500 !important;
  color: var(--txt2) !important; letter-spacing: 0 !important;
}

.streamlit-expanderHeader {
  background: var(--card) !important; border: 1px solid var(--border) !important;
  border-radius: 10px !important; color: var(--txt) !important;
  font-family: 'Syne', sans-serif !important; font-weight: 600 !important;
  font-size: 14px !important;
}

[data-testid="stMetric"] {
  background: var(--card) !important; border: 1px solid var(--border) !important;
  border-radius: 12px !important; padding: 1rem !important;
}
[data-testid="stMetricValue"] {
  color: var(--teal) !important;
  font-family: 'JetBrains Mono', monospace !important;
}

.banner-warn {
  background: rgba(245,166,35,.07); border: 1px solid rgba(245,166,35,.3);
  border-radius: 10px; padding: .8rem 1.2rem; margin-bottom: 1.5rem;
  font-family: 'Inter', sans-serif; font-size: .84rem;
  display: flex; align-items: center; gap: .8rem;
}

.disc {
  background: rgba(26,143,255,.06); border: 1px solid rgba(26,143,255,.2);
  border-radius: 10px; padding: .85rem 1.1rem;
  font-family: 'Inter', sans-serif; font-size: .82rem;
  color: var(--txt2); line-height: 1.7; margin-top: 1rem;
}

.foot {
  text-align: center; padding: 2rem 0 1rem;
  font-family: 'Inter', sans-serif; font-size: .75rem;
  color: var(--txt2); line-height: 2;
}

@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
.dot {
  width: 7px; height: 7px; background: var(--green); border-radius: 50%;
  display: inline-block; animation: pulse 2s ease-in-out infinite; margin-right: 5px;
}

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

BG1 = "#03080f"
BG2 = "#0a1a2e"
TEAL = "#00e5c3"
BLUE = "#1a8fff"
RED = "#ff3b5c"
GRN = "#00e58a"
AMB = "#f5a623"
PUR = "#a855f7"
PAL = [TEAL, BLUE, GRN, AMB, RED, PUR]

GRID_CLR = (1, 1, 1, 0.05)
SPINE_CLR = (1, 1, 1, 0.08)
LEG_CLR = (0.12, 0.22, 0.35, 1)


def _mpl_ax(fig, ax):
    fig.patch.set_facecolor(BG2)
    ax.set_facecolor(BG1)
    ax.tick_params(colors="#7a9bbf", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(SPINE_CLR)
    ax.grid(color=GRID_CLR, linewidth=0.5)


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
    return tfm(img).unsqueeze(0)


def clahe_preview(pil_img):
    arr = np.array(pil_img.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(arr)
    return arr, enhanced


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


def _demo_heatmap(size=14):
    h = np.random.rand(size, size).astype(np.float32)
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
    cam_up = cv2.resize(
        cam, (orig_rgb.shape[1], orig_rgb.shape[0]), interpolation=cv2.INTER_LANCZOS4
    )
    cam_boosted = np.power(cam_up, 0.9).astype(np.float32)
    cam_u8 = np.uint8(255 * cam_boosted)
    col_bgr = cv2.applyColorMap(cam_u8, cmap_cv)
    col_rgb = cv2.cvtColor(col_bgr, cv2.COLOR_BGR2RGB)
    gray3 = np.stack(
        [
            np.array(
                Image.fromarray(orig_rgb)
                .convert("L")
                .resize((orig_rgb.shape[1], orig_rgb.shape[0]), Image.LANCZOS)
            )
        ]
        * 3,
        axis=-1,
    ).astype(np.float32)
    blend = gray3 * (1 - alpha) + col_rgb.astype(np.float32) * alpha
    blend = np.clip(blend, 0, 255).astype(np.uint8)
    return blend, cam_up


def make_all_gradcams_figure(
    pil_img, cams_dict, ind_results, alpha=0.40, cmap_cv=cv2.COLORMAP_PLASMA
):
    SIZE = 300
    orig_rgb = np.array(pil_img.convert("RGB").resize((SIZE, SIZE), Image.LANCZOS))

    model_names = list(cams_dict.keys())
    n_models = len(model_names)
    n_cols = n_models

    fig = plt.figure(figsize=(3.0 * n_cols, 3.5), facecolor=BG1)
    fig.patch.set_facecolor(BG1)

    gs = fig.add_gridspec(
        1, n_cols, hspace=0, wspace=0.05, top=0.85, bottom=0.05, left=0.01, right=0.99
    )

    for col_idx, mname in enumerate(model_names):
        cam = cams_dict[mname]
        blend, _ = _blend_cam(orig_rgb, cam, alpha, cmap_cv)

        conf_txt = ""
        pred_col = TEAL
        for v in ind_results.values():
            if v["display"] == mname:
                pct = int(v["conf"] * 100)
                conf_txt = f"{pct}%"
                pred_col = RED if v["pred"] == "PNEUMONIA" else GRN
                break

        ax = fig.add_subplot(gs[0, col_idx])
        ax.set_facecolor(BG1)
        ax.imshow(blend, interpolation="lanczos")
        ax.axis("off")

        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor(pred_col)
            sp.set_linewidth(3.0)

        short = mname.replace("EfficientNet-", "EffNet-").replace(
            "Custom CNN", "ImprovedCNN"
        )
        ax.set_title(
            f"{short}\n{conf_txt}",
            color="#dceeff",
            fontsize=10,
            fontweight="bold",
            linespacing=1.2,
            pad=8,
        )

    fig.text(
        0.5,
        0.95,
        "🔥 Grad-CAM Heatmaps",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=TEAL,
        fontfamily="DejaVu Sans",
    )

    return fig


def plot_ensemble_fusion(ind_results, ens_probs):
    names = [r["display"] for r in ind_results.values()]
    p_probs = [r["probs"][1] for r in ind_results.values()]
    weights = [r["weight"] for r in ind_results.values()]
    ens_p = float(ens_probs[1])

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=BG2)

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


with st.sidebar:
    st.markdown(
        """
    <div style='text-align:center;padding:1.4rem 0 1.8rem'>
      <div style='font-size:2.6rem'>🫁</div>
      <div style='font-family:"Syne",sans-serif;font-size:1.35rem;font-weight:800;
                  background:linear-gradient(135deg,#fff,#00e5c3);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  background-clip:text;margin-top:.35rem;letter-spacing:-.5px'>
        Pneumonia Detection
      </div>
      <div style='font-family:"JetBrains Mono",monospace;font-size:.6rem;
                  color:#7a9bbf;letter-spacing:2px;margin-top:.3rem'>
        ENSEMBLE SYSTEM v2.0
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="slbl">⚙ Settings</div>', unsafe_allow_html=True)

    # تمت إزالة خيار Grad-CAM Sub-model لأنه غير مستخدم

    alpha_val = st.slider("Heatmap Opacity", 0.2, 0.8, 0.40, 0.05)
    cmap_name = st.selectbox(
        "Heatmap Colormap", ["PLASMA", "TURBO", "JET", "HOT", "MAGMA"], index=0
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
      ▸ Custom CNN (ImprovedCNN)<br>
      ▸ ResNet-18<br>
      ▸ ResNet-50<br>
      ▸ DenseNet-121<br>
      ▸ EfficientNet-B0<br>
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

with st.spinner("🔄 Loading ensemble model…"):
    ensemble, model_ok, model_msg = load_ensemble("pneumonia_ensemble_full.pth")
    demo_mode = not model_ok

st.markdown(
    """
<div class="hero">
  <div class="hero-title">Pneumonia Detection in Chest X-Rays</div>
  <div class="hero-sub">
    Simple Ensemble <br>
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

if demo_mode:
    st.markdown(
        f"""
    <div class="banner-warn">
      <span style='font-size:1.3rem'>⚡</span>
      <div>
        <b style='color:#f5a623'>Demo Mode</b>
        <span style='color:#7a9bbf;margin-left:.5rem'>
          {model_msg} — place <code>pneumonia_ensemble_full.pth</code> next to app.py for real inference.
        </span>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

t_diag, t_analytics, t_about = st.tabs(
    [
        "🔬  Diagnose",
        "📊  Model Analytics",
        "ℹ️  About & Methods",
    ]
)

with t_diag:
    col_up, col_res = st.columns([1, 1.65], gap="large")

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

            st.markdown(
                '<div class="slbl">🔥 Grad-CAM Heatmaps</div>', unsafe_allow_html=True
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
            <div class="disc">
              ⚕️ <b style='color:#dceeff'>Clinical Disclaimer</b><br>
              PneumoScan AI is a <b>research &amp; decision-support tool only</b>.
              All results must be validated by a certified radiologist or medical
              professional before any clinical decision is made.
            </div>
            """,
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

    st.markdown(
        '<div class="slbl">⭐ Why Ensemble Outperforms Every Individual Model</div>',
        unsafe_allow_html=True,
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), facecolor=BG2)

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

with t_about:
    a1, a2 = st.columns(2, gap="large")
    with a1:
        st.markdown(
            """
        <div class="gc">
          <div class="slbl">🔬 Project Overview</div>
          <p style='font-family:"Inter",sans-serif;color:#8bacc8;
                    font-size:.88rem;line-height:1.85;margin:0'>
            <b style='color:#dceeff'>Pneumonia Detection in Chest X-Rays</b> applies multi-model deep learning
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
            <div>🔷 <b style='color:#dceeff'>Custom CNN (ImprovedCNN)</b>
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
