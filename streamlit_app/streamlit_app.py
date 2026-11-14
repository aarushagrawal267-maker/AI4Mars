"""Mars Terrain Intelligence — Streamlit edition (for Streamlit Community Cloud).

Runs the trained U-Net and DeepLabV3 checkpoints committed in deploy/bundle/ with the project's own Python code
(ml/ for inference and terrain statistics, backend/app/navigation for A*). Results, experiments and the AI4Mars
test split come from the same bundle. Training and evaluation need the full app (FastAPI + React) from the repo.

Local run:  streamlit run streamlit_app/streamlit_app.py
"""
from __future__ import annotations

import io
import json
import sqlite3
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.navigation.planner import plan as plan_route  # noqa: E402
from ml.architectures.factory import load_checkpoint  # noqa: E402
from ml.classes import CLASS_KEYS, CLASS_LABELS, DEFAULT_COLORS  # noqa: E402
from ml.datasets.cache import read_gray, read_label  # noqa: E402
from ml.inference.predictor import predict  # noqa: E402
from ml.inference.terrain import class_grid, terrain_statistics  # noqa: E402
from ml.inference.visualize import colorize, heatmap, overlay  # noqa: E402
from ml.preprocessing.transforms import ImageValidationError, decode_upload  # noqa: E402

BUNDLE = ROOT / "deploy" / "bundle"
RAW = BUNDLE / "data" / "raw"
LABEL = dict(zip(CLASS_KEYS, CLASS_LABELS))
ARCH = {"unet": "U-Net", "deeplabv3": "DeepLabV3"}
MAX_UPLOAD_MB = 20
MAX_OUTPUT_SIDE = 1024

st.set_page_config(page_title="Mars Terrain Intelligence", page_icon="🪐", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; max-width: 1400px;}
  [data-testid="stMetricValue"] {font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 1.55rem;}
  .mti-eyebrow {font-family: ui-monospace, monospace; font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; color: #E8743B;}
  .mti-swatch {display:inline-block; width:.8rem; height:.8rem; border-radius:3px; margin-right:.35rem; vertical-align:-1px;}
  .mti-note {font-size: .82rem; opacity: .75;}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- data

@st.cache_data
def load_results() -> tuple[list[dict], list[dict]]:
    """Experiments and evaluations from the bundled results database (read-only)."""
    db = sqlite3.connect(f"file:{BUNDLE / 'data' / 'mti.sqlite3'}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    exps = [dict(r) for r in db.execute("SELECT * FROM experiments WHERE is_fixture = 0 ORDER BY created_at")]
    evals = [dict(r) for r in db.execute("SELECT * FROM evaluations ORDER BY created_at")]
    db.close()
    for e in exps:
        e["config"] = json.loads(e["config"])
        e["history"] = json.loads(e["history"])
        log = BUNDLE / "experiments" / e["id"] / "train.log"
        e["log"] = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
        e["checkpoint"] = BUNDLE / "checkpoints" / e["id"] / "best.pt"
    names = {e["id"]: e for e in exps}
    for v in evals:
        v["metrics"] = json.loads(v["metrics"])
        v["experiment"] = names.get(v["experiment_id"])
    return exps, [v for v in evals if v["experiment"]]


@st.cache_data
def load_index() -> dict:
    return json.loads((BUNDLE / "data" / "processed" / "index.json").read_text(encoding="utf-8"))


@st.cache_resource(max_entries=1)  # one model in memory at a time keeps the app within ~1 GB
def get_model(checkpoint: str):
    model, meta = load_checkpoint(Path(checkpoint), "cpu")
    return model, meta


def models() -> list[dict]:
    exps, _ = load_results()
    usable = [e for e in exps if e["status"] == "completed" and e["checkpoint"].is_file()]
    return sorted(usable, key=lambda e: e["best_val_miou"] or 0, reverse=True)


def pct(v, d=1):
    return "—" if v is None else f"{v * 100:.{d}f}%"


def legend_html() -> str:
    return " &nbsp; ".join(f'<span class="mti-swatch" style="background:{DEFAULT_COLORS[k]}"></span>{LABEL[k]}' for k in CLASS_KEYS)


def class_bar(percent: dict, title: str = "") -> alt.Chart:
    df = pd.DataFrame({"class": [LABEL[k] for k in CLASS_KEYS], "percent": [percent[k] or 0 for k in CLASS_KEYS]})
    scale = alt.Scale(domain=[LABEL[k] for k in CLASS_KEYS], range=[DEFAULT_COLORS[k] for k in CLASS_KEYS])
    bars = alt.Chart(df, title=title).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("percent:Q", title="% of pixels"), y=alt.Y("class:N", sort=None, title=None),
        color=alt.Color("class:N", scale=scale, legend=None), tooltip=["class", alt.Tooltip("percent:Q", format=".2f")])
    text = bars.mark_text(align="left", dx=4).encode(text=alt.Text("percent:Q", format=".1f"), color=alt.value("#9aa0a6"))
    return (bars + text).properties(height=170)


# ------------------------------------------------------------------------- analysis

def run_analysis(rgb: np.ndarray, exp: dict, name: str) -> dict:
    model, meta = get_model(str(exp["checkpoint"]))
    res = predict(model, int(meta["resolution"]), rgb, "cpu", MAX_OUTPUT_SIDE)
    oh, ow = res["output_size"]
    original = np.asarray(Image.fromarray(rgb).resize((ow, oh), Image.BILINEAR)) if rgb.shape[:2] != (oh, ow) else rgb
    stats = terrain_statistics(res["class_map"], res["confidence"], res["uncertainty"])
    return {"name": name, "model": exp, "res": res, "original": original, "stats": stats, "params": meta["parameters"]}


def show_analysis(a: dict) -> None:
    res, s, cmap = a["res"], a["stats"], a["res"]["class_map"]
    st.markdown(f'<span class="mti-eyebrow">analysis · {ARCH[a["model"]["arch"]]}</span>', unsafe_allow_html=True)
    st.subheader(a["name"])
    alpha = st.slider("Overlay opacity", 0.1, 0.9, 0.5, 0.05, key="alpha")
    t1, t2, t3, t4, t5 = st.tabs(["Overlay", "Mask", "Original", "Confidence", "Uncertainty"])
    t1.image(overlay(a["original"], cmap, alpha=alpha), width=720)
    t2.image(colorize(cmap), width=720)
    t3.image(a["original"], width=720)
    t4.image(heatmap(res["confidence"]), width=720, caption="Max softmax probability (uncalibrated): dark = low, bright = high")
    t5.image(heatmap(res["uncertainty"]), width=720, caption="Normalised predictive entropy: bright = the model hesitates between classes")
    st.markdown(legend_html(), unsafe_allow_html=True)

    c = st.columns(5)
    c[0].metric("Dominant terrain", LABEL[s["dominant_class"]] if s["dominant_class"] else "—")
    c[1].metric("Mean confidence", pct(s["confidence"]["mean"]))
    c[2].metric("Below 50% confidence", f'{s["confidence"]["low_confidence_percent"]:.1f}%')
    c[3].metric("Forward pass", f'{res["inference_ms"]:.0f} ms')
    c[4].metric("Output", f'{res["output_size"][1]}×{res["output_size"][0]}')
    left, right = st.columns(2)
    with left:
        st.altair_chart(class_bar(s["class_percent"], "Terrain distribution"), width="stretch")
        st.dataframe(pd.DataFrame({
            "Class": [LABEL[k] for k in CLASS_KEYS],
            "Share %": [round(s["class_percent"][k], 2) for k in CLASS_KEYS],
            "Pixels": [s["class_pixels"][k] for k in CLASS_KEYS],
            "Mean confidence": [pct(s["confidence"]["per_class_mean"][k]) for k in CLASS_KEYS],
            "Regions": [s["spatial"][k]["regions"] for k in CLASS_KEYS],
        }), hide_index=True, width="stretch")
    with right:
        h = pd.DataFrame(s["confidence"]["histogram"])
        h["bin"] = (h["from"] * 100).round().astype(int).astype(str) + "–" + (h["to"] * 100).round().astype(int).astype(str)
        st.altair_chart(alt.Chart(h, title="Confidence distribution").mark_bar(color="#62A0DE", cornerRadiusEnd=3).encode(
            x=alt.X("bin:N", sort=None, title="confidence %"), y=alt.Y("percent:Q", title="% of pixels"),
            tooltip=["bin", alt.Tooltip("percent:Q", format=".2f")]).properties(height=200), width="stretch")
        bands = pd.DataFrame(s["band_profile"]).melt(id_vars=["band", "from_top_percent"], value_vars=CLASS_KEYS, var_name="cls", value_name="pct")
        bands["class"] = bands["cls"].map(LABEL)
        scale = alt.Scale(domain=[LABEL[k] for k in CLASS_KEYS], range=[DEFAULT_COLORS[k] for k in CLASS_KEYS])
        st.altair_chart(alt.Chart(bands, title="Near-to-far profile (band 1 = top of frame)").mark_area().encode(
            x=alt.X("band:O", title="horizontal band"), y=alt.Y("pct:Q", stack="normalize", title="share"),
            color=alt.Color("class:N", scale=scale, legend=alt.Legend(orient="bottom", title=None))).properties(height=200), width="stretch")
    st.write(s["summary"])
    st.caption("Every pixel receives a class, including regions the AI4Mars labels never cover (the rover's own hardware, sky, "
               "terrain beyond ~30 m); predictions there are not meaningful. Confidence is not calibrated and is not accuracy.")
    png = io.BytesIO()
    Image.fromarray(colorize(cmap)).save(png, format="PNG")
    st.download_button("Download segmentation mask (PNG)", png.getvalue(), file_name="segmentation_mask.png", mime="image/png")


def page_analysis() -> None:
    st.title("Image Analysis")
    st.write("Segment every pixel of a Mars rover image into soil, bedrock, sand or big rock with a trained model.")
    ms = models()
    choice = st.selectbox("Model", ms, format_func=lambda e: f'{ARCH[e["arch"]]} — {e["name"]} (val mIoU {pct(e["best_val_miou"])})')
    src = st.radio("Image source", ["Expert-labelled test image", "Upload my own image"], horizontal=True)
    if src.startswith("Upload"):
        up = st.file_uploader("JPG, JPEG or PNG (max 20 MB). Grayscale NavCam frames match the training data best.", type=["jpg", "jpeg", "png"])
        if up is not None:
            data = up.getvalue()
            if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
                st.error(f"The file is larger than {MAX_UPLOAD_MB} MB.")
                return
            try:
                rgb = np.asarray(decode_upload(data))
            except ImageValidationError as exc:
                st.error(str(exc))
                return
            if st.button("Run analysis", type="primary"):
                with st.spinner("Running the model…"):
                    st.session_state["analysis"] = run_analysis(rgb, choice, up.name)
    else:
        tests = [r for r in load_index()["records"] if r["split"] == "test"]
        rid = st.selectbox("Test image", [r["id"] for r in tests], index=min(5, len(tests) - 1))
        rec = next(r for r in tests if r["id"] == rid)
        st.image(read_gray(RAW / rec["image"]), width=260, caption="Selected image (held out from training)")
        if st.button("Run analysis", type="primary"):
            gray = read_gray(RAW / rec["image"])
            with st.spinner("Running the model…"):
                st.session_state["analysis"] = run_analysis(np.repeat(gray[:, :, None], 3, axis=2), choice, rid)
    if "analysis" in st.session_state:
        st.divider()
        show_analysis(st.session_state["analysis"])


# --------------------------------------------------------------------- navigation

def page_navigation() -> None:
    st.title("Navigation Simulator")
    st.warning("Experimental simulation. Costs are assumptions, not validated measurements of rover mobility. Terrain class alone "
               "cannot determine rover safety, grid cells are image-space (not metres), and this is not connected to any rover.")
    a = st.session_state.get("analysis")
    if not a:
        st.info("Run an analysis in **Image Analysis** first; the route is planned on its segmentation.")
        return
    cmap, unc = a["res"]["class_map"], a["res"]["uncertainty"]
    left, right = st.columns([2, 1])
    with right:
        size = st.slider("Grid cells on the long side", 16, 96, 48, 8)
        grid = class_grid(cmap, max(2, round(cmap.shape[0] * size / max(cmap.shape))), max(2, round(cmap.shape[1] * size / max(cmap.shape))), unc)
        rows, cols = grid["rows"], grid["cols"]
        st.markdown("**Start / destination** (row, column)")
        c1, c2 = st.columns(2)
        sr = c1.number_input("Start row", 0, rows - 1, rows - 2)
        sc = c2.number_input("Start col", 0, cols - 1, cols // 2)
        gr = c1.number_input("Goal row", 0, rows - 1, max(1, rows // 3))
        gc = c2.number_input("Goal col", 0, cols - 1, int(cols * 0.8))
        st.markdown("**Terrain costs** (assumptions)")
        defaults = {"soil": 1.0, "bedrock": 2.0, "sand": 5.0, "big_rock": 20.0}
        costs = {}
        for k in CLASS_KEYS:
            cc1, cc2 = st.columns([2, 1])
            v = cc1.number_input(LABEL[k], 0.1, 1000.0, defaults[k], 0.5, key=f"cost_{k}")
            p = cc2.toggle("passable", True, key=f"pass_{k}")
            costs[k] = {"cost": float(v), "passable": bool(p)}
        uw = st.slider("Uncertainty penalty", 0.0, 5.0, 0.0, 0.5)
        diag = st.toggle("Diagonal moves", True)
        st.markdown("**Restricted rectangle** (optional)")
        block = st.toggle("Add restricted area", False)
        blocked = []
        if block:
            r0, r1 = st.slider("Rows", 0, rows - 1, (rows // 3, rows // 2))
            c0, c1_ = st.slider("Columns", 0, cols - 1, (cols // 3, cols // 2))
            blocked = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1_ + 1)]
    if (sr, sc) == (gr, gc):
        left.error("Start and destination must be different cells.")
        return
    result = plan_route(grid["cells"], costs, (int(sr), int(sc)), (int(gr), int(gc)), blocked=blocked, allow_diagonal=diag,
                        uncertainty=grid["uncertainty"], uncertainty_weight=uw)
    with left:
        cell = max(6, min(14, 720 // max(rows, cols)))
        cells = np.asarray(grid["cells"], dtype=np.int16)
        ids = np.where(cells < 0, 255, cells).astype(np.uint8)  # -1 (no prediction) -> 255 (unlabelled colour)
        img = Image.fromarray(colorize(ids)).resize((cols * cell, rows * cell), Image.NEAREST)
        draw = ImageDraw.Draw(img)
        imp = {i for i, k in enumerate(CLASS_KEYS) if not costs[k]["passable"]}
        for r in range(rows):
            for c in range(cols):
                if grid["cells"][r][c] in imp or (r, c) in set(blocked):
                    draw.rectangle([c * cell, r * cell, (c + 1) * cell - 1, (r + 1) * cell - 1], fill=(30, 30, 32), outline=(232, 92, 84))
        style = {"terrain_aware": ((232, 116, 59), 4), "shortest": ((242, 240, 236), 2), "alternative": ((98, 160, 222), 2)}
        for route in reversed(result["routes"]):
            if route["found"]:
                pts = [(c * cell + cell / 2, r * cell + cell / 2) for r, c in route["path"]]
                draw.line(pts, fill=style[route["kind"]][0], width=style[route["kind"]][1], joint="curve")
        for (r, c), col, t in (((sr, sc), (70, 190, 140), "S"), ((gr, gc), (242, 240, 236), "D")):
            draw.ellipse([c * cell - 3, r * cell - 3, (c + 1) * cell + 2, (r + 1) * cell + 2], fill=col, outline=(11, 12, 14), width=2)
            draw.text((c * cell + cell / 2 - 3, r * cell + cell / 2 - 6), t, fill=(11, 12, 14))
        st.image(img, width="stretch", caption="Orange: terrain-aware route · white: shortest (ignores terrain) · blue: alternative · dark cells: restricted/impassable")
        if not result["found"]:
            st.error(result["reason"])
        else:
            st.dataframe(pd.DataFrame([{
                "Route": r["label"], "Length (cells)": r["length"], "Cumulative cost": r["cost"], "Cost per cell": r["mean_cost_per_unit"],
                **{f"{LABEL[k]} %": r["composition"][k] for k in CLASS_KEYS}} for r in result["routes"] if r["found"]]), hide_index=True, width="stretch")
        st.caption(result["algorithm"] + ". " + result["units"])


# ------------------------------------------------------------------ results pages

def page_overview() -> None:
    exps, evals = load_results()
    idx = load_index()
    st.markdown('<span class="mti-eyebrow">AI4Mars · semantic segmentation · Streamlit edition</span>', unsafe_allow_html=True)
    st.title("Mars Terrain Intelligence")
    st.write("Pixel-level terrain intelligence for Mars rover imagery: U-Net and DeepLabV3 trained on NASA JPL's AI4Mars dataset "
             "(Curiosity NavCam) classify every pixel as soil, bedrock, sand or big rock. Independent student research project; "
             "not a NASA system and not connected to any rover.")
    tests = [v for v in evals if v["split"] == "test" and v["label_variant"] == "min3"]
    best = max(tests, key=lambda v: v["metrics"]["mean_iou"]) if tests else None
    stats = idx["statistics"]
    c = st.columns(4)
    c[0].metric("Validated image–label pairs", f'{stats["total_images"]:,}')
    c[1].metric("Trained models", len(models()))
    c[2].metric("Best test mIoU (min3)", pct(best["metrics"]["mean_iou"]) if best else "—", help=best["experiment"]["name"] if best else None)
    c[3].metric("Expert test images", stats["splits"]["test"]["min3"]["images"])
    st.subheader("Held-out results (official expert test set)")
    rows = []
    for v in evals:
        m = v["metrics"]
        rows.append({"Model": ARCH[v["experiment"]["arch"]], "Split": v["split"] + (f' · {v["label_variant"]}' if v["split"] == "test" else ""),
                     "Images": v["images"], "Pixel acc.": pct(m["pixel_accuracy"]), "mIoU": pct(m["mean_iou"]), "Mean Dice": pct(m["mean_dice"]),
                     **{f"IoU {LABEL[k]}": pct(m["per_class"][k]["iou"]) for k in CLASS_KEYS}})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("All numbers were computed by the full application from the stored predictions; nothing here is typed by hand. "
               "Unlabelled pixels (value 255) are excluded from every metric.")


def page_evaluation() -> None:
    st.title("Model Evaluation")
    _, evals = load_results()
    variants = sorted({v["label_variant"] for v in evals if v["split"] == "test"})
    variant = st.radio("Test labels (expert agreement)", variants, index=variants.index("min3") if "min3" in variants else 0, horizontal=True,
                       help="minN: a pixel is labelled only if at least N of 3 experts marked it. min3 is the strictest.")
    sel = [v for v in evals if v["split"] == "test" and v["label_variant"] == variant]
    cols = st.columns(len(sel))
    for col, v in zip(cols, sel):
        m = v["metrics"]
        with col:
            st.markdown(f'**{ARCH[v["experiment"]["arch"]]}** — {v["experiment"]["name"]}')
            a, b, c = st.columns(3)
            a.metric("mIoU", pct(m["mean_iou"]))
            b.metric("Pixel acc.", pct(m["pixel_accuracy"]))
            c.metric("Mean Dice", pct(m["mean_dice"]))
            st.caption(f'{v["images"]} images · {m["model"]["parameters"] / 1e6:.2f} M params · {m["model"]["checkpoint_mb"]:.1f} MB · '
                       f'{m["inference"]["forward_ms_median"]:.1f} ms median forward pass ({m["inference"]["device"]})')
    df = pd.DataFrame([{"model": ARCH[v["experiment"]["arch"]], "class": LABEL[k], "IoU": v["metrics"]["per_class"][k]["iou"] or 0}
                       for v in sel for k in CLASS_KEYS])
    st.altair_chart(alt.Chart(df, title="Per-class IoU").mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("class:N", sort=[LABEL[k] for k in CLASS_KEYS], title=None), xOffset="model:N",
        y=alt.Y("IoU:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        color=alt.Color("model:N", scale=alt.Scale(domain=["U-Net", "DeepLabV3"], range=["#E8743B", "#62A0DE"]), legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["model", "class", alt.Tooltip("IoU:Q", format=".1%")]).properties(height=280), width="stretch")
    st.subheader("Confusion matrices (rows = true class, share of row)")
    cols = st.columns(len(sel))
    for col, v in zip(cols, sel):
        cm = np.asarray(v["metrics"]["confusion_matrix"], dtype=float)
        share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        col.markdown(f'**{ARCH[v["experiment"]["arch"]]}**')
        col.dataframe(pd.DataFrame((share * 100).round(1), index=[f"true {LABEL[k]}" for k in CLASS_KEYS],
                                   columns=[f"pred {LABEL[k]}" for k in CLASS_KEYS]), width="stretch")
    st.caption("Undefined values (a class never predicted or never present) count as 0 in the chart; the full app reports them as n/a.")


def page_experiments() -> None:
    st.title("Experiments")
    exps, _ = load_results()
    st.dataframe(pd.DataFrame([{"Name": e["name"], "Architecture": ARCH[e["arch"]], "Status": e["status"],
                                "Epochs": f'{e["epochs_completed"]}/{e["config"]["epochs"]}', "Best val mIoU": pct(e["best_val_miou"]),
                                "Train / val": f'{e["n_train"]} / {e["n_val"]}', "Resolution": e["config"]["resolution"],
                                "Wall time (min)": round((e["elapsed_seconds"] or 0) / 60)} for e in exps]), hide_index=True, width="stretch")
    hist = pd.DataFrame([{"experiment": f'{ARCH[e["arch"]]} ({e["epochs_completed"]} ep)', **h} for e in exps for h in e["history"]])
    if not hist.empty:
        a, b = st.columns(2)
        a.altair_chart(alt.Chart(hist, title="Validation mIoU per epoch").mark_line(point=True).encode(
            x="epoch:O", y=alt.Y("val_mean_iou:Q", axis=alt.Axis(format="%"), title="val mIoU"),
            color=alt.Color("experiment:N", legend=alt.Legend(orient="bottom", title=None))).properties(height=260), width="stretch")
        b.altair_chart(alt.Chart(hist, title="Training loss per epoch").mark_line(point=True).encode(
            x="epoch:O", y=alt.Y("train_loss:Q", title="train loss"),
            color=alt.Color("experiment:N", legend=alt.Legend(orient="bottom", title=None))).properties(height=260), width="stretch")
    pick = st.selectbox("Experiment details", exps, format_func=lambda e: e["name"])
    st.json(pick["config"], expanded=False)
    with st.expander("Training log"):
        st.code(pick["log"][-8000:] or "No log.", language="text")


def page_dataset() -> None:
    st.title("Dataset Explorer")
    idx = load_index()
    s = idx["statistics"]
    st.write("NASA JPL AI4Mars merged v0.6 (Zenodo [10.5281/zenodo.15995036](https://doi.org/10.5281/zenodo.15995036), CC-BY-4.0), "
             "Curiosity NavCam subset. Label encoding: 0 soil · 1 bedrock · 2 sand · 3 big rock · 255 unlabelled.")
    c = st.columns(5)
    c[0].metric("Valid pairs", f'{s["total_images"]:,}')
    c[1].metric("Train", f'{s["splits"]["train"]["images"]:,}')
    c[2].metric("Validation", f'{s["splits"]["val"]["images"]:,}')
    c[3].metric("Test (expert)", s["splits"]["test"]["min3"]["images"])
    c[4].metric("Unlabelled (train)", f'{s["splits"]["train"]["unlabeled_percent"]:.1f}%')
    a, b = st.columns(2)
    a.altair_chart(class_bar(s["splits"]["train"]["class_percent_of_labeled"], "Training labels: share of labelled pixels"), width="stretch")
    b.altair_chart(class_bar(s["splits"]["test"]["min3"]["class_percent_of_labeled"], "Test labels (min3): share of labelled pixels"), width="stretch")
    st.caption(f'Class imbalance: big rock is {s["splits"]["train"]["class_percent_of_labeled"]["big_rock"]:.2f}% of labelled training pixels. '
               f'Leakage check: {idx["leakage_check"]["train_val_shared_positions"]} rover positions shared by train and validation; '
               f'{idx["leakage_check"]["test_images_sharing_position_with_train"]} of {idx["leakage_check"]["test_images"]} test images share a position with training data.')
    st.subheader("Expert-labelled test images")
    tests = [r for r in idx["records"] if r["split"] == "test"]
    only_rock = st.toggle("Only images that contain big rocks", False)
    if only_rock:
        tests = [r for r in tests if r["labels"].get("min1", {}).get("counts", [0, 0, 0, 0])[3] > 0]
    variant = st.radio("Label variant", ["min1", "min2", "min3"], horizontal=True)
    page = st.number_input(f"Page (12 per page, {len(tests)} images)", 1, max(1, (len(tests) + 11) // 12), 1)
    grid = st.columns(4)
    for i, r in enumerate(tests[(page - 1) * 12: page * 12]):
        img = read_gray(RAW / r["image"])
        if variant in r["labels"]:
            lab = read_label(RAW / r["labels"][variant]["path"])
            shown = overlay(img, lab, alpha=0.55)
        else:
            shown = img
        grid[i % 4].image(Image.fromarray(shown).resize((300, 300)), caption=r["id"][:28], width="stretch")
    st.markdown(legend_html() + ' &nbsp; <span class="mti-note">(uncoloured = unlabelled)</span>', unsafe_allow_html=True)


def page_about() -> None:
    st.title("About")
    st.markdown("""
**Mars Terrain Intelligence** is an independent student research and engineering project inspired by NASA JPL's AI4Mars
dataset. It is not an official NASA system, is not endorsed by NASA/JPL, and has never been connected to a rover.

- **Models:** U-Net (from scratch) and DeepLabV3 with a COCO-pretrained MobileNetV3-Large backbone, trained with identical
  settings (192 px, 10 epochs, AdamW, cross-entropy with ignore index 255 + Dice) on 3,570 Curiosity NavCam images.
- **Evaluation:** official 322-image expert test set at native resolution; unlabelled pixels excluded.
- **This Streamlit edition** runs the trained checkpoints on the server with the project's own Python code. Training,
  evaluation and dataset management are in the full application (FastAPI + React) in the GitHub repository.
- **Limitations:** big rock is rare (≈1% of labelled training pixels) and poorly segmented; regions the labels never
  cover (rover body, sky, far terrain) still receive a class; single training seed; CPU-limited training.

**Citation:** R. M. Swan, D. Atha, H. A. Leopold, M. Gildner, S. Oij, C. Chiu, M. Ono. *AI4Mars: A Dataset for Terrain-Aware
Autonomous Driving on Mars.* CVPR Workshops, 2021. Dataset: doi:10.5281/zenodo.15995036.

Source code: [github.com/aarushagrawal267-maker/AI4Mars](https://github.com/aarushagrawal267-maker/AI4Mars)
""")


PAGES = {
    "Overview": page_overview,
    "Image Analysis": page_analysis,
    "Navigation Simulator": page_navigation,
    "Model Evaluation": page_evaluation,
    "Experiments": page_experiments,
    "Dataset Explorer": page_dataset,
    "About": page_about,
}

with st.sidebar:
    st.markdown("### 🪐 Mars Terrain Intelligence")
    choice = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.caption("Independent student research project. Not affiliated with or endorsed by NASA/JPL.")

PAGES[choice]()
