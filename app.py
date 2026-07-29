from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from cyclegan.preview import write_synthetic_domains
from cyclegan.comparison import architecture_profiles
from cyclegan.data import list_images

st.set_page_config(page_title="CycleCanvas", page_icon="🎨", layout="wide")
st.markdown("""<style>.stApp{background:linear-gradient(145deg,#fff7ed,#fdf2f8 50%,#eef2ff)}.hero{padding:2.1rem;border-radius:28px;background:linear-gradient(120deg,#c2410c,#db2777,#6d28d9);color:white;box-shadow:0 18px 45px #9d174d33;animation:rise .55s ease-out}@keyframes rise{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;transition:none!important;scroll-behavior:auto!important}html:focus-within{scroll-behavior:auto!important}}[data-testid=stMetric]{background:white;border:1px solid #fed7aa;padding:12px;border-radius:16px}</style><div class=hero><h1>🎨 CycleCanvas</h1><p>Explore unpaired translation, cycles, and checkpoint provenance.</p></div>""", unsafe_allow_html=True)

with st.sidebar:
    page = st.radio("Workspace", ["Translate", "Training", "Evaluation", "Architecture lab"])
    direction = st.selectbox("Direction", ["AtoB", "BtoA"])
    checkpoint = st.text_input("Checkpoint", str(ROOT / "outputs" / "checkpoint.pt"))
    st.caption("A checkpoint is required for real translation. The palette demo is explicitly synthetic.")

if page == "Translate":
    source = st.file_uploader("Upload a JPG/PNG/WebP image", type=["jpg", "jpeg", "png", "webp"])
    if source and st.button("Translate with checkpoint", type="primary", use_container_width=True):
        if not Path(checkpoint).exists(): st.error("Checkpoint not found. Train or select one first.")
        else:
            suffix = Path(source.name).suffix or ".png"
            with tempfile.TemporaryDirectory() as directory:
                input_path = Path(directory) / f"input{suffix}"; input_path.write_bytes(source.getvalue())
                try:
                    from cyclegan.inference import translate
                    output = translate(checkpoint, input_path, Path(directory) / "translated.jpg", direction)
                    left, right = st.columns(2); left.image(source.getvalue(), caption="Input"); right.image(str(output), caption=f"Checkpoint output · {direction}")
                except Exception as exc: st.error(str(exc))
    with st.expander("No checkpoint yet? View the synthetic domain UI demo"):
        if st.button("Create synthetic palettes"):
            paths = write_synthetic_domains(ROOT / "outputs" / "synthetic_domains")
            a, b = st.columns(2); a.image(str(paths[0]), caption="Synthetic domain A — not CycleGAN output"); b.image(str(paths[1]), caption="Synthetic domain B — not CycleGAN output")
elif page == "Training":
    epochs = st.slider("Epochs", 2, 300, 200); decay = st.slider("Decay starts", 1, epochs - 1, min(100, epochs - 1)); size = st.select_slider("Image size", [64, 128, 256, 512], value=256)
    st.code(f"python cli.py train --epochs {epochs} --decay-start {decay} --image-size {size}", language="bash")
    history = ROOT / "outputs" / "history.json"
    if history.exists():
        values = json.loads(history.read_text(encoding="utf-8")); st.line_chart({"generator": [x["g"] for x in values], "discriminator": [x["d"] for x in values], "cycle": [x["cycle"] for x in values]})
    else: st.info("Add unpaired images to data/trainA and data/trainB, then run the command above.")
elif page == "Evaluation":
    counts = {name: len(list_images(ROOT / "data" / name)) for name in ("trainA", "trainB", "testA", "testB")}
    columns = st.columns(4)
    for column, name in zip(columns, counts): column.metric(name, counts[name])
    st.write("Held-out test images never enter the training loader. Evaluation translates each domain and cycles it back, saving both images plus mean cycle L1.")
    st.code("python cli.py evaluate --checkpoint outputs/checkpoint.pt --test-a data/testA --test-b data/testB", language="bash")
    report_path = ROOT / "outputs" / "evaluation" / "evaluation.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8")); st.metric("Held-out mean cycle L1", report["mean_cycle_l1"]); st.dataframe(report["records"], use_container_width=True)
    else: st.info("Add held-out images to testA/testB and run the evaluation command after training.")
else:
    size = st.select_slider("Educational comparison size", [16, 32, 48, 64], value=32)
    dense_depth = st.slider("Dense depth", 2, 8, 3)
    blocks = st.slider("Residual blocks", 1, 9, 3)
    profiles = [item.to_dict() for item in architecture_profiles(size, dense_depth=dense_depth, residual_blocks=blocks)]
    st.dataframe(profiles, use_container_width=True, hide_index=True)
    st.write("A dense DNN flattens every pixel, loses spatial locality, and grows rapidly with resolution. The CycleGAN ResNet shares convolutional features and adds identity skip paths, giving gradients shorter routes while preserving content.")
    steps = st.slider("Learning steps", 1, 100, 20)
    batch_size = st.slider("Synthetic batch size", 1, 16, 4)
    st.code(f"python cli.py compare-architectures --image-size {size} --dense-depth {dense_depth} --residual-blocks {blocks} --steps {steps} --batch-size {batch_size}", language="bash")
    if st.button("Run controlled learning comparison", type="primary", use_container_width=True):
        try:
            from cyclegan.comparison import run_learning_comparison
            with st.spinner("Training both small generators on the same fixed mini-batch…"):
                st.session_state.architecture_result = run_learning_comparison(
                    steps=steps,
                    image_size=size,
                    batch_size=batch_size,
                    dense_depth=dense_depth,
                    residual_blocks=blocks,
                )
        except Exception as exc:
            st.error(str(exc))
    if result := st.session_state.get("architecture_result"):
        st.dataframe(
            [{"architecture": name, **metrics} for name, metrics in result["results"].items()],
            use_container_width=True,
            hide_index=True,
        )
        st.info(result["warning"])
    st.caption("The optional command trains both small generators on the same synthetic flip task. It is a learning diagnostic—not a realism score and not a replacement for CycleGAN.")
