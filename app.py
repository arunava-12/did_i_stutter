from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from disfluency_utils import (
    alignment_html, count_disfluencies, detect_blocks, make_clean_reference,
    normalize_for_wer, wer_breakdown,
)

st.set_page_config(page_title="DisfluencyLab", page_icon="🎙️", layout="wide")

ROOT = Path(__file__).resolve().parent
AUDIO_DIR = ROOT / "data" / "audio"
TRANSCRIPT_DIR = ROOT / "data" / "transcripts"
RESULTS_DIR = ROOT / "results"
TTS_DIR = RESULTS_DIR / "tts"
SR = 16000

st.title("🎙️ DisfluencyLab")
st.caption("Speech → Features → Log-Mel → ASR → CTC → Text → TTS → Speech, on disfluent speech")


# --------------------------------------------------
# Loading
# --------------------------------------------------
@st.cache_data
def load_csv(filename: str, mtime: float) -> pd.DataFrame:
    """mtime is part of the cache key so re-exported CSVs show up immediately."""
    path = RESULTS_DIR / filename
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def csv(filename):
    path = RESULTS_DIR / filename
    return load_csv(filename, path.stat().st_mtime if path.exists() else 0.0)


def to_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """Accept the old wide export (wav2vec2_transcript, whisper_transcript, ...)."""
    if df.empty or "model" in df.columns:
        return df
    rows = []
    for col in df.columns:
        if col.endswith("_transcript"):
            stem = col[: -len("_transcript")]
            sec = f"{stem}_seconds"
            for _, r in df.iterrows():
                rows.append({
                    "recording_id": r["recording_id"], "model": stem,
                    "prediction": r[col],
                    "inference_seconds": r[sec] if sec in df.columns else np.nan,
                })
    return pd.DataFrame(rows)


@st.cache_data
def read_reference(recording_id: str) -> str | None:
    if not TRANSCRIPT_DIR.exists():
        return None
    for ext in (".txt", ".orth"):
        p = TRANSCRIPT_DIR / f"{recording_id}{ext}"
        if p.exists():
            return p.read_text(encoding="utf-8", errors="replace").strip()
    return None


@st.cache_data
def load_audio(path: str):
    import librosa
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y


asr_results = to_long_format(csv("asr_results.csv"))
wer_results = csv("wer_results.csv")
comparison = csv("model_comparison.csv")
disfl_counts = csv("disfluency_counts.csv")
tts_results = csv("tts_results.csv")

if asr_results.empty:
    st.error("results/asr_results.csv is missing or empty.")
    st.info("Run the notebook (Section 10 exports a ZIP) and unzip it into the project root.")
    st.stop()

if "reference_type" not in wer_results.columns and not wer_results.empty:
    wer_results["reference_type"] = "verbatim"   # old exports only had one reference

audio_files = sorted(AUDIO_DIR.glob("*.wav")) if AUDIO_DIR.exists() else []
recording_ids = sorted(asr_results["recording_id"].astype(str).unique())

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
page = st.sidebar.radio(
    "Section",
    ["Overview", "Waveform & Log-Mel", "CTC Viewer", "Model Comparison",
     "Recording Explorer", "TTS Playback", "Raw Results"],
)
st.sidebar.caption("Three-recording pilot: results are indicative, not general evidence.")

# --------------------------------------------------
# Overview
# --------------------------------------------------
if page == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recordings evaluated", len(recording_ids))
    c2.metric("Systems evaluated", asr_results["model"].nunique())
    c3.metric("Mean inference time", f"{pd.to_numeric(asr_results['inference_seconds'], errors='coerce').mean():.2f} s")
    if "rtf" in asr_results.columns:
        c4.metric("Mean real-time factor", f"{pd.to_numeric(asr_results['rtf'], errors='coerce').mean():.2f}")

    missing = []
    if not AUDIO_DIR.exists():
        missing.append("data/audio/")
    if not TRANSCRIPT_DIR.exists():
        missing.append("data/transcripts/")
    if missing:
        st.warning(f"Missing local folders: {', '.join(missing)}. Audio and reference pages will be limited.")

    st.subheader("Disfluency profile of the references")
    if disfl_counts.empty:   # compute on the fly if the notebook didn't export it
        rows = []
        for rid in recording_ids:
            ref = read_reference(rid)
            if ref:
                rows.append({"recording_id": rid, **count_disfluencies(ref)})
        disfl_counts = pd.DataFrame(rows)
    if not disfl_counts.empty:
        st.dataframe(disfl_counts, width="stretch", hide_index=True)

    st.subheader("Aggregated model results")
    if comparison.empty:
        st.warning("model_comparison.csv is empty or missing.")
    else:
        st.dataframe(comparison, width="stretch", hide_index=True)

    st.info("**How to read the results:** *verbatim* WER keeps stutters in the reference, so models that "
            "clean speech up are penalised with deletions. *Clean* WER uses the intended fluent text, so "
            "models that transcribe every repetition are penalised with insertions.")

# --------------------------------------------------
# Waveform & Log-Mel
# --------------------------------------------------
elif page == "Waveform & Log-Mel":
    st.subheader("Waveform, candidate blocks and log-Mel spectrogram")
    if not audio_files:
        st.warning("No WAV files found in data/audio/.")
        st.stop()
    try:
        import librosa
    except ImportError:
        st.error("This page needs librosa (`pip install librosa`).")
        st.stop()

    path = AUDIO_DIR / f"{st.selectbox('Recording', [p.stem for p in audio_files])}.wav"
    left, right = st.columns(2)
    min_gap = left.slider("Minimum silent gap counted as a block (s)", 0.2, 2.0, 0.4, 0.1)
    top_db = right.slider("Silence threshold (dB below peak)", 20, 50, 35, 1)

    y = load_audio(str(path))
    t = np.arange(len(y)) / SR
    blocks = detect_blocks(y, SR, min_gap=min_gap, top_db=top_db)
    st.audio(str(path))
    m1, m2, m3 = st.columns(3)
    m1.metric("Duration", f"{len(y) / SR:.2f} s")
    m2.metric("Candidate blocks", len(blocks))
    m3.metric("Total blocked time", f"{sum(e - s for s, e in blocks):.2f} s")

    step = max(1, len(y) // 20000)                      # downsample for fast plotting
    wave = go.Figure(go.Scatter(x=t[::step], y=y[::step], mode="lines", name="waveform"))
    for s, e in blocks:
        wave.add_vrect(x0=s, x1=e, fillcolor="red", opacity=0.25, line_width=0)
    wave.update_layout(title="Waveform (red = candidate block)", xaxis_title="Time (s)",
                       yaxis_title="Amplitude", height=300, margin=dict(t=40, b=30))
    st.plotly_chart(wave, width="stretch")

    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=400, hop_length=160, n_mels=80)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    spec = px.imshow(log_mel, origin="lower", aspect="auto", color_continuous_scale="magma",
                     x=np.arange(log_mel.shape[1]) * 160 / SR,
                     labels=dict(x="Time (s)", y="Mel band", color="dB"),
                     title="Log-Mel spectrogram (80 bands, 25 ms window, 10 ms hop)")
    for s, e in blocks:
        spec.add_vrect(x0=s, x1=e, fillcolor="white", opacity=0.15, line_width=0)
    spec.update_layout(height=380, margin=dict(t=40, b=30))
    st.plotly_chart(spec, width="stretch")
    st.caption("Blocks are detected by an energy threshold; ordinary pauses are included, so confirm by listening.")

# --------------------------------------------------
# CTC viewer
# --------------------------------------------------
elif page == "CTC Viewer":
    st.subheader("CTC posteriorgram and path collapse (Wav2Vec2)")
    rid = st.selectbox("Recording", recording_ids)
    npz = RESULTS_DIR / f"ctc_{rid}.npz"
    if not npz.exists():
        st.info(f"No `results/ctc_{rid}.npz`. Re-run the notebook (Section 10 exports the posteriors).")
        st.stop()

    data = np.load(npz, allow_pickle=False)
    probs, vocab = data["probs"].astype(np.float32), [str(v) for v in data["vocab"]]
    blank_id, stride = int(data["blank_id"]), float(data["stride"])
    total = probs.shape[0] * stride
    c1, c2 = st.columns(2)
    if total > 1.5:
        length = c1.slider("Window length (s)", 1.0, float(min(10.0, total)), float(min(4.0, total)), 0.5)
    else:
        length = total
    if total - length > 0.1:
        start = c2.slider("Window start (s)", 0.0, float(total - length), 0.0, 0.1)
    else:
        start = 0.0
    a, b = int(start / stride), int((start + length) / stride)

    ids = probs.argmax(-1)
    window = ids[a:b]
    raw = "".join("_" if i == blank_id else vocab[i] for i in window)
    merged = [i for k, i in enumerate(window) if k == 0 or i != window[k - 1]]
    collapsed = "".join(vocab[i] for i in merged if i != blank_id).replace("|", " ")

    fig = px.imshow(probs[a:b].T, origin="lower", aspect="auto", color_continuous_scale="viridis",
                    x=np.arange(a, b) * stride, y=vocab,
                    labels=dict(x="Time (s)", y="Token", color="P"))
    fig.update_yaxes(type="category")
    fig.update_layout(height=520, margin=dict(t=30, b=30))
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Raw frame path** (`_` = blank, `|` = word boundary, 20 ms per symbol)")
    st.code(raw, language=None)
    st.markdown("**After merge-repeats → remove-blanks**")
    st.code(collapsed.strip() or "(empty)", language=None)
    st.metric("Blank frames in whole recording", f"{np.mean(ids == blank_id):.0%}")

# --------------------------------------------------
# Model comparison
# --------------------------------------------------
elif page == "Model Comparison":
    st.subheader("Model comparison")
    if not comparison.empty:
        st.dataframe(comparison, width="stretch", hide_index=True)
        wer_cols = [c for c in ("verbatim_pooled_WER", "clean_pooled_WER") if c in comparison.columns]
        if wer_cols and "model" in comparison.columns:
            long = comparison.melt(id_vars="model", value_vars=wer_cols,
                                   var_name="reference", value_name="WER")
            long["reference"] = long["reference"].str.replace("_pooled_WER", "", regex=False)
            st.plotly_chart(px.bar(long, x="model", y="WER", color="reference", barmode="group",
                                   text_auto=".3f", title="Pooled WER: verbatim vs clean reference"),
                            width="stretch")
        else:
            numeric = comparison.select_dtypes("number").columns.tolist()
            if numeric and "model" in comparison.columns:
                metric = st.selectbox("Metric", numeric)
                st.plotly_chart(px.bar(comparison, x="model", y=metric, color="model", text_auto=".3f"),
                                width="stretch")
    else:
        st.warning("model_comparison.csv is empty or missing.")

    if not wer_results.empty and {"model", "recording_id", "WER"} <= set(wer_results.columns):
        st.subheader("Per-recording WER and error types")
        ref_type = st.radio("Reference", sorted(wer_results["reference_type"].unique()), horizontal=True)
        sub = wer_results[wer_results["reference_type"] == ref_type]
        st.plotly_chart(px.bar(sub, x="recording_id", y="WER", color="model", barmode="group",
                               title=f"WER per recording ({ref_type} reference)"),
                        width="stretch")
        err_cols = [c for c in ("substitutions", "deletions", "insertions") if c in sub.columns]
        if err_cols:
            errs = sub.groupby("model")[err_cols].sum().reset_index().melt("model", var_name="type", value_name="count")
            st.plotly_chart(px.bar(errs, x="model", y="count", color="type",
                                   color_discrete_map={"substitutions": "#f59e0b", "deletions": "#ef4444",
                                                       "insertions": "#22c55e"},
                                   title=f"Error types ({ref_type} reference)"),
                            width="stretch")
            st.caption("Many **deletions** against the verbatim reference = the model is cleaning up disfluencies. "
                       "Many **insertions** against the clean reference = it is transcribing the stutter.")

# --------------------------------------------------
# Recording explorer
# --------------------------------------------------
elif page == "Recording Explorer":
    st.subheader("Recording explorer")
    rid = st.selectbox("Recording", recording_ids)
    wav = AUDIO_DIR / f"{rid}.wav"
    if wav.exists():
        st.audio(str(wav))
    else:
        st.info("Audio file not found locally; showing text results only.")

    raw_ref = read_reference(rid)
    if raw_ref is None:
        st.warning("No matching transcript found in data/transcripts/.")
        st.stop()
    verbatim, clean = normalize_for_wer(raw_ref), make_clean_reference(raw_ref)
    with st.expander("Reference transcripts", expanded=True):
        st.markdown("**Raw**");      st.text(raw_ref)
        st.markdown("**Verbatim (normalised)**"); st.text(verbatim)
        st.markdown("**Clean (fluent intended text)**"); st.text(clean)

    view = st.radio("Align against", ["verbatim", "clean"], horizontal=True)
    ref_for_view = verbatim if view == "verbatim" else clean
    rows = asr_results[asr_results["recording_id"].astype(str) == rid]
    st.caption("Legend: orange = substitution, red = deletion, green = insertion")
    for _, row in rows.iterrows():
        pred = str(row["prediction"]) if pd.notna(row["prediction"]) else ""
        v, c = wer_breakdown(verbatim, pred), wer_breakdown(clean, pred)
        with st.expander(f"{row['model']} — verbatim WER {v['WER']:.2f} · clean WER {c['WER']:.2f}", expanded=True):
            st.markdown("**Prediction:** " + (pred or "_(empty)_"))
            st.markdown(alignment_html(ref_for_view, pred), unsafe_allow_html=True)
            shown = v if view == "verbatim" else c
            st.caption(f"S={shown['S']}  D={shown['D']}  I={shown['I']}  hits={shown['H']}  N={shown['N']}")

# --------------------------------------------------
# TTS
# --------------------------------------------------
elif page == "TTS Playback":
    st.subheader("TTS: fluent speech from the transcript")
    rid = st.selectbox("Recording", recording_ids)
    tts_wav = TTS_DIR / f"{rid}.wav"
    if not tts_wav.exists():
        st.info(f"No `results/tts/{rid}.wav`. Run notebook Section 9 and unzip the results.")
        st.stop()
    left, right = st.columns(2)
    left.markdown("**Original (disfluent)**")
    if (AUDIO_DIR / f"{rid}.wav").exists():
        left.audio(str(AUDIO_DIR / f"{rid}.wav"))
    right.markdown("**TTS output**")
    right.audio(str(tts_wav))
    if not tts_results.empty and (tts_results["recording_id"].astype(str) == rid).any():
        r = tts_results[tts_results["recording_id"].astype(str) == rid].iloc[0]
        st.markdown(f"**TTS input text** (from {r['tts_source_model']}): {r['tts_text']}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Original → TTS duration", f"{r['original_seconds']} s → {r['tts_seconds']} s")
        c2.metric("Blocks: original → TTS", f"{int(r['original_blocks'])} → {int(r['tts_blocks'])}")
        c3.metric("Round-trip WER", f"{r['round_trip_WER']:.3f}")
    st.caption("TTS output is fluent by construction: the pipeline normalises the disfluency away.")

# --------------------------------------------------
# Raw results
# --------------------------------------------------
elif page == "Raw Results":
    for title, df, name in (("ASR results", asr_results, "asr_results.csv"),
                            ("WER results", wer_results, "wer_results.csv"),
                            ("Model comparison", comparison, "model_comparison.csv")):
        if df.empty:
            continue
        st.subheader(title)
        st.dataframe(df, width="stretch", hide_index=True)
        st.download_button(f"Download {name}", df.to_csv(index=False).encode("utf-8"), name, "text/csv",
                           key=name)
