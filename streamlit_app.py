# streamlit_app.py
# Combined Streamlit app: (1) Audio Voice Sample Analyzer
# (2) Video Frame Extractor, (3) Facial Sentiment Analysis

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display as lbd
import tempfile
import subprocess
import os
import time
from pathlib import Path
import cv2
from tqdm import tqdm
from imageio_ffmpeg import get_ffmpeg_exe

# DeepFace is large; import lazily when needed to keep startup light
DeepFace = None

st.set_page_config(page_title="Media Analysis Studio", layout="wide")

# ------------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------------

FFMPEG_BIN = get_ffmpeg_exe()  # portable ffmpeg binary (no apt-get needed)

def save_uploaded_file(uploaded_file, dirpath):
    """Save an UploadedFile to a temp dir and return its absolute path."""
    suffix = Path(uploaded_file.name).suffix
    dst = Path(dirpath) / f"uploaded{suffix}"
    with open(dst, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(dst)

def file_size_str(path):
    size_kb = os.path.getsize(path) / 1024
    if size_kb > 1024:
        return f"{size_kb/1024:.2f} MB"
    return f"{size_kb:.2f} KB"

def get_video_stats(path):
    """Return basic stats (length, size, resolution, fps) using OpenCV."""
    try:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        duration = (frame_count / fps) if fps else 0
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        return {
            "Length": f"{minutes}m {seconds}s",
            "FPS": f"{fps:.2f}" if fps else "N/A",
            "Frames": frame_count,
            "Resolution": f"{width}x{height}",
            "Size": file_size_str(path),
        }
    except Exception as e:
        return {"Error": f"Error getting video stats: {e}"}

def display_stats_table(stats_dict, key=None):
    if "Error" in stats_dict:
        st.error(stats_dict["Error"])
    else:
        df = pd.DataFrame(list(stats_dict.items()), columns=["Metric", "Value"])
        st.table(df)

def bgr_to_rgb(img):
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ------------------------------------------------------------------------------------
# 1) Audio Voice Sample Analyzer
# ------------------------------------------------------------------------------------

def analyze_audio_from_video(video_path):
    """Extract audio with ffmpeg; analyze with librosa; return stats, summary, figs, and extremes df."""
    with tempfile.TemporaryDirectory() as tmp:
        audio_path = str(Path(tmp) / "audio.wav")

        # Extract mono 22.05kHz WAV to keep processing light
        cmd = [
            FFMPEG_BIN, "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "22050",
            "-ac", "1",
            audio_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Video clip stats via OpenCV
        clip_stats = get_video_stats(video_path)
        clip_stats_str = (
            "Clip Statistics:\n"
            f"- Duration: {clip_stats.get('Length','N/A')}\n"
            f"- File Size: {clip_stats.get('Size','N/A')}\n"
            f"- Resolution: {clip_stats.get('Resolution','N/A')}\n"
            f"- FPS: {clip_stats.get('FPS','N/A')}"
        )

        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        total_dur = len(y) / sr if sr else 0

        # Pitch tracking
        pitch, mag = librosa.piptrack(y=y, sr=sr)
        pitch_track = pitch.max(axis=0) if pitch.size else np.array([])
        valid_idx = np.where(pitch_track > 0)[0]
        pitch_mean = float(np.mean(pitch_track[valid_idx])) if valid_idx.size else 0.0
        pitch_std = float(np.std(pitch_track[valid_idx])) if valid_idx.size else 0.0
        typical_pitch_range = 150 if pitch_mean > 150 else 100  # rough female/male anchor

        # Energy (RMS)
        energy = float(np.mean(librosa.feature.rms(y=y)[0])) if y.size else 0.0
        ideal_energy_range = 0.15

        # Speaking rate (approx via nonsilent segments per minute)
        nonsilent_segments = librosa.effects.split(y, top_db=20)
        speaking_rate = (len(nonsilent_segments) / total_dur * 60) if total_dur > 0 else 0.0
        typical_speaking_rate = 120  # wpm (approx)

        # Pause frequency (per minute)
        pause_count = max(0, len(nonsilent_segments) - 1)
        pause_frequency = (pause_count / total_dur * 60) if total_dur > 0 else 0.0

        # Scores
        pitch_score = max(0, min(1, 1 - abs(pitch_mean - typical_pitch_range) / max(typical_pitch_range, 1))) * 100
        energy_score = max(0, min(1, energy / max(ideal_energy_range, 1e-6))) * 100
        rate_score = max(0, min(1, speaking_rate / max(typical_speaking_rate, 1e-6))) * 100
        effectiveness_index = (pitch_score + energy_score + rate_score) / 3

        # Extremes + timestamps
        amplitude_extremes = (float(np.max(y)) if y.size else 0.0, float(np.min(y)) if y.size else 0.0)
        max_amp_time = (int(np.argmax(y)) / sr) if y.size else 0.0
        min_amp_time = (int(np.argmin(y)) / sr) if y.size else 0.0

        if valid_idx.size:
            times_frames = librosa.frames_to_time(np.arange(len(pitch_track)), sr=sr)
            max_pitch_idx = valid_idx[np.argmax(pitch_track[valid_idx])]
            min_pitch_idx = valid_idx[np.argmin(pitch_track[valid_idx])]
            max_pitch_value = float(pitch_track[max_pitch_idx])
            min_pitch_value = float(pitch_track[min_pitch_idx])
            max_pitch_time = float(times_frames[max_pitch_idx])
            min_pitch_time = float(times_frames[min_pitch_idx])
        else:
            max_pitch_value = min_pitch_value = 0.0
            max_pitch_time = min_pitch_time = 0.0

        extremes_df = pd.DataFrame({
            "Attribute": ["Max Amplitude", "Min Amplitude", "Max Pitch", "Min Pitch"],
            "Value": [
                f"{amplitude_extremes[0]:.2f}",
                f"{amplitude_extremes[1]:.2f}",
                f"{max_pitch_value:.2f} Hz",
                f"{min_pitch_value:.2f} Hz",
            ],
            "Timestamp (s)": [
                f"{max_amp_time:.2f}",
                f"{min_amp_time:.2f}",
                f"{max_pitch_time:.2f}",
                f"{min_pitch_time:.2f}",
            ],
        })

        # Plot 1: Performance vs Benchmarks (relative)
        metrics = ["Pitch (Hz)", "Energy", "Speaking Rate (wpm)"]
        values = [pitch_mean, energy, speaking_rate]
        ideals = [typical_pitch_range, ideal_energy_range, typical_speaking_rate]
        relative = [(v / i if i else 0.0) for v, i in zip(values, ideals)]

        fig1 = plt.figure(figsize=(8, 4))
        colors = ["green" if r >= 0.8 else "yellow" if r >= 0.5 else "red" for r in relative]
        plt.bar(metrics, relative, color=colors)
        plt.axhline(1.0, linestyle="--", label="Ideal Level")
        plt.ylim(0, max(1.5, max(relative) + 0.2))
        plt.ylabel("Relative (Ideal = 1.0)")
        plt.title("Voice Characteristics vs. Ideal Benchmarks")
        plt.legend()
        plt.tight_layout()

        # Plot 2: Waveform + Spectrogram
        fig2 = plt.figure(figsize=(10, 6))
        gs = fig2.add_gridspec(2, 1, hspace=0.3)
        ax0 = fig2.add_subplot(gs[0, 0])
        n_show = len(y) // 10 if len(y) >= 10 else len(y)
        ax0.plot(np.arange(n_show), y[:n_show], label="Waveform")
        ax0.axhline(0.1, color="g", linestyle="--", label="Ideal Loudness Upper (~0.1)")
        ax0.axhline(-0.1, color="g", linestyle="--", label="Ideal Loudness Lower (~-0.1)")
        ax0.set_title("Audio Waveform (first 10%)")
        ax0.set_xlabel("Samples")
        ax0.set_ylabel("Amplitude")
        ax0.legend()

        ax1 = fig2.add_subplot(gs[1, 0])
        S = np.abs(librosa.stft(y))
        D = librosa.amplitude_to_db(S, ref=np.max)
        img = lbd.specshow(D, y_axis="linear", x_axis="time", sr=sr, ax=ax1)
        fig2.colorbar(img, ax=ax1, format="%+2.0f dB")
        ax1.axhline(100, color="orange", linestyle="--", label="Ideal Freq Lower (~100 Hz)")
        ax1.axhline(300, color="orange", linestyle="--", label="Ideal Freq Upper (~300 Hz)")
        ax1.set_title("Spectrogram with Ideal Frequency Zones")
        ax1.legend()
        plt.tight_layout()

        # Summary text
        pitch_status = "High (excited/stressed)" if pitch_mean > typical_pitch_range * 1.5 else \
                       "Normal" if pitch_mean > typical_pitch_range * 0.8 else "Low (calm/monotone)"
        energy_status = "Low (monotone)" if energy < ideal_energy_range * 0.5 else \
                        "Normal" if energy < ideal_energy_range * 1.5 else "High (energetic)"
        rate_status = "Slow (hesitant)" if speaking_rate < typical_speaking_rate * 0.5 else \
                      "Normal" if speaking_rate < typical_speaking_rate * 1.2 else "Fast (rushed)"
        recommendations = (
            f"- Increase pitch variation for engagement if {pitch_status}.\n"
            f"- Boost energy with louder delivery if {energy_status}.\n"
            f"- Adjust pace toward ~{typical_speaking_rate} wpm if {rate_status}.\n"
            f"- Reduce pauses ({pause_frequency:.1f}/min) for smoother flow if frequent."
        )
        summary = (
            f"Voice Analysis Report:\n"
            f"- Average Pitch: {pitch_mean:.2f} Hz (Std: {pitch_std:.2f} Hz) — {pitch_status}\n"
            f"- Energy Level: {energy:.3f} — {energy_status}\n"
            f"- Speaking Rate: {speaking_rate:.1f} wpm — {rate_status}\n"
            f"- Pause Frequency: {pause_frequency:.1f} pauses/min\n"
            f"- Delivery Effectiveness Index: {effectiveness_index:.1f}/100\n\n"
            f"Business Insights:\n"
            f"This delivery may struggle to engage audiences due to {energy_status.lower()} energy "
            f"and {rate_status.lower()} pace.\n\n"
            f"Recommendations:\n{recommendations}"
        )

        return clip_stats_str, summary, fig1, extremes_df, fig2

# ------------------------------------------------------------------------------------
# 2) Video Frame Extractor
# ------------------------------------------------------------------------------------

def extract_frames_ffmpeg(video_path, frame_interval_seconds, out_dir):
    """Use ffmpeg to extract frames at 1/frame_interval fps; return 3 odd-numbered frames and timestamps."""
    pattern = str(Path(out_dir) / "frame_%04d.jpg")
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", video_path,
        "-vf", f"fps=1/{frame_interval_seconds}",
        pattern
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    frame_files = sorted([f for f in os.listdir(out_dir) if f.startswith("frame_") and f.endswith(".jpg")])
    odd = frame_files[::2][:3]  # top 3 odd frames (frame_0001, 0003, 0005,...)
    frame_paths = [str(Path(out_dir) / f) for f in odd]
    timestamps = [((i * 2 + 1) * frame_interval_seconds) for i in range(len(odd))]
    return frame_paths, timestamps

# ------------------------------------------------------------------------------------
# 3) Facial Sentiment Analysis (DeepFace)
# ------------------------------------------------------------------------------------

def lazy_import_deepface():
    global DeepFace
    if DeepFace is None:
        from deepface import DeepFace as _DF
        DeepFace = _DF
    return DeepFace

def describe_file(path):
    """Descriptive stats for image or video."""
    stats = {"File Size": file_size_str(path)}
    ext = Path(path).suffix.lower()
    if ext in [".png", ".jpg", ".jpeg"]:
        img = cv2.imread(path)
        if img is None:
            stats["Resolution"] = "Error reading image"
        else:
            h, w = img.shape[:2]
            stats["Resolution"] = f"{w} x {h}"
    elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
        stats.update(get_video_stats(path))
    else:
        stats["Resolution"] = "Unsupported file type"
    return stats

def detect_faces_in_image(image_path):
    DF = lazy_import_deepface()
    faces = DF.extract_faces(img_path=image_path, enforce_detection=False)
    return faces

def annotate_image(img_bgr, faces, frame_label=""):
    """Draw boxes & labels; return annotated RGB image and a sorted faces list with 'grid_marker' tags."""
    sorted_faces = sorted(faces, key=lambda f: (f['facial_area']['y'], f['facial_area']['x']))
    annotated = img_bgr.copy()
    for i, face in enumerate(sorted_faces):
        fa = face['facial_area']
        x, y, w, h = fa["x"], fa["y"], fa["w"], fa["h"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{frame_label}-{i+1}" if frame_label else str(i+1)
        cv2.putText(annotated, label, (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        face['grid_marker'] = f"Face {i+1}"
    return bgr_to_rgb(annotated), sorted_faces

def analyze_face_emotions(img_bgr, faces):
    DF = lazy_import_deepface()
    recs = []
    sorted_faces = sorted(faces, key=lambda f: (f['facial_area']['y'], f['facial_area']['x']))
    for face in sorted_faces:
        fa = face['facial_area']
        x, y, w, h = fa["x"], fa["y"], fa["w"], fa["h"]
        crop = img_bgr[y:y+h, x:x+w]
        # DeepFace.analyze expects a path or RGB array; pass RGB array
        rgb = bgr_to_rgb(crop)
        try:
            analysis = DF.analyze(img_path=rgb, actions=['emotion'], enforce_detection=False)
            dominant = analysis[0]['dominant_emotion']
        except Exception:
            dominant = "unknown"
        recs.append({"Face": face.get("grid_marker", "Unknown"), "Sentiment": dominant})
    return recs

def process_media_for_sentiment(path, frame_interval=10):
    """Handle both image and video; return list of annotated RGB images and a DataFrame of results."""
    ext = Path(path).suffix.lower()
    annotated_images = []
    records_all = []

    if ext in [".png", ".jpg", ".jpeg"]:
        img = cv2.imread(path)
        faces = detect_faces_in_image(path)
        annotated_rgb, sorted_faces = annotate_image(img, faces, frame_label="IMG")
        annotated_images.append(annotated_rgb)
        recs = analyze_face_emotions(img, sorted_faces)
        records_all.extend([{"Frame": Path(path).name, **r} for r in recs])

    elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        if not fps:
            cap.release()
            raise RuntimeError("Unable to read FPS from video.")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(frame_interval * fps))
        indices = range(0, frame_count, step)

        for idx in tqdm(indices, total=min(len(range(0, frame_count, step)), 50)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame_bgr = cap.read()
            if not ret:
                continue
            # Save the frame temporarily only if DeepFace requires a path; here we pass arrays
            faces = lazy_import_deepface().extract_faces(img_path=frame_bgr, enforce_detection=False)
            annotated_rgb, sorted_faces = annotate_image(frame_bgr, faces, frame_label=f"F{idx}")
            annotated_images.append(annotated_rgb)
            recs = analyze_face_emotions(frame_bgr, sorted_faces)
            records_all.extend([{"Frame": f"Frame {idx}", **r} for r in recs])
            # Keep display light
            if len(annotated_images) >= 12:  # limit preview
                break
        cap.release()
    else:
        raise ValueError("Unsupported file type.")

    df = pd.DataFrame(records_all) if records_all else pd.DataFrame(columns=["Frame", "Face", "Sentiment"])
    return annotated_images, df

# ------------------------------------------------------------------------------------
# UI
# ------------------------------------------------------------------------------------

st.title("🎬📊 Media Analysis Studio")
st.caption("One Streamlit app with: Audio Voice Analysis • Video Frame Extraction • Facial Sentiment Analysis")

tab1, tab2, tab3 = st.tabs(["🔊 Audio Voice Analyzer", "🖼️ Video Frame Extractor", "🙂 Facial Sentiment Analysis"])

# ----------------------- TAB 1: Audio Voice Analyzer ------------------------------
with tab1:
    st.subheader("Upload a video to extract and analyze its audio voice characteristics")
    vid1 = st.file_uploader("Upload Video (MP4/MOV/AVI/MKV)", type=["mp4", "mov", "avi", "mkv"], key="aud_vid")
    analyze_btn = st.button("Analyze Audio", type="primary")

    if analyze_btn:
        if not vid1:
            st.warning("Please upload a video first.")
        else:
            with st.spinner("Extracting audio and analyzing…"):
                try:
                    with tempfile.TemporaryDirectory() as tmp:
                        video_path = save_uploaded_file(vid1, tmp)
                        clip_stats_str, summary, fig1, extremes_df, fig2 = analyze_audio_from_video(video_path)

                    c1, c2 = st.columns([1, 2], gap="large")
                    with c1:
                        st.text_area("Clip Statistics", clip_stats_str, height=140)
                        st.metric("Delivery Effectiveness Index", f"{summary.split('Effectiveness Index: ')[1].split('/100')[0]}/100")
                    with c2:
                        st.text_area("Voice Analysis Summary", summary, height=260)

                    st.markdown("#### Performance vs Benchmarks")
                    st.pyplot(fig1, clear_figure=True)

                    st.markdown("#### Extreme Attribute Timestamps")
                    st.dataframe(extremes_df, use_container_width=True)

                    st.markdown("#### Waveform & Spectrogram")
                    st.pyplot(fig2, clear_figure=True)

                except subprocess.CalledProcessError as e:
                    st.error("ffmpeg error during audio extraction.")
                    st.code(e.stderr.decode("utf-8") if e.stderr else str(e))
                except Exception as e:
                    st.exception(e)

# ----------------------- TAB 2: Video Frame Extractor -----------------------------
with tab2:
    st.subheader("Extract frames from a video at a given sampling interval")
    vid2 = st.file_uploader("Upload Video", type=["mp4", "mov", "avi", "mkv"], key="frame_vid")
    interval = st.slider("Frame Interval (seconds)", min_value=1, max_value=30, step=1, value=10)
    st.info(f"Sampling rate: 1 frame every {interval} seconds.")
    if st.button("Extract Frames"):
        if not vid2:
            st.warning("Please upload a video first.")
        else:
            start_t = time.time()
            with st.spinner("Extracting frames…"):
                try:
                    with tempfile.TemporaryDirectory() as tmp:
                        video_path = save_uploaded_file(vid2, tmp)
                        stats = get_video_stats(video_path)
                        frame_dir = Path(tmp) / "frames"
                        frame_dir.mkdir(parents=True, exist_ok=True)
                        paths, stamps = extract_frames_ffmpeg(video_path, interval, str(frame_dir))
                        elapsed = time.time() - start_t

                        st.markdown("#### Video Statistics")
                        display_stats_table(stats, key="stats_table")

                        st.markdown("#### Sampled Frames (odd indices)")
                        cols = st.columns(3)
                        for i in range(3):
                            if i < len(paths):
                                img = cv2.imread(paths[i])
                                cols[i].image(bgr_to_rgb(img), caption=f"t ≈ {stamps[i]:.1f}s", use_column_width=True)
                            else:
                                cols[i].info("No frame")

                        st.text(f"Process took {elapsed:.2f} seconds.")
                except subprocess.CalledProcessError as e:
                    st.error("ffmpeg error during frame extraction.")
                    st.code(e.stderr.decode("utf-8") if e.stderr else str(e))
                except Exception as e:
                    st.exception(e)

# ----------------------- TAB 3: Facial Sentiment Analysis -------------------------
with tab3:
    st.subheader("Detect faces and estimate emotions on images/videos")
    media = st.file_uploader("Upload Image or Video", type=["png", "jpg", "jpeg", "mp4", "avi", "mov", "mkv"], key="sent_media")
    frame_int = st.slider("Frame Interval for Videos (seconds)", min_value=1, max_value=30, step=1, value=10, key="sent_interval")

    cA, cB = st.columns([1, 2], gap="large")
    with cA:
        if st.button("Show File Stats"):
            if not media:
                st.warning("Please upload a file.")
            else:
                with tempfile.TemporaryDirectory() as tmp:
                    path = save_uploaded_file(media, tmp)
                    stats = describe_file(path)
                    display_stats_table(stats, key="desc_table")

    with cB:
        if media:
            # Show a quick preview
            with tempfile.TemporaryDirectory() as tmp:
                p = save_uploaded_file(media, tmp)
                ext = Path(p).suffix.lower()
                if ext in [".png", ".jpg", ".jpeg"]:
                    st.image(bgr_to_rgb(cv2.imread(p)), caption="Input Preview", use_container_width=True)
                else:
                    st.video(p)

    if st.button("Process (Detect + Sentiment)", type="primary"):
        if not media:
            st.warning("Please upload a file.")
        else:
            try:
                with st.spinner("Running face detection and emotion analysis…"):
                    with tempfile.TemporaryDirectory() as tmp:
                        p = save_uploaded_file(media, tmp)
                        images, df = process_media_for_sentiment(p, frame_interval=frame_int)

                    st.markdown("#### Annotated Frames")
                    if images:
                        # Display as rows of up to 3 images
                        for i in range(0, len(images), 3):
                            row = images[i:i+3]
                            cols = st.columns(3)
                            for j, im in enumerate(row):
                                cols[j].image(im, use_column_width=True)
                    else:
                        st.info("No faces detected.")

                    st.markdown("#### Sentiment Analysis Results")
                    st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.exception(e)

st.divider()
st.caption("Built with Streamlit • ffmpeg via imageio-ffmpeg • librosa • OpenCV • DeepFace")
