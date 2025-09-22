# app.py
import streamlit as st
import librosa
import librosa.display  # needed for specshow
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import shutil
import os
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

st.set_page_config(page_title="Audio Voice Sample Analyzer", layout="wide")

# ---------- Utilities ----------

def _check_ffmpeg():
    """Ensure ffmpeg & ffprobe exist on the PATH."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError(
            "ffmpeg/ffprobe not found. On Streamlit Cloud, add a `packages.txt` file with:\n\nffmpeg"
        )

def _save_plot_to_png_bytes():
    """Save current Matplotlib figure to PNG bytes and close it."""
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf

def _extract_audio_ffmpeg(video_path: str, audio_path: str):
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", audio_path],
        check=True,
        timeout=60,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    )

def _probe_duration(video_path: str) -> float:
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path,
        ],
        text=True, capture_output=True
    )
    return float(probe.stdout.strip()) if probe.stdout.strip() else 0.0

# ---------- Core analysis ----------

def analyze_audio_from_video(uploaded_file) -> dict:
    """
    Returns a dict containing:
      clip_stats: str
      summary: str
      performance_plot_png: bytes
      extremes_table_df: pd.DataFrame
      tech_plot_png: bytes
    """
    _check_ffmpeg()

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        video_path = tmpdir / "uploaded_video.mp4"
        audio_path = tmpdir / "temp_audio.wav"

        # Persist uploaded file to disk
        video_path.write_bytes(uploaded_file.read())

        # 1) Extract audio
        _extract_audio_ffmpeg(str(video_path), str(audio_path))

        # Clip stats
        duration_sec = _probe_duration(str(video_path))
        minutes, seconds = divmod(int(duration_sec), 60)
        file_size_kb = video_path.stat().st_size / 1024.0
        size_value = file_size_kb
        size_unit = "KB"
        if file_size_kb > 1024:
            size_value = file_size_kb / 1024.0
            size_unit = "MB"
        clip_stats = (
            f"Clip Statistics:\n"
            f"- Duration: {minutes}m {seconds}s\n"
            f"- File Size: {size_value:.2f} {size_unit}"
        )

        # 2) Analyze audio features
        # Use native sampling rate to avoid resampling artifacts
        y, sr = librosa.load(str(audio_path), sr=None)
        total_duration = len(y) / sr if sr else 0.0

        # Pitch via piptrack (2D: n_freqs x n_frames)
        pitches, mags = librosa.piptrack(y=y, sr=sr)
        positive = pitches > 0
        pitch_vals = pitches[positive]
        pitch_mean = float(np.mean(pitch_vals)) if pitch_vals.size else 0.0
        pitch_std = float(np.std(pitch_vals)) if pitch_vals.size else 0.0
        # Rough male/female typical range pivot like your original logic
        typical_pitch_range = 150 if pitch_mean > 150 else 100

        # Energy (RMS)
        rms = librosa.feature.rms(y=y)[0]
        energy = float(np.mean(rms)) if rms.size else 0.0
        ideal_energy_range = 0.15

        # Speaking rate approximation (using non-silent segments)
        # librosa.effects.split returns non-silent intervals
        nonsilent_intervals = librosa.effects.split(y, top_db=20)
        speaking_rate = (len(nonsilent_intervals) / total_duration * 60.0) if total_duration > 0 else 0.0

        typical_speaking_rate = 120.0  # wpm heuristic

        # Pause frequency (gaps between non-silent intervals)
        pause_count = max(0, len(nonsilent_intervals) - 1)
        pause_frequency = (pause_count / total_duration * 60.0) if total_duration > 0 else 0.0

        # Delivery Effectiveness Index (like your weights)
        pitch_score = max(0, min(1, 1 - abs(pitch_mean - typical_pitch_range) / max(typical_pitch_range, 1))) * 100
        energy_score = max(0, min(1, energy / max(ideal_energy_range, 1e-6))) * 100
        rate_score = max(0, min(1, speaking_rate / max(typical_speaking_rate, 1e-6))) * 100
        effectiveness_index = (pitch_score + energy_score + rate_score) / 3.0

        # 3) Extremes with timestamps
        # Amplitude extremes (sample index -> seconds)
        if len(y) > 0:
            max_amp_idx = int(np.argmax(y))
            min_amp_idx = int(np.argmin(y))
            max_amp_time = max_amp_idx / sr
            min_amp_time = min_amp_idx / sr
        else:
            max_amp_time = min_amp_time = 0.0

        # Pitch extremes: summarize per-frame to map to time
        # Take the max pitch per frame (over freq bins), then filter zeros
        if pitches.size:
            frame_pitch = np.max(pitches, axis=0)  # length = n_frames
            frame_idx_max_pitch = int(np.argmax(frame_pitch)) if frame_pitch.size else 0
            max_pitch_value = float(frame_pitch[frame_idx_max_pitch]) if frame_pitch.size else 0.0

            # Min positive pitch across all frames
            positive_frame_pitch = frame_pitch[frame_pitch > 0]
            if positive_frame_pitch.size:
                min_pitch_value = float(np.min(positive_frame_pitch))
                # Frame index for the first occurrence of the min value
                frame_idx_min_pitch = int(np.where(frame_pitch == min_pitch_value)[0][0])
            else:
                min_pitch_value = 0.0
                frame_idx_min_pitch = 0

            # Map frame indices to times
            max_pitch_time = librosa.frames_to_time(frame_idx_max_pitch, sr=sr)
            min_pitch_time = librosa.frames_to_time(frame_idx_min_pitch, sr=sr)
        else:
            max_pitch_value = min_pitch_value = 0.0
            max_pitch_time = min_pitch_time = 0.0

        extremes_table = pd.DataFrame({
            'Attribute': ['Max Amplitude', 'Min Amplitude', 'Max Pitch', 'Min Pitch'],
            'Value': [
                f"{float(np.max(y)):.2f}" if len(y) else "0.00",
                f"{float(np.min(y)):.2f}" if len(y) else "0.00",
                f"{max_pitch_value:.2f} Hz",
                f"{min_pitch_value:.2f} Hz"
            ],
            'Timestamp (s)': [
                f"{max_amp_time:.2f}",
                f"{min_amp_time:.2f}",
                f"{max_pitch_time:.2f}",
                f"{min_pitch_time:.2f}"
            ]
        })

        # 4) Visualizations

        # 4a) Performance vs Benchmarks (relative bars)
        metrics = ['Pitch (Hz)', 'Energy', 'Speaking Rate (wpm)']
        values = [pitch_mean, energy, speaking_rate]
        ideals = [float(typical_pitch_range), float(ideal_energy_range), float(typical_speaking_rate)]
        relatives = [v / i if i != 0 else 0 for v, i in zip(values, ideals)]

        plt.figure(figsize=(8, 4.5))
        # simple color coding like your original
        colors = ['green' if v >= 0.8 else 'yellow' if v >= 0.5 else 'red' for v in relatives]
        bars = plt.bar(metrics, relatives, color=colors)
        plt.axhline(y=1.0, color='b', linestyle='--', label='Ideal Level')
        plt.title('Voice Characteristics vs. Ideal Benchmarks (Relative)')
        plt.ylabel('Relative Performance (Ideal = 1.0)')
        plt.ylim(0, max(1.2, max(relatives + [1.0]) + 0.1))
        plt.legend()
        performance_plot_png = _save_plot_to_png_bytes()

        # 4b) Waveform + Spectrogram with ideal zones
        # Waveform (first 10% for detail, as in your code)
        plt.figure(figsize=(10, 6))
        # Top plot: waveform
        plt.subplot(2, 1, 1)
        if len(y):
            n = len(y) // 10 if len(y) // 10 > 0 else len(y)
            t_samples = np.arange(n)
            plt.plot(t_samples, y[:n], label='Waveform', linewidth=0.8)
        plt.axhline(y=0.1, color='green', linestyle='--', label='Ideal Loudness Upper (~0.1)')
        plt.axhline(y=-0.1, color='green', linestyle='--', label='Ideal Loudness Lower (~-0.1)')
        plt.title('Audio Waveform (Zoomed) with Ideal Loudness Zones')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.legend()

        # Bottom plot: spectrogram
        plt.subplot(2, 1, 2)
        if len(y):
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            librosa.display.specshow(D, y_axis='linear', x_axis='time', sr=sr)
            plt.colorbar(format='%+2.0f dB')
            plt.axhline(y=100, color='orange', linestyle='--', label='Ideal Freq Lower (~100 Hz)')
            plt.axhline(y=300, color='orange', linestyle='--', label='Ideal Freq Upper (~300 Hz)')
        plt.title('Spectrogram with Ideal Frequency Zones')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.legend()
        tech_plot_png = _save_plot_to_png_bytes()

        # 5) Summary/insights
        pitch_status = "High (excited/stressed)" if pitch_mean > typical_pitch_range * 1.5 else \
                       "Normal" if pitch_mean > typical_pitch_range * 0.8 else \
                       "Low (calm/monotone)"
        energy_status = "Low (monotone)" if energy < ideal_energy_range * 0.5 else \
                        "Normal" if energy < ideal_energy_range * 1.5 else \
                        "High (energetic)"
        rate_status = "Slow (hesitant)" if speaking_rate < typical_speaking_rate * 0.5 else \
                      "Normal" if speaking_rate < typical_speaking_rate * 1.2 else \
                      "Fast (rushed)"

        recommendations = (
            f"- Increase pitch variation for engagement if {pitch_status}.\n"
            f"- Boost energy with louder delivery if {energy_status}.\n"
            f"- Adjust pace to ~{int(typical_speaking_rate)} wpm if {rate_status}.\n"
            f"- Reduce pauses ({pause_frequency:.1f}/min) for smoother flow if frequent."
        )

        summary = (
            f"Voice Analysis Report:\n"
            f"- Average Pitch: {pitch_mean:.2f} Hz (Std: {pitch_std:.2f} Hz) - {pitch_status}\n"
            f"- Energy Level: {energy:.2f} - {energy_status}\n"
            f"- Speaking Rate: {speaking_rate:.1f} wpm - {rate_status}\n"
            f"- Pause Frequency: {pause_frequency:.1f} pauses/min\n"
            f"- Delivery Effectiveness Index: {effectiveness_index:.1f}/100\n\n"
            f"Business Insights:\n"
            f"This delivery may struggle to engage audiences due to {energy_status.lower()} energy "
            f"and {rate_status.lower()} pace.\n\n"
            f"Recommendations:\n{recommendations}"
        )

        return {
            "clip_stats": clip_stats,
            "summary": summary,
            "performance_plot_png": performance_plot_png.getvalue(),
            "extremes_table_df": extremes_table,
            "tech_plot_png": tech_plot_png.getvalue(),
        }


# ---------- Streamlit UI ----------

st.title("🎙️ Audio Voice Sample Analyzer")
st.markdown("Upload a video to extract and analyze its audio voice characteristics. Ideal for calls, pitches, or podcasts.")

tab_input, tab_analysis, tab_tech = st.tabs(["Input", "Voice Analysis", "Technical Analysis"])

with tab_input:
    uploaded_video = st.file_uploader("Upload Video (MP4)", type=["mp4"], accept_multiple_files=False)
    analyze_clicked = st.button("Analyze Audio")

    if analyze_clicked:
        if uploaded_video is None:
            st.warning("Please upload an MP4 file first.")
        else:
            with st.spinner("Extracting audio and analyzing…"):
                try:
                    results = analyze_audio_from_video(uploaded_video)
                    # Store in session state so other tabs can read
                    st.session_state["analysis_results"] = results
                    st.success("Analysis complete!")
                except Exception as e:
                    st.error(f"Error: {e}")

    # Show clip stats right away if available
    if "analysis_results" in st.session_state:
        st.text_area("Clip Statistics", st.session_state["analysis_results"]["clip_stats"], height=100)

with tab_analysis:
    if "analysis_results" not in st.session_state:
        st.info("Run an analysis in the **Input** tab to see results here.")
    else:
        res = st.session_state["analysis_results"]
        st.text_area("Voice Analysis Summary", res["summary"], height=300)

        st.subheader("Performance vs. Benchmarks")
        st.image(res["performance_plot_png"], caption="Relative performance (Ideal = 1.0).", use_column_width=True)

        st.subheader("Extreme Attribute Timestamps")
        st.dataframe(res["extremes_table_df"], use_container_width=True)

with tab_tech:
    if "analysis_results" not in st.session_state:
        st.info("Run an analysis in the **Input** tab to see results here.")
    else:
        res = st.session_state["analysis_results"]
        st.image(res["tech_plot_png"], caption="Waveform (zoom) and Spectrogram with ideal zones.", use_column_width=True)

st.caption("Note: Pitch/energy/rate heuristics are approximate and context-dependent.")
