import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from pathlib import Path
import pandas as pd
import subprocess

# Set page config
st.set_page_config(page_title="Audio Voice Sample Analyzer", layout="wide")

st.title("🎙️ Audio Voice Sample Analyzer")
st.markdown("Upload a video to extract and analyze its audio voice characteristics. Ideal for calls, pitches, or podcasts.")

# Create temporary directory
temp_dir = "temp"
os.makedirs(temp_dir, exist_ok=True)

def analyze_audio_from_video(video_file):
    try:
        # Set up paths
        video_path = os.path.join(temp_dir, "uploaded_video.mp4")
        audio_path = os.path.join(temp_dir, "temp_audio.wav")

        # Save uploaded file
        with open(video_path, "wb") as f:
            f.write(video_file.read())

        # Step 1: Extract audio from video and get clip stats
        result = subprocess.run(
            ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', audio_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise Exception(f"FFmpeg error: {result.stderr}")

        # Get video duration and size
        probe = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            text=True, capture_output=True
        )
        duration = float(probe.stdout.strip()) if probe.stdout.strip() else 0
        minutes, seconds = divmod(duration, 60)
        file_size = os.path.getsize(video_path) / 1024  # KB
        size_unit = "KB"
        if file_size > 1024:
            file_size /= 1024
            size_unit = "MB"
        clip_stats = f"Clip Statistics:\n- Duration: {int(minutes)}m {int(seconds)}s\n- File Size: {file_size:.2f} {size_unit}"

        # Step 2: Analyze audio features
        y, sr = librosa.load(audio_path)
        duration_sec = len(y) / sr

        # Pitch analysis
        pitch, _ = librosa.piptrack(y=y, sr=sr)
        pitch_vals = pitch[pitch > 0]
        pitch_mean = np.mean(pitch_vals) if pitch_vals.size else 0
        pitch_std = np.std(pitch_vals) if pitch_vals.size else 0
        typical_pitch_range = 150 if pitch_mean > 150 else 100

        # Energy (RMS)
        energy = np.mean(librosa.feature.rms(y=y)[0])
        ideal_energy_range = 0.15

        # Speaking rate (approximation)
        silent_intervals = librosa.effects.split(y, top_db=20)
        speaking_rate = len(silent_intervals) / duration_sec * 60 if duration_sec > 0 else 0
        typical_speaking_rate = 120

        # Pause frequency
        pause_count = max(0, len(silent_intervals) - 1)
        pause_frequency = pause_count / duration_sec * 60 if duration_sec > 0 else 0

        # Delivery Effectiveness Index
        pitch_score = max(0, min(1, 1 - abs(pitch_mean - typical_pitch_range) / typical_pitch_range)) * 100
        energy_score = max(0, min(1, energy / ideal_energy_range)) * 100
        rate_score = max(0, min(1, speaking_rate / typical_speaking_rate)) * 100
        effectiveness_index = (pitch_score + energy_score + rate_score) / 3

        # Step 3: Find extreme attributes with timestamps
        time_points = np.linspace(0, duration_sec, len(y))

        max_amp_idx = np.argmax(np.abs(y))
        max_amp_time = time_points[max_amp_idx]
        max_amp_val = y[max_amp_idx]

        min_amp_idx = np.argmin(np.abs(y))
        min_amp_time = time_points[min_amp_idx]
        min_amp_val = y[min_amp_idx]

        # Max pitch
        max_pitch_idx = np.unravel_index(np.argmax(pitch), pitch.shape)[1] if pitch.size > 0 else 0
        max_pitch_time = time_points[max_pitch_idx] if max_pitch_idx < len(time_points) else 0
        max_pitch_val = np.max(pitch) if pitch.size > 0 else 0

        # Min pitch (excluding silence)
        valid_pitch = pitch[pitch > 0]
        if valid_pitch.size > 0:
            min_pitch_idx = np.unravel_index(np.argmin(pitch), pitch.shape)[1]
            min_pitch_time = time_points[min_pitch_idx] if min_pitch_idx < len(time_points) else 0
            min_pitch_val = np.min(valid_pitch)
        else:
            min_pitch_time = 0
            min_pitch_val = 0

        extremes_table = pd.DataFrame({
            'Attribute': ['Max Amplitude', 'Min Amplitude', 'Max Pitch', 'Min Pitch'],
            'Value': [f"{max_amp_val:.2f}", f"{min_amp_val:.2f}", f"{max_pitch_val:.2f} Hz", f"{min_pitch_val:.2f} Hz"],
            'Timestamp (s)': [f"{max_amp_time:.2f}", f"{min_amp_time:.2f}", f"{max_pitch_time:.2f}", f"{min_pitch_time:.2f}"]
        })

        # Step 4: Visualize results

        # Performance Plot
        metrics = ['Pitch (Hz)', 'Energy', 'Speaking Rate (wpm)']
        values = [pitch_mean, energy, speaking_rate]
        ideal_values = [typical_pitch_range, ideal_energy_range, typical_speaking_rate]
        relative_values = [v / i if i != 0 else 0 for v, i in zip(values, ideal_values)]

        colors = ['green' if v >= 0.8 else 'yellow' if v >= 0.5 else 'red' for v in relative_values]

        fig1, ax1 = plt.subplots(figsize=(10, 5))
        bars = ax1.bar(metrics, relative_values, color=colors)
        ax1.axhline(y=1.0, color='b', linestyle='--', label='Ideal Level')
        ax1.set_title('Voice Characteristics vs. Ideal Benchmarks (Relative)')
        ax1.set_ylabel('Relative Performance (Ideal = 1.0)')
        ax1.legend()
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        performance_plot_path = os.path.join(temp_dir, "performance_plot.png")
        plt.tight_layout()
        plt.savefig(performance_plot_path)
        plt.close()

        # Technical Plot: Waveform + Spectrogram
        fig2, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Waveform
        axes[0].plot(y[:len(y)//10], label='Waveform', color='blue')
        axes[0].axhline(y=0.1, color='green', linestyle='--', label='Ideal Loudness Upper (~0.1)')
        axes[0].axhline(y=-0.1, color='green', linestyle='--', label='Ideal Loudness Lower (~-0.1)')
        axes[0].set_title('Audio Waveform with Ideal Loudness Zones')
        axes[0].set_xlabel('Time (samples)')
        axes[0].set_ylabel('Amplitude')
        axes[0].legend()
        axes[0].grid(True)

        # Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, y_axis='linear', x_axis='time', sr=sr, ax=axes[1])
        fig2.colorbar(img, ax=axes[1], format='%+2.0f dB')
        axes[1].axhline(y=100, color='orange', linestyle='--', label='Ideal Freq Lower (~100 Hz)')
        axes[1].axhline(y=300, color='orange', linestyle='--', label='Ideal Freq Upper (~300 Hz)')
        axes[1].set_title('Spectrogram with Ideal Frequency Zones')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Frequency (Hz)')
        axes[1].legend()
        tech_plot_path = os.path.join(temp_dir, "tech.png")
        plt.tight_layout()
        plt.savefig(tech_plot_path)
        plt.close()

        # Step 5: Summary report
        pitch_status = "High (excited/stressed)" if pitch_mean > typical_pitch_range * 1.5 else "Normal" if pitch_mean > typical_pitch_range * 0.8 else "Low (calm/monotone)"
        energy_status = "Low (monotone)" if energy < ideal_energy_range * 0.5 else "Normal" if energy < ideal_energy_range * 1.5 else "High (energetic)"
        rate_status = "Slow (hesitant)" if speaking_rate < typical_speaking_rate * 0.5 else "Normal" if speaking_rate < typical_speaking_rate * 1.2 else "Fast (rushed)"
        recommendations = (
            f"- Increase pitch variation for engagement if {pitch_status}.\n"
            f"- Boost energy with louder delivery if {energy_status}.\n"
            f"- Adjust pace to {typical_speaking_rate} wpm if {rate_status}.\n"
            f"- Reduce pauses ({pause_frequency:.1f}/min) for smoother flow if frequent."
        )

        summary = (
            f"Voice Analysis Report:\n"
            f"- Average Pitch: {pitch_mean:.2f} Hz (Std: {pitch_std:.2f} Hz) - {pitch_status}\n"
            f"- Energy Level: {energy:.2f} - {energy_status}\n"
            f"- Speaking Rate: {speaking_rate:.1f} wpm - {rate_status}\n"
            f"- Pause Frequency: {pause_frequency:.1f} pauses/min\n"
            f"- Delivery Effectiveness Index: {effectiveness_index:.1f}/100\n"
            f"Business Insights:\n"
            f"This delivery may struggle to engage audiences due to {energy_status.lower()} energy and {rate_status.lower()} pace.\n"
            f"Recommendations:\n{recommendations}"
        )

        return clip_stats, summary, performance_plot_path, extremes_table, tech_plot_path

    except Exception as e:
        return f"Error: {str(e)}", "", None, None, None
    finally:
        # Cleanup not done here — we'll clean up after Streamlit renders
        pass

# File uploader
uploaded_video = st.file_uploader("📤 Upload Video (MP4)", type=["mp4"])

if uploaded_video is not None:
    with st.spinner("Analyzing audio... This may take a minute."):
        clip_stats, summary, perf_plot, extremes_df, tech_plot = analyze_audio_from_video(uploaded_video)

    # Display results in tabs
    tab1, tab2, tab3 = st.tabs(["📋 Clip Stats & Summary", "📊 Voice Analysis", "🔬 Technical Analysis"])

    with tab1:
        st.subheader("Clip Statistics")
        st.text(clip_stats)
        st.subheader("Analysis Summary")
        st.text_area("Voice Analysis Summary", summary, height=300)

    with tab2:
        col1, col2 = st.columns([2, 1])
        with col1:
            if perf_plot and os.path.exists(perf_plot):
                st.image(perf_plot, caption="Performance vs. Benchmarks", use_column_width=True)
        with col2:
            if extremes_df is not None:
                st.write("### Extreme Attribute Timestamps")
                st.dataframe(extremes_df, use_container_width=True)

    with tab3:
        if tech_plot and os.path.exists(tech_plot):
            st.image(tech_plot, caption="Waveform and Spectrogram with Ideal Zones", use_column_width=True)

# Cleanup after rendering (optional, Streamlit temp files are usually cleaned automatically)
# Uncomment if you want explicit cleanup
# if os.path.exists(temp_dir):
#     shutil.rmtree(temp_dir)
