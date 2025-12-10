import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import os

def analyze_audio(file_path, label, color):
    print(f"Analyzing {label}: {file_path}...")
    try:
        y, sr = librosa.load(file_path)
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return None

    # 1. Pitch (F0) Extraction using pYIN
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    # Filter only voiced parts
    f0_clean = f0[~np.isnan(f0)]
    
    # 2. Spectral Centroid (Brightness/Timbre)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    
    # 3. RMS Energy (Loudness dynamics)
    rms = librosa.feature.rms(y=y)[0]
    
    stats = {
        "label": label,
        "pitch_mean": np.mean(f0_clean) if len(f0_clean) > 0 else 0,
        "pitch_std": np.std(f0_clean) if len(f0_clean) > 0 else 0,
        "pitch_range": (np.max(f0_clean) - np.min(f0_clean)) if len(f0_clean) > 0 else 0,
        "energy_std": np.std(rms),
        "timbre_complexity": np.std(spectral_centroids),
        "y": y,
        "sr": sr,
        "f0": f0,
        "times": librosa.times_like(f0),
        "color": color
    }
    return stats

def generate_comparison_chart(gemini_stats, ours_stats):
    if not gemini_stats or not ours_stats:
        return

    plt.figure(figsize=(12, 12))
    plt.suptitle("Fidelity Analysis: Gemini 2.5 (Reference) vs PsycheTTS Placeholder (Legacy)", fontsize=16)

    # 1. Waveforms
    plt.subplot(3, 2, 1)
    librosa.display.waveshow(gemini_stats['y'], sr=gemini_stats['sr'], color=gemini_stats['color'], alpha=0.6)
    plt.title(f"Gemini Waveform (Authentic)")
    plt.xlabel("")

    plt.subplot(3, 2, 2)
    librosa.display.waveshow(ours_stats['y'], sr=ours_stats['sr'], color=ours_stats['color'], alpha=0.6)
    plt.title(f"PsycheTTS Waveform (Legacy Placeholder)")
    plt.xlabel("")

    # 2. Spectrograms (Richness)
    plt.subplot(3, 2, 3)
    D_gemini = librosa.amplitude_to_db(np.abs(librosa.stft(gemini_stats['y'])), ref=np.max)
    librosa.display.specshow(D_gemini, y_axis='log', x_axis='time', sr=gemini_stats['sr'])
    plt.colorbar(format='%+2.0f dB')
    plt.title("Gemini Spectrogram (Note Harmonics)")

    plt.subplot(3, 2, 4)
    D_ours = librosa.amplitude_to_db(np.abs(librosa.stft(ours_stats['y'])), ref=np.max)
    librosa.display.specshow(D_ours, y_axis='log', x_axis='time', sr=ours_stats['sr'])
    plt.colorbar(format='%+2.0f dB')
    plt.title("PsycheTTS Spectrogram (Note Flatness)")

    # 3. Pitch Contour (Prosody)
    plt.subplot(3, 1, 3)
    plt.plot(gemini_stats['times'], gemini_stats['f0'], label=f"Gemini (StdDev: {gemini_stats['pitch_std']:.1f} Hz)", color=gemini_stats['color'], linewidth=2)
    plt.plot(ours_stats['times'], ours_stats['f0'], label=f"PsycheTTS (StdDev: {ours_stats['pitch_std']:.1f} Hz)", color=ours_stats['color'], linewidth=2, linestyle="--")
    plt.title("Prosody Curve (Pitch Movement)")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = "assets/analysis_chart.png"
    plt.savefig(output_path)
    print(f"✅ Generated Chart: {output_path}")

    # Text Report
    print("\n--- 🔎 AUDIO FORENSICS REPORT ---")
    print(f"METRIC                 | GEMINI (SOTA)      | PSYCHE (LEGACY)")
    print(f"-----------------------|--------------------|----------------")
    print(f"Intonation (Pitch Std) | {gemini_stats['pitch_std']:<18.2f} | {ours_stats['pitch_std']:.2f} (Flatter)")
    print(f"Dynamic Range (Hz)     | {gemini_stats['pitch_range']:<18.2f} | {ours_stats['pitch_range']:.2f}")
    print(f"Energy Var (Emotion)   | {gemini_stats['energy_std']:<18.4f} | {ours_stats['energy_std']:.4f}")
    print(f"Timbre Complexity      | {gemini_stats['timbre_complexity']:<18.2f} | {ours_stats['timbre_complexity']:.2f}")

def main():
    gemini_path = "assets/audio/ref_gemini_sinhala.wav"
    ours_path = "assets/audio/ours_psyche_sinhala.m4a"

    if not os.path.exists(gemini_path):
        print(f"Missing {gemini_path}")
        return

    gemini_stats = analyze_audio(gemini_path, "Gemini 2.5 Pro", "#007AFF") # Apple Blue
    ours_stats = analyze_audio(ours_path, "PsycheTTS (v1)", "#FF9F0A")   # Apple Orange

    generate_comparison_chart(gemini_stats, ours_stats)

if __name__ == "__main__":
    main()
