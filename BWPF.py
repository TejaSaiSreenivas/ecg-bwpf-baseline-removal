# BWPF: Baseline Wandering Path Finding Algorithm for ECG Denoising
# Based on: "Baseline wandering removal from ECG signal by wandering path
#            finding algorithm" - IEEE EICT 2017 (DOI: 10.1109/EICT.2017.8275164)

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# 1. SYNTHETIC ECG SIGNAL GENERATOR

def generate_synthetic_noisy_ecg(num_heartbeats=10, sampling_rate=360):
    """
    Generates a mock ECG signal with severe baseline wandering for testing.
    Returns: (noisy_ecg, clean_ecg, time_axis)
    """
    # Create a single idealized heartbeat (P-QRS-T complex)
    t_beat = np.linspace(0, 1, sampling_rate)
    p_wave     = 0.2 * np.exp(-((t_beat - 0.20) ** 2) / 0.001)
    qrs_complex = (  1.0 * np.exp(-((t_beat - 0.40) ** 2) / 0.0005)
                   - 0.2 * np.exp(-((t_beat - 0.38) ** 2) / 0.0001)
                   - 0.3 * np.exp(-((t_beat - 0.43) ** 2) / 0.0001))
    t_wave     = 0.4 * np.exp(-((t_beat - 0.70) ** 2) / 0.004)
    single_beat = p_wave + qrs_complex + t_wave

    # Tile beats into a full ECG strip
    clean_ecg    = np.tile(single_beat, num_heartbeats)
    total_samples = len(clean_ecg)
    time_axis    = np.arange(total_samples)

    # Baseline wander: two low-frequency sine waves (simulates respiration/movement)
    wander = (1.5 * np.sin(2 * np.pi * 0.10 * (time_axis / sampling_rate)) +
              0.8 * np.sin(2 * np.pi * 0.05 * (time_axis / sampling_rate)))

    # High-frequency noise: EMG / muscle artifact
    noise = np.random.normal(0, 0.05, total_samples)

    noisy_ecg = clean_ecg + wander + noise
    return noisy_ecg, clean_ecg, time_axis

# 2. BWPF ALGORITHM (with overlap-add smoothing to fix boundary discontinuities)

def bwpf_remove_baseline(signal, window_size, poly_degree=3):
    """
    Implements the BWPF (Baseline Wandering Path Finding) algorithm using
    piecewise polynomial fitting with 25% overlap-add blending to eliminate
    sharp discontinuities at segment boundaries.

    Parameters:
    signal      : 1D numpy array — raw noisy ECG
    window_size : int — number of samples per segment
    poly_degree : int — polynomial degree for fitting (default=3, cubic)

    Returns:
    clean_signal   : signal with baseline removed
    wandering_path : estimated baseline curve
    """
    n_samples     = len(signal)
    wandering_path = np.zeros(n_samples)
    overlap       = window_size // 4          # 25% overlap for smooth blending
    step          = window_size - overlap     # advance step per iteration

    for i in range(0, n_samples, step):
        start_idx = i
        end_idx   = min(i + window_size, n_samples)

        x_segment    = np.arange(start_idx, end_idx)
        y_segment    = signal[start_idx:end_idx]

        # Fit a polynomial via Ordinary Least Squares
        coefficients = np.polyfit(x_segment, y_segment, poly_degree)
        fitted_curve = np.polyval(coefficients, x_segment)

        seg_len = end_idx - start_idx

        if i == 0:
            # First segment: write directly
            wandering_path[start_idx:end_idx] = fitted_curve
        else:
            # Blend the overlap region with linear cross-fade
            blend_end = min(start_idx + overlap, n_samples)
            blend_len = blend_end - start_idx

            if blend_len > 0:
                weights_new = np.linspace(0, 1, blend_len)
                weights_old = 1.0 - weights_new
                wandering_path[start_idx:blend_end] = (
                    weights_old * wandering_path[start_idx:blend_end] +
                    weights_new * fitted_curve[:blend_len]
                )

            # Non-overlapping tail: write directly
            if blend_end < end_idx:
                wandering_path[blend_end:end_idx] = fitted_curve[blend_len:]

    clean_signal = signal - wandering_path
    return clean_signal, wandering_path

# 3. CONVENTIONAL HIGH-PASS FILTER (Butterworth) — Comparison Baseline

def highpass_filter(signal, cutoff_hz=0.5, fs=360, order=4):
    """
    4th-order Butterworth high-pass filter.
    Cutoff at 0.5 Hz removes baseline wander (< 0.5 Hz) while preserving ECG.
    """
    nyquist   = fs / 2.0
    norm_cutoff = cutoff_hz / nyquist
    b, a = butter(order, norm_cutoff, btype='high', analog=False)
    return filtfilt(b, a, signal)          # zero-phase forward-backward filter

# 4. PERFORMANCE METRICS

def compute_metrics(reference, estimated, label=""):
    """Computes MSE and SNR between the reference clean signal and the result."""
    mse          = np.mean((estimated - reference) ** 2)
    signal_power = np.mean(reference ** 2)
    snr          = 10 * np.log10(signal_power / mse) if mse > 0 else float('inf')
    print(f"  [{label}]  MSE = {mse:.6f}   |   SNR = {snr:.2f} dB")
    return mse, snr

# 5. MAIN EXECUTION

if __name__ == "__main__":

    # --- Parameters ---
    sampling_rate          = 360    # Hz (PhysioNet MIT-BIH standard)
    num_heartbeats         = 12
    window_length_seconds  = 1.0    # 0.8–1.2 s is optimal to isolate wander
    window_samples         = int(window_length_seconds * sampling_rate)
    degree_of_polynomial   = 3      # Cubic polynomial

    # --- Generate Signal ---
    noisy_ecg, original_clean_ecg, time_axis = generate_synthetic_noisy_ecg(
        num_heartbeats=num_heartbeats, sampling_rate=sampling_rate
    )

    # --- Run BWPF Algorithm ---
    bwpf_filtered_ecg, estimated_wander = bwpf_remove_baseline(
        noisy_ecg,
        window_size=window_samples,
        poly_degree=degree_of_polynomial
    )

    # --- Run Conventional High-Pass Filter ---
    hp_filtered_ecg = highpass_filter(noisy_ecg, cutoff_hz=0.5, fs=sampling_rate)

    # --- Print Metrics ---
    print("\n=== Performance Metrics (vs. Ground Truth Clean ECG) ===")
    compute_metrics(original_clean_ecg, bwpf_filtered_ecg, label="BWPF Algorithm ")
    compute_metrics(original_clean_ecg, hp_filtered_ecg,   label="High-Pass Filter")

    # --- Convert samples to time (seconds) for readable x-axis ---
    time_sec = time_axis / sampling_rate

    # --- Plot Results ---
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("ECG Baseline Wander Removal — BWPF Algorithm(IEEE EICT 2017, DOI: 10.1109/EICT.2017.8275164)",
                 fontsize=13, fontweight='bold')

    # Plot 1: Raw noisy ECG + estimated wandering path
    axes[0].plot(time_sec, noisy_ecg, color='gray', alpha=0.75,
                 label="Raw ECG with Baseline Wander")
    axes[0].plot(time_sec, estimated_wander, color='red', linewidth=2,
                 label="Estimated Wandering Path (BWPF)")
    axes[0].set_title("Step 1 — Raw Signal & Estimated Baseline Path")
    axes[0].set_ylabel("Amplitude (mV)")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.4)

    # Plot 2: BWPF cleaned output
    axes[1].plot(time_sec, bwpf_filtered_ecg, color='blue',
                 label="BWPF Cleaned ECG")
    axes[1].plot(time_sec, original_clean_ecg, color='green', alpha=0.5,
                 linestyle='--', label="Ground Truth")
    axes[1].set_title("Step 2 — BWPF Output vs. Ground Truth")
    axes[1].set_ylabel("Amplitude (mV)")
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.4)

    # Plot 3: Conventional Butterworth high-pass filter output
    axes[2].plot(time_sec, hp_filtered_ecg, color='darkorange',
                 label="High-Pass Filter (0.5 Hz Butterworth)")
    axes[2].plot(time_sec, original_clean_ecg, color='green', alpha=0.5,
                 linestyle='--', label="Ground Truth")
    axes[2].set_title("Step 3 — Conventional High-Pass Filter Output vs. Ground Truth")
    axes[2].set_ylabel("Amplitude (mV)")
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.4)

    # Plot 4: Residual error comparison
    bwpf_error = bwpf_filtered_ecg - original_clean_ecg
    hp_error   = hp_filtered_ecg   - original_clean_ecg
    axes[3].plot(time_sec, bwpf_error, color='blue',       alpha=0.7, label="BWPF Error")
    axes[3].plot(time_sec, hp_error,   color='darkorange',  alpha=0.7, label="HP Filter Error")
    axes[3].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[3].set_title("Step 4 — Residual Error (Filtered − Ground Truth)")
    axes[3].set_xlabel("Time (seconds)")
    axes[3].set_ylabel("Error (mV)")
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig("BWPF_result.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("\nFigure saved as: BWPF_result.png")
