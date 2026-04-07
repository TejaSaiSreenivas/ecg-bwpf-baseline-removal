# ECG Baseline Wander Removal — BWPF Algorithm

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![IEEE Paper](https://img.shields.io/badge/IEEE-DOI%3A10.1109%2FEICT.2017.8275164-orange)](https://ieeexplore.ieee.org/document/8275164)

A Python implementation of the **Baseline Wandering Path Finding (BWPF)** algorithm for ECG signal denoising, based on the IEEE EICT 2017 paper. The algorithm estimates and removes low-frequency baseline drift from electrocardiogram (ECG) signals using piecewise polynomial fitting with overlap-add smoothing.

---

##  Paper Reference

> **"Baseline wandering removal from ECG signal by wandering path finding algorithm"**  
> Presented at the *3rd International Conference on Electrical Information and Communication Technology (EICT 2017)*  
> IEEE. DOI: [10.1109/EICT.2017.8275164](https://ieeexplore.ieee.org/document/8275164)

---

##  Overview

Baseline wander is a low-frequency artifact (< 0.5 Hz) in ECG signals caused by patient respiration, body movement, and poor electrode contact. It significantly distorts the signal morphology, making clinical diagnosis unreliable.

This implementation:
- Segments the ECG signal into overlapping windows
- Fits a piecewise cubic polynomial (degree-3) to each segment using Ordinary Least Squares
- Uses **25% overlap-add blending** to eliminate discontinuities at segment boundaries
- Subtracts the reconstructed baseline path from the raw signal
- Benchmarks against a conventional 4th-order Butterworth high-pass filter
- Reports MSE and SNR (dB) for quantitative evaluation

---

##  Repository Structure

```
ecg-bwpf-baseline-removal/
│
├── BWPF.py           # Core implementation (BWPF + comparison + metrics)
├── BWPF_result.png        # Output figure (auto-generated on run)
├── README.md              # This file
└── requirements.txt       # Python dependencies
```

---

##  Installation

```bash
git clone https://github.com/<your-username>/ecg-bwpf-baseline-removal.git
cd ecg-bwpf-baseline-removal
pip install -r requirements.txt
```

**requirements.txt:**
```
numpy
matplotlib
scipy
```

---

##  Usage

```bash
python BWPF.py
```

The script will:
1. Generate a synthetic noisy ECG (12 heartbeats, 360 Hz)
2. Run the BWPF algorithm
3. Run the conventional Butterworth high-pass filter for comparison
4. Print MSE and SNR metrics to the terminal
5. Display and save a 4-panel figure as `BWPF_result.png`

**Sample Terminal Output:**
```
=== Performance Metrics (vs. Ground Truth Clean ECG) ===
  [BWPF Algorithm ]  MSE = 0.003241   |   SNR = 18.92 dB
  [High-Pass Filter]  MSE = 0.004873   |   SNR = 16.48 dB
```

---

##  Algorithm Summary

```
INPUT : noisy ECG signal s[n], window size W, polynomial degree d
OUTPUT: clean ECG signal, estimated baseline path

1. Divide s[n] into overlapping segments (25% overlap)
2. For each segment i:
     a. Extract samples x[i] and y[i]
     b. Fit polynomial of degree d via np.polyfit (OLS)
     c. Evaluate fitted curve via np.polyval
     d. Blend with previous segment using linear cross-fade
3. Subtract estimated baseline path from s[n]
4. Return clean signal
```

---

##  Output

The 4-panel output figure includes:
| 1 | Raw noisy ECG + estimated BWPF baseline path |
| 2 | BWPF cleaned output vs. ground truth |
| 3 | Butterworth high-pass filter output vs. ground truth |
| 4 | Residual error comparison between both methods |

---

##  Citation

If you use this code in your work, please cite the original paper:

**IEEE Format:**
```
[1] [Author(s)], "Baseline wandering removal from ECG signal by wandering path 
finding algorithm," in Proc. 3rd Int. Conf. Electrical Information and 
Communication Technology (EICT), Khulna, Bangladesh, 2017, 
doi: 10.1109/EICT.2017.8275164.
```

**BibTeX:**
```bibtex
@inproceedings{bwpf_eict2017,
  title     = {Baseline wandering removal from {ECG} signal by wandering path finding algorithm},
  booktitle = {2017 3rd International Conference on Electrical Information and Communication Technology (EICT)},
  year      = {2017},
  pages     = {1--5},
  doi       = {10.1109/EICT.2017.8275164},
  publisher = {IEEE}
}
```

---

##  License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

##  Author

Implemented as part of a Digital Signal Processing coursework MiniProject.  
Paper implementation by **Teja Sai Sreenivas**.
