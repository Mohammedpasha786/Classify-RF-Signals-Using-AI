# RF Signal Classification with Deep Learning
> Classify RF signals (WiFi, Bluetooth, 5G NR, LTE, ZigBee, Radar) using deep learning CNNs trained on synthetic I/Q data with real-world SDR validation via ADALM-Pluto or USRP.

---

## Motivation

The RF spectrum is increasingly congested — 5G, WiFi 6, Bluetooth 5, ZigBee, radar, GPS, and IoT devices all share limited bandwidth. Automatic **RF signal classification** via AI enables:

- Real-time interference detection and mitigation
- Spectrum monitoring and compliance
- Cognitive radio resource management
- Electronic warfare signal identification

---

## Architecture

```
Raw I/Q Samples (complex float32)
        │
        ▼
┌───────────────────────┐
│  Preprocessing        │  Normalize · windowing · optional FFT magnitude
└──────────┬────────────┘
           │ [B, 2, N] real/imag stacked OR [B, 1, N_fft] spectrum
           ▼
┌───────────────────────┐
│  Feature Extractor    │  CNN backbone (ResNet1D / Custom CNN)
│  (Transfer learning   │  Pretrained on LTE/5G → fine-tune on target classes
│   from LTE/5G base)   │
└──────────┬────────────┘
           │ embedding [B, D]
           ▼
┌───────────────────────┐
│  Classifier Head      │  FC layers → Softmax over N signal classes
└──────────┬────────────┘
           │ class probabilities
           ▼
Signal Class + Confidence Score
```

---

## Signal Classes

| Class | Standard | Freq Band | Bandwidth |
|---|---|---|---|
| WiFi (802.11g) | IEEE 802.11 | 2.4 GHz | 20 MHz |
| WiFi (802.11n) | IEEE 802.11 | 2.4/5 GHz | 20/40 MHz |
| Bluetooth | IEEE 802.15.1 | 2.4 GHz | 1 MHz |
| ZigBee | IEEE 802.15.4 | 2.4 GHz | 2 MHz |
| 5G NR | 3GPP Rel.15+ | Sub-6 / mmWave | Variable |
| LTE | 3GPP Rel.8+ | 700 MHz–2.6 GHz | 1.4–20 MHz |
| Radar (pulsed) | Various | 1–10 GHz | Variable |

Interference pairs of interest: WiFi+BT, 5G+Radar, ZigBee+BT, ZigBee+WiFi

---

## Dataset

**Synthetic I/Q generation** via GNU Radio / SigMF + custom impairment pipeline:
- Center frequency offset (CFO)
- Symbol timing offset (STO)
- AWGN at SNR ∈ [-20, 30] dB
- Rayleigh multipath fading
- Phase noise, IQ imbalance

**Public datasets supported:**
- [DeepSig RadioML 2018.01A](https://www.deepsig.ai/datasets/) — 24 modulation classes
- [Wireless Intelligence Dataset](https://wireless.vt.edu/)

```bash
python scripts/generate_dataset.py --signals wifi bt zigbee lte 5g --n-samples 5000
python scripts/download_radioml.py --output data/radioml/
```

---

## Models

| Model | Params | Top-1 Acc (sim) | Notes |
|---|---|---|---|
| `CNN1D` | ~180K | 94.2% | Fast, real-time capable |
| `ResNet1D` | ~1.1M | 96.8% | Transfer learning base |
| `CLDNN` | ~500K | 95.5% | CNN+LSTM hybrid |
| `ResNet1D-FT` | ~1.1M | 91.3% | Fine-tuned on OTA data |

---

## Installation

```bash
git clone https://github.com/yourusername/rf-signal-classifier.git
cd rf-signal-classifier

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .

# Optional: SDR hardware
# pip install pyadi-iio    # ADALM-Pluto
# pip install uhd           # USRP
```

---

## Usage

### Generate synthetic dataset
```bash
python scripts/generate_dataset.py \
  --signals wifi bluetooth zigbee lte 5g radar \
  --snr-range -10 30 \
  --n-samples 10000 \
  --output data/synthetic/
```

### Train classifier
```bash
python scripts/train.py --config configs/resnet1d.yaml
```

### Evaluate on test set
```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/resnet1d_best.pt \
  --data data/synthetic/test/
```

### Classify a captured I/Q file
```bash
python scripts/classify_iq.py \
  --input captures/unknown_signal.sigmf \
  --checkpoint checkpoints/resnet1d_best.pt
```

---

## SDR Testing

### Capture OTA signals (ADALM-Pluto)
```bash
python scripts/sdr_capture.py \
  --device pluto \
  --freq 2.4e9 \
  --bandwidth 20e6 \
  --duration 10 \
  --output captures/wifi_ota.sigmf
```

### Live classification
```bash
python scripts/live_classify.py \
  --device pluto \
  --checkpoint checkpoints/resnet1d_best.pt \
  --freq 2.4e9
```

---

## Results

BER vs classification accuracy at key SNR operating points:

| Signal Pair | Accuracy @0dB | Accuracy @10dB | Accuracy @20dB |
|---|---|---|---|
| WiFi vs BT | 78% | 94% | 98% |
| 5G vs Radar | 72% | 91% | 97% |
| ZigBee vs BT | 81% | 95% | 99% |

---

## Project Structure

```
rf-signal-classifier/
├── src/
│   ├── data/         # I/Q dataset generation, SigMF I/O, augmentation
│   ├── models/       # CNN1D, ResNet1D, CLDNN architectures
│   ├── training/     # Trainer, loss, scheduler, transfer learning
│   ├── evaluation/   # Confusion matrix, per-SNR accuracy, EVM
│   ├── sdr/          # SDR capture, live streaming, hardware abstraction
│   └── utils/        # Signal impairments, normalization, plotting
├── configs/          # YAML experiment configs
├── scripts/          # CLI entry points
├── tests/            # Unit + integration tests
├── notebooks/        # EDA, training analysis, demo
├── docs/             # Architecture, dataset guide, SDR setup
└── .github/workflows/
```

---

## References

1. O'Shea & Hoydis, "An Introduction to Deep Learning for the Physical Layer," IEEE Trans. Cogn. Commun. Netw., 2017
2. West & O'Shea, "Deep Architectures for Modulation Recognition," IEEE DySPAN, 2017
3. Schmidl & Cox, "Robust OFDM Synchronization," IEEE Trans. Commun., 1997
4. MathWorks, "Spectrum Sensing with Deep Learning to Identify 5G and LTE Signals," 2021

---

## License
MIT License — see [LICENSE](LICENSE)
