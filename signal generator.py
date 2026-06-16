Synthetic I/Q waveform generator for RF signal classification.

Generates baseband complex I/Q samples for:
  WiFi (OFDM), Bluetooth (GFSK), ZigBee (O-QPSK),
  LTE (SC-FDMA / OFDM), 5G NR (CP-OFDM), Radar (pulsed LFM)

All outputs: complex64 numpy arrays, normalized to unit RMS power.
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List


# ── Signal class registry ──────────────────────────────────────────────

SIGNAL_CLASSES = [
    "wifi_g", "wifi_n", "bluetooth", "zigbee",
    "lte", "5g_nr", "radar_lfm",
]

CLASS_TO_IDX = {cls: i for i, cls in enumerate(SIGNAL_CLASSES)}
IDX_TO_CLASS = {i: cls for cls, i in CLASS_TO_IDX.items()}


@dataclass
class GeneratorConfig:
    sample_rate:   float = 20e6     # Hz
    n_samples:     int   = 1024     # samples per example
    snr_db:        float = 20.0
    seed:          Optional[int] = None


# ── Base impairment pipeline ───────────────────────────────────────────

def apply_impairments(
    signal: np.ndarray,
    snr_db: float,
    cfo_hz: float = 0.0,
    sample_rate: float = 20e6,
    phase_noise_std: float = 0.01,
    iq_imbalance_db: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Apply channel impairments to a clean baseband signal."""
    if rng is None:
        rng = np.random.default_rng()

    N = len(signal)
    t = np.arange(N) / sample_rate

    # CFO (carrier frequency offset)
    if cfo_hz != 0.0:
        signal = signal * np.exp(1j * 2 * np.pi * cfo_hz * t)

    # Phase noise
    if phase_noise_std > 0:
        pn = rng.standard_normal(N) * phase_noise_std
        signal = signal * np.exp(1j * np.cumsum(pn))

    # IQ imbalance
    if iq_imbalance_db != 0.0:
        amp_imb = 10 ** (iq_imbalance_db / 40.0)
        signal = signal.real * amp_imb + 1j * signal.imag / amp_imb

    # AWGN
    sig_rms = np.sqrt(np.mean(np.abs(signal) ** 2))
    snr_linear = 10 ** (snr_db / 10.0)
    noise_rms = sig_rms / np.sqrt(snr_linear)
    noise = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) * noise_rms / np.sqrt(2)
    return signal + noise


def normalize(signal: np.ndarray) -> np.ndarray:
    """Normalize to unit RMS power."""
    rms = np.sqrt(np.mean(np.abs(signal) ** 2))
    return signal / (rms + 1e-12)


# ── Signal generators ──────────────────────────────────────────────────

class WiFiGenerator:
    """
    802.11g/n-like OFDM waveform.
    64-point FFT, 52 data subcarriers, CP=16, QPSK per subcarrier.
    """
    N_FFT    = 64
    CP_LEN   = 16
    N_DATA   = 48
    N_PILOT  = 4

    def __init__(self, variant: str = "g"):
        self.variant = variant  # "g" (20 MHz) or "n" (40 MHz HT)

    def generate(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        sym_len = self.N_FFT + self.CP_LEN
        n_symbols = max(1, n_samples // sym_len)
        samples = np.zeros(n_symbols * sym_len, dtype=complex)

        data_idx = np.array([i for i in range(1, 27)] + [i for i in range(38, 64)])  # skip DC & pilots

        for i in range(n_symbols):
            freq = np.zeros(self.N_FFT, dtype=complex)
            # Random QPSK on data subcarriers
            qpsk = (2 * rng.integers(0, 2, len(data_idx)) - 1 +
                    1j * (2 * rng.integers(0, 2, len(data_idx)) - 1)) / np.sqrt(2)
            freq[data_idx] = qpsk
            # Known pilot tones
            freq[[7, 21, 43, 57]] = 1.0 + 0j
            time = np.fft.ifft(freq) * np.sqrt(self.N_FFT)
            ofdm = np.concatenate([time[-self.CP_LEN:], time])
            samples[i * sym_len:(i + 1) * sym_len] = ofdm

        return normalize(samples[:n_samples])


class BluetoothGenerator:
    """
    Bluetooth BR/EDR — GFSK modulation at 1 Msym/s.
    BT = 0.5 Gaussian filter.
    """
    SYM_RATE   = 1e6
    MOD_INDEX  = 0.35

    def generate(self, n_samples: int, rng: np.random.Generator,
                 sample_rate: float = 20e6) -> np.ndarray:
        sps = int(sample_rate / self.SYM_RATE)
        n_symbols = n_samples // sps + 1

        bits = rng.integers(0, 2, n_symbols)
        nrz  = 2 * bits.astype(float) - 1  # NRZ: -1/+1

        # Upsample
        upsampled = np.repeat(nrz, sps).astype(float)

        # Gaussian filter (BT=0.5)
        t_g  = np.arange(-3, 3, 1.0 / sps)
        BT   = 0.5
        h    = (np.sqrt(2 * np.pi) * BT / np.log(2)) * \
               np.exp(-2 * (np.pi * BT * t_g / np.log(2)) ** 2)
        h   /= h.sum()
        filtered = np.convolve(upsampled, h, mode="same")

        # FM modulation
        phase = 2 * np.pi * self.MOD_INDEX * np.cumsum(filtered) / sps
        signal = np.exp(1j * phase)
        return normalize(signal[:n_samples])


class ZigBeeGenerator:
    """
    ZigBee (IEEE 802.15.4) — O-QPSK with half-sine pulse shaping.
    Chip rate: 2 Mchip/s at 2.4 GHz.
    """
    CHIP_RATE = 2e6

    # 16 spreading chips per symbol (4 bits)
    CHIPS = np.array([
        [1,-1,-1,-1, 1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1,-1],
        [-1,-1,-1, 1,-1,-1, 1,-1,-1,-1, 1,-1, 1,-1,-1, 1],
        [-1, 1,-1,-1,-1,-1,-1, 1, 1,-1,-1,-1,-1, 1,-1,-1],
        [-1,-1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1],
        [ 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1,-1,-1, 1,-1,-1],
        [-1,-1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1,-1,-1, 1, 1],
        [1, -1,-1, 1,-1,-1,-1,-1,-1, 1,-1, 1,-1,-1,-1, 1],
        [-1, 1, 1,-1,-1,-1, 1,-1,-1,-1,-1, 1,-1,-1,-1, 1],
    ], dtype=float)

    def generate(self, n_samples: int, rng: np.random.Generator,
                 sample_rate: float = 20e6) -> np.ndarray:
        sps = int(sample_rate / self.CHIP_RATE)
        n_nibbles = n_samples // (16 * sps) + 2

        nibbles = rng.integers(0, 8, n_nibbles)
        chips   = self.CHIPS[nibbles % 8].flatten()

        # Half-sine pulse shaping
        t_p = np.arange(sps) / sps
        pulse = np.sin(np.pi * t_p)
        i_chips = chips[0::2]
        q_chips = chips[1::2]
        i_up = np.repeat(i_chips, sps)
        q_up = np.repeat(q_chips, sps)
        i_filt = np.convolve(i_up, pulse, mode="same")
        q_filt = np.convolve(q_up, pulse, mode="same")

        # O-QPSK: Q delayed by half chip
        signal = i_filt[:n_samples] + 1j * q_filt[:n_samples]
        return normalize(signal)


class LTEGenerator:
    """
    LTE-like CP-OFDM waveform (simplified).
    Resource block: 12 subcarriers × 7 symbols per slot.
    """
    N_FFT    = 128
    CP_LEN   = 10
    N_RB     = 6        # resource blocks (1.4 MHz BW)

    def generate(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        sym_len  = self.N_FFT + self.CP_LEN
        n_sym    = max(1, n_samples // sym_len)
        samples  = np.zeros(n_sym * sym_len, dtype=complex)
        n_data   = self.N_RB * 12

        for i in range(n_sym):
            freq = np.zeros(self.N_FFT, dtype=complex)
            # QPSK on active subcarriers (centered)
            qpsk = (2 * rng.integers(0, 2, n_data) - 1 +
                    1j * (2 * rng.integers(0, 2, n_data) - 1)) / np.sqrt(2)
            start = self.N_FFT // 2 - n_data // 2
            freq[start:start + n_data] = qpsk
            time = np.fft.ifft(freq) * np.sqrt(self.N_FFT)
            ofdm = np.concatenate([time[-self.CP_LEN:], time])
            samples[i * sym_len:(i + 1) * sym_len] = ofdm

        return normalize(samples[:n_samples])


class FiveGNRGenerator:
    """
    5G NR-like CP-OFDM with numerology μ=1 (SCS=30 kHz).
    """
    N_FFT    = 256
    CP_LEN   = 18

    def generate(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        sym_len = self.N_FFT + self.CP_LEN
        n_sym   = max(1, n_samples // sym_len)
        samples = np.zeros(n_sym * sym_len, dtype=complex)
        n_data  = 100   # active subcarriers

        for i in range(n_sym):
            freq = np.zeros(self.N_FFT, dtype=complex)
            # 16-QAM on active subcarriers
            qam16 = (2 * rng.integers(0, 4, n_data) - 3 +
                     1j * (2 * rng.integers(0, 4, n_data) - 3)) / np.sqrt(10)
            start = self.N_FFT // 2 - n_data // 2
            freq[start:start + n_data] = qam16
            time = np.fft.ifft(freq) * np.sqrt(self.N_FFT)
            ofdm = np.concatenate([time[-self.CP_LEN:], time])
            samples[i * sym_len:(i + 1) * sym_len] = ofdm

        return normalize(samples[:n_samples])


class RadarLFMGenerator:
    """
    Linear Frequency Modulated (LFM/chirp) pulse radar signal.
    """
    def __init__(self, pulse_width: float = 10e-6, bandwidth: float = 10e6,
                 prf: float = 5e3):
        self.pulse_width  = pulse_width   # seconds
        self.bandwidth    = bandwidth     # Hz (chirp sweep)
        self.prf          = prf           # pulse repetition frequency

    def generate(self, n_samples: int, rng: np.random.Generator,
                 sample_rate: float = 20e6) -> np.ndarray:
        samples = np.zeros(n_samples, dtype=complex)
        pulse_samp = int(self.pulse_width * sample_rate)
        pri_samp   = int(sample_rate / self.prf)

        t_pulse = np.arange(pulse_samp) / sample_rate
        chirp   = np.exp(1j * np.pi * self.bandwidth / self.pulse_width * t_pulse ** 2)

        pos = 0
        while pos + pulse_samp < n_samples:
            samples[pos:pos + pulse_samp] = chirp
            pos += pri_samp

        return normalize(samples)


# ── Unified generator interface ────────────────────────────────────────

class SignalDatasetGenerator:
    """
    Generate labeled (I/Q, class_idx) examples for all signal classes.
    """

    def __init__(self, cfg: GeneratorConfig = None):
        self.cfg = cfg or GeneratorConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

        self._generators: Dict[str, object] = {
            "wifi_g":    WiFiGenerator("g"),
            "wifi_n":    WiFiGenerator("n"),
            "bluetooth": BluetoothGenerator(),
            "zigbee":    ZigBeeGenerator(),
            "lte":       LTEGenerator(),
            "5g_nr":     FiveGNRGenerator(),
            "radar_lfm": RadarLFMGenerator(),
        }

    def generate_example(
        self,
        signal_class: str,
        snr_db: Optional[float] = None,
        cfo_hz: float = 0.0,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate one labeled I/Q example.

        Returns:
            iq:        Complex array [n_samples]
            class_idx: Integer class label
        """
        if signal_class not in self._generators:
            raise ValueError(f"Unknown signal class '{signal_class}'. "
                             f"Choose from: {list(self._generators)}")

        gen = self._generators[signal_class]
        snr = snr_db if snr_db is not None else self.cfg.snr_db

        # Generate clean signal
        if signal_class in ("bluetooth", "zigbee", "radar_lfm"):
            clean = gen.generate(self.cfg.n_samples, self.rng,
                                 sample_rate=self.cfg.sample_rate)
        else:
            clean = gen.generate(self.cfg.n_samples, self.rng)

        # Apply impairments
        cfo = cfo_hz or self.rng.uniform(-10e3, 10e3)
        noisy = apply_impairments(
            clean,
            snr_db=snr,
            cfo_hz=cfo,
            sample_rate=self.cfg.sample_rate,
            phase_noise_std=0.02,
            iq_imbalance_db=self.rng.uniform(0, 1.0),
            rng=self.rng,
        )

        return noisy, CLASS_TO_IDX[signal_class]

    def generate_batch(
        self,
        signal_classes: List[str],
        n_per_class: int,
        snr_range: Tuple[float, float] = (-10.0, 30.0),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a balanced dataset batch.

        Returns:
            X: [N, 2, n_samples] float32 (real/imag stacked)
            y: [N] int64 class labels
        """
        N_total = len(signal_classes) * n_per_class
        X = np.zeros((N_total, 2, self.cfg.n_samples), dtype=np.float32)
        y = np.zeros(N_total, dtype=np.int64)

        idx = 0
        for cls in signal_classes:
            for _ in range(n_per_class):
                snr = self.rng.uniform(*snr_range)
                iq, label = self.generate_example(cls, snr_db=snr)
                X[idx, 0] = iq.real.astype(np.float32)
                X[idx, 1] = iq.imag.astype(np.float32)
                y[idx]    = label
                idx += 1

        # Shuffle
        perm = self.rng.permutation(N_total)
        return X[perm], y[perm]
