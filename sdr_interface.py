
Hardware abstraction layer for SDR capture and live RF classification.

Supported devices:
  - ADALM-Pluto (via pyadi-iio)
  - USRP (via uhd Python bindings)
  - File playback (SigMF / raw complex64)
  - Simulation (loopback for testing without hardware)

Usage:
    sdr = SDRInterface.from_config(cfg)
    sdr.configure(center_freq=2.4e9, sample_rate=20e6, rx_gain=50)
    for frame in sdr.stream(frame_size=1024):
        label, conf = classifier.predict(frame)
import numpy as np
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterator, Optional, Tuple


@dataclass
class SDRConfig:
    device:      str   = "simulation"   # "pluto" | "usrp" | "hackrf" | "simulation" | "file"
    center_freq: float = 2.4e9          # Hz
    sample_rate: float = 20e6           # Hz
    rx_gain:     float = 50.0           # dB
    bandwidth:   float = 20e6           # Hz
    file_path:   Optional[str] = None   # for file playback


class BaseSDR(ABC):
    """Abstract SDR interface."""

    @abstractmethod
    def configure(self, center_freq: float, sample_rate: float, rx_gain: float):
        ...

    @abstractmethod
    def receive(self, n_samples: int) -> np.ndarray:
        """Return n_samples complex64 I/Q samples."""
        ...

    def stream(
        self, frame_size: int = 1024, overlap: int = 0
    ) -> Iterator[np.ndarray]:
        """Continuously yield frames of I/Q samples."""
        stride = frame_size - overlap
        buffer = np.zeros(0, dtype=np.complex64)
        while True:
            chunk = self.receive(stride).astype(np.complex64)
            buffer = np.concatenate([buffer, chunk])
            while len(buffer) >= frame_size:
                yield buffer[:frame_size].copy()
                buffer = buffer[stride:]

    def close(self):
        pass


# ── Simulation (no hardware) ──────────────────────────────────────────

class SimulationSDR(BaseSDR):
    """
    Software loopback SDR for testing without hardware.
    Generates synthetic I/Q from the signal generator.
    """

    def __init__(self, cfg: SDRConfig, signal_class: str = "wifi_g", snr_db: float = 20.0):
        from src.data.signal_generator import SignalDatasetGenerator, GeneratorConfig
        self._gen = SignalDatasetGenerator(GeneratorConfig(
            sample_rate=cfg.sample_rate, seed=0
        ))
        self.signal_class = signal_class
        self.snr_db = snr_db
        self._cfg = cfg

    def configure(self, center_freq: float, sample_rate: float, rx_gain: float):
        self._cfg.center_freq = center_freq
        self._cfg.sample_rate = sample_rate
        print(f"[SimSDR] configured: fc={center_freq/1e6:.1f} MHz, "
              f"fs={sample_rate/1e6:.1f} MHz, gain={rx_gain} dB")

    def receive(self, n_samples: int) -> np.ndarray:
        self._gen.cfg.n_samples = n_samples
        iq, _ = self._gen.generate_example(self.signal_class, self.snr_db)
        time.sleep(n_samples / self._cfg.sample_rate * 0.1)  # simulate latency
        return iq.astype(np.complex64)


# ── File playback ──────────────────────────────────────────────────────

class FileSDR(BaseSDR):
    """Replay I/Q from a SigMF or raw complex64 file."""

    def __init__(self, file_path: str, loop: bool = True):
        path = Path(file_path)
        if path.suffix == ".sigmf-data":
            self._data = np.fromfile(file_path, dtype=np.complex64)
        elif path.suffix in (".dat", ".bin", ".raw"):
            self._data = np.fromfile(file_path, dtype=np.complex64)
        else:
            raise ValueError(f"Unknown file format: {path.suffix}")
        self._pos  = 0
        self._loop = loop
        print(f"[FileSDR] loaded {len(self._data):,} samples from {file_path}")

    def configure(self, center_freq: float, sample_rate: float, rx_gain: float):
        print(f"[FileSDR] (file playback — hardware config ignored)")

    def receive(self, n_samples: int) -> np.ndarray:
        end = self._pos + n_samples
        if end > len(self._data):
            if self._loop:
                self._pos = 0
                end = n_samples
            else:
                raise StopIteration("End of file")
        chunk = self._data[self._pos:end].copy()
        self._pos = end
        return chunk


# ── ADALM-Pluto ───────────────────────────────────────────────────────

class PlutoSDR(BaseSDR):
    """
    ADALM-Pluto interface via pyadi-iio.
    Install: pip install pyadi-iio
    """

    def __init__(self, uri: str = "ip:192.168.2.1"):
        try:
            import adi
            self._sdr = adi.Pluto(uri)
        except ImportError:
            raise ImportError("Install pyadi-iio: pip install pyadi-iio")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to PlutoSDR at {uri}: {e}")
        print(f"[PlutoSDR] connected to {uri}")

    def configure(self, center_freq: float, sample_rate: float, rx_gain: float):
        self._sdr.rx_lo             = int(center_freq)
        self._sdr.sample_rate       = int(sample_rate)
        self._sdr.rx_rf_bandwidth   = int(sample_rate)
        self._sdr.rx_hardwaregain_chan0 = rx_gain
        self._sdr.rx_buffer_size    = 4096
        print(f"[PlutoSDR] fc={center_freq/1e6:.3f} MHz  fs={sample_rate/1e6:.1f} MSPS  gain={rx_gain} dB")

    def receive(self, n_samples: int) -> np.ndarray:
        self._sdr.rx_buffer_size = n_samples
        raw = self._sdr.rx()
        iq = (raw / 2**11).astype(np.complex64)
        return iq


# ── USRP ──────────────────────────────────────────────────────────────

class USRPSDR(BaseSDR):
    """
    USRP B2xx/N2xx interface via UHD Python bindings.
    Install: via UHD system package + `pip install uhd`
    """

    def __init__(self, args: str = ""):
        try:
            import uhd
            self._usrp = uhd.usrp.MultiUSRP(args)
        except ImportError:
            raise ImportError("Install UHD and Python bindings")
        print(f"[USRP] connected: {self._usrp.get_mboard_name()}")

    def configure(self, center_freq: float, sample_rate: float, rx_gain: float):
        import uhd
        self._usrp.set_rx_rate(sample_rate)
        self._usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(center_freq))
        self._usrp.set_rx_gain(rx_gain)
        print(f"[USRP] fc={center_freq/1e6:.3f} MHz  fs={sample_rate/1e6:.1f} MSPS  gain={rx_gain} dB")

    def receive(self, n_samples: int) -> np.ndarray:
        import uhd
        streamer_args = uhd.usrp.StreamArgs("fc32", "sc16")
        rx_streamer = self._usrp.get_rx_stream(streamer_args)
        recv_buffer = np.zeros((1, n_samples), dtype=np.complex64)
        meta = uhd.types.RXMetadata()
        rx_streamer.recv(recv_buffer, meta)
        return recv_buffer[0]


# ── Factory ────────────────────────────────────────────────────────────

def create_sdr(cfg: SDRConfig, **kwargs) -> BaseSDR:
    """Instantiate the correct SDR class from config."""
    device = cfg.device.lower()
    if device == "simulation":
        return SimulationSDR(cfg, **kwargs)
    elif device == "file":
        return FileSDR(cfg.file_path)
    elif device == "pluto":
        uri = kwargs.get("uri", "ip:192.168.2.1")
        sdr = PlutoSDR(uri)
        sdr.configure(cfg.center_freq, cfg.sample_rate, cfg.rx_gain)
        return sdr
    elif device in ("usrp", "usrp_b200", "usrp_n200"):
        sdr = USRPSDR()
        sdr.configure(cfg.center_freq, cfg.sample_rate, cfg.rx_gain)
        return sdr
    else:
        raise ValueError(f"Unknown SDR device: {cfg.device}")
