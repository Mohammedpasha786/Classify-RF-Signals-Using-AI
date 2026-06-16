Unit tests for signal generation, models, and evaluation.
import numpy as np
import pytest
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.signal_generator import (
    SignalDatasetGenerator, GeneratorConfig,
    WiFiGenerator, BluetoothGenerator, ZigBeeGenerator,
    LTEGenerator, FiveGNRGenerator, RadarLFMGenerator,
    SIGNAL_CLASSES, CLASS_TO_IDX, apply_impairments, normalize,
)
from src.models.classifier import CNN1D, ResNet1D, CLDNN, build_model
from src.evaluation.evaluation import (
    confusion_matrix, per_class_metrics, compute_accuracy
)

RNG = np.random.default_rng(42)
N_IQ = 512


# ── Signal generators ──────────────────────────────────────────────────

class TestSignalGenerators:

    @pytest.mark.parametrize("GenClass,kwargs", [
        (WiFiGenerator,      {"variant": "g"}),
        (BluetoothGenerator, {}),
        (ZigBeeGenerator,    {}),
        (LTEGenerator,       {}),
        (FiveGNRGenerator,   {}),
        (RadarLFMGenerator,  {}),
    ])
    def test_output_length(self, GenClass, kwargs):
        gen = GenClass(**kwargs)
        if hasattr(gen, 'generate'):
            try:
                iq = gen.generate(N_IQ, RNG, sample_rate=20e6)
            except TypeError:
                iq = gen.generate(N_IQ, RNG)
        assert len(iq) == N_IQ

    @pytest.mark.parametrize("GenClass,kwargs", [
        (WiFiGenerator,      {"variant": "g"}),
        (BluetoothGenerator, {}),
    ])
    def test_unit_power(self, GenClass, kwargs):
        gen = GenClass(**kwargs)
        try:
            iq = gen.generate(N_IQ, RNG, sample_rate=20e6)
        except TypeError:
            iq = gen.generate(N_IQ, RNG)
        rms = np.sqrt(np.mean(np.abs(iq)**2))
        assert 0.9 < rms < 1.1

    def test_complex_output(self):
        gen = WiFiGenerator()
        iq = gen.generate(N_IQ, RNG)
        assert np.iscomplexobj(iq)

    def test_all_classes_generate(self):
        cfg = GeneratorConfig(n_samples=N_IQ, seed=0)
        dataset_gen = SignalDatasetGenerator(cfg)
        for cls in SIGNAL_CLASSES:
            iq, label = dataset_gen.generate_example(cls, snr_db=20.0)
            assert len(iq) == N_IQ
            assert label == CLASS_TO_IDX[cls]

    def test_invalid_class_raises(self):
        cfg = GeneratorConfig(n_samples=N_IQ, seed=0)
        gen = SignalDatasetGenerator(cfg)
        with pytest.raises(ValueError):
            gen.generate_example("5g_wifi_radar_fusion", snr_db=10.0)

    def test_batch_shape(self):
        cfg = GeneratorConfig(n_samples=N_IQ, seed=0)
        gen = SignalDatasetGenerator(cfg)
        signals = ["wifi_g", "bluetooth", "zigbee"]
        X, y = gen.generate_batch(signals, n_per_class=10)
        assert X.shape == (30, 2, N_IQ)
        assert y.shape == (30,)
        assert set(y.tolist()).issubset({CLASS_TO_IDX[s] for s in signals})

    def test_snr_affects_noise_level(self):
        gen = WiFiGenerator()
        clean = normalize(gen.generate(1000, RNG))
        noisy_lo  = apply_impairments(clean.copy(), snr_db=0.0,  rng=RNG)
        noisy_hi  = apply_impairments(clean.copy(), snr_db=30.0, rng=RNG)
        noise_lo  = np.std(np.abs(noisy_lo  - clean))
        noise_hi  = np.std(np.abs(noisy_hi  - clean))
        assert noise_lo > noise_hi


# ── Models ─────────────────────────────────────────────────────────────

class TestCNN1D:
    B, N_CLASSES = 4, 7

    def test_forward_shape(self):
        model = CNN1D(n_classes=self.N_CLASSES)
        x = torch.randn(self.B, 2, N_IQ)
        out = model(x)
        assert out.shape == (self.B, self.N_CLASSES)

    def test_param_count(self):
        model = CNN1D(n_classes=7)
        assert model.get_num_params() < 500_000

    def test_output_no_nan(self):
        model = CNN1D(n_classes=7)
        x = torch.randn(2, 2, N_IQ)
        out = model(x)
        assert not torch.isnan(out).any()


class TestResNet1D:
    B, N_CLASSES = 4, 7

    def test_forward_shape(self):
        model = ResNet1D(n_classes=self.N_CLASSES)
        x = torch.randn(self.B, 2, N_IQ)
        out = model(x)
        assert out.shape == (self.B, self.N_CLASSES)

    def test_embedding_shape(self):
        model = ResNet1D(n_classes=7)
        x = torch.randn(2, 2, N_IQ)
        emb = model.get_embedding(x)
        assert emb.shape == (2, 512)

    def test_freeze_backbone(self):
        model = ResNet1D(n_classes=7)
        model.freeze_backbone("layer2")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = model.get_num_params(trainable_only=False)
        assert trainable < total

    def test_replace_classifier(self):
        model = ResNet1D(n_classes=7)
        model.replace_classifier(3)
        x = torch.randn(2, 2, N_IQ)
        out = model(x)
        assert out.shape == (2, 3)

    def test_transfer_learning_flow(self):
        """Full transfer learning: load pretrained, freeze, replace head, run forward."""
        model = ResNet1D(n_classes=7)
        model.freeze_backbone("layer2")
        model.replace_classifier(3)   # new task: 3 classes
        x = torch.randn(2, 2, N_IQ)
        out = model(x)
        assert out.shape == (2, 3)


class TestCLDNN:
    def test_forward_shape(self):
        model = CLDNN(n_classes=7)
        x = torch.randn(4, 2, N_IQ)
        out = model(x)
        assert out.shape == (4, 7)

    def test_param_count(self):
        model = CLDNN(n_classes=7)
        assert model.get_num_params() < 2_000_000


class TestBuildModel:
    @pytest.mark.parametrize("name", ["cnn1d", "resnet1d", "cldnn"])
    def test_builds_all_architectures(self, name):
        model = build_model(name, n_classes=5)
        x = torch.randn(2, 2, N_IQ)
        out = model(x)
        assert out.shape == (2, 5)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError):
            build_model("transformer_xl_v3", n_classes=7)


# ── Evaluation ─────────────────────────────────────────────────────────

class TestEvaluation:

    def _make_loader(self, n_classes: int = 4, n_samples: int = 32):
        from torch.utils.data import TensorDataset, DataLoader
        X = torch.randn(n_samples, 2, N_IQ)
        y = torch.randint(0, n_classes, (n_samples,))
        return DataLoader(TensorDataset(X, y), batch_size=8)

    def test_confusion_matrix_shape(self):
        model = CNN1D(n_classes=4)
        loader = self._make_loader(n_classes=4)
        cm = confusion_matrix(model, loader, torch.device("cpu"), n_classes=4)
        assert cm.shape == (4, 4)
        assert cm.sum() == 32

    def test_confusion_matrix_nonneg(self):
        model = CNN1D(n_classes=4)
        loader = self._make_loader(n_classes=4)
        cm = confusion_matrix(model, loader, torch.device("cpu"), n_classes=4)
        assert np.all(cm >= 0)

    def test_per_class_metrics_keys(self):
        cm = np.array([[10, 2], [1, 12]])
        metrics = per_class_metrics(cm, class_names=["A", "B"])
        assert "per_class" in metrics
        assert "overall_accuracy" in metrics
        assert len(metrics["per_class"]) == 2

    def test_overall_accuracy_range(self):
        cm = np.array([[10, 2], [1, 12]])
        metrics = per_class_metrics(cm)
        assert 0.0 <= metrics["overall_accuracy"] <= 1.0

    def test_compute_accuracy_returns_float(self):
        model = CNN1D(n_classes=4)
        loader = self._make_loader(n_classes=4)
        result = compute_accuracy(model, loader, torch.device("cpu"))
        assert 0.0 <= result["accuracy"] <= 1.0
        assert result["n_samples"] == 32
