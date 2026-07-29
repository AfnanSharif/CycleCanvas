from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cyclegan import CycleGANConfig
from cyclegan.config import LinearDecay
from cyclegan.data import list_images
from cyclegan.preview import write_synthetic_domains
from cyclegan.comparison import architecture_profiles
from cyclegan.evaluation import test_split_counts


class CoreTests(unittest.TestCase):
    def test_config_validation(self) -> None:
        config = CycleGANConfig(epochs=20, decay_start_epoch=10, image_size=128)
        self.assertIs(config.validate(), config)
        with self.assertRaises(ValueError): CycleGANConfig(image_size=65).validate()

    def test_decay_schedule(self) -> None:
        schedule = LinearDecay(20, 10)
        self.assertEqual(schedule(5), 1.0)
        self.assertAlmostEqual(schedule(15), 0.5)
        self.assertEqual(schedule(20), 0.0)

    def test_image_discovery_filters_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "a.JPG").touch(); (root / "note.txt").touch()
            self.assertEqual([path.name for path in list_images(root)], ["a.JPG"])

    def test_synthetic_demo_is_reproducible_and_honest(self) -> None:
        with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
            first = write_synthetic_domains(one, seed=7, size=32)
            second = write_synthetic_domains(two, seed=7, size=32)
            self.assertEqual(first[0].read_bytes(), second[0].read_bytes())
            self.assertIn(b"NOT CycleGAN output", first[0].read_bytes())

    def test_train_and_test_splits_are_independent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "testA").mkdir(); (root / "testB").mkdir()
            (root / "testA" / "horse.png").touch(); (root / "testB" / "zebra.jpg").touch()
            self.assertEqual(test_split_counts(root / "testA", root / "testB"), {"testA": 1, "testB": 1})

    def test_resnet_vs_dense_profile_explains_learning_tradeoffs(self) -> None:
        dense, residual = architecture_profiles(32, hidden_features=128, dense_depth=4, residual_blocks=3, resnet_features=16)
        self.assertFalse(dense.spatial_inductive_bias)
        self.assertTrue(residual.spatial_inductive_bias)
        self.assertEqual(residual.skip_connections, 3)
        self.assertGreater(dense.parameter_estimate, residual.parameter_estimate)


if __name__ == "__main__": unittest.main()
