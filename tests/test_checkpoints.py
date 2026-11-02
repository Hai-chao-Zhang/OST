"""Regression checks for finding and sharing OST model weights."""

import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from train import (
    EarlyStopping,
    checkpoint_directory,
    evaluation_checkpoint,
)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.stopping = EarlyStopping(logging.getLogger(__name__), verbose=False)
        self.source = nn.Linear(2, 1)
        with torch.no_grad():
            self.source.weight.copy_(torch.tensor([[1.25, -2.5]]))
            self.source.bias.fill_(0.75)

    def make_args(self, **overrides):
        values = dict(
            dataset="vifi", is_pos=False, model="VisionPosition",
            dec_model="Transformer", checkpoint_dir=None, checkpoint=None,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def assert_model_restored(self, target):
        probe = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
        torch.testing.assert_close(target(probe), self.source(probe))

    def test_default_paths_separate_datasets_and_select_phase_two(self):
        paths = []
        for dataset in ("vifi", "jrdb"):
            with self.subTest(dataset=dataset):
                args = self.make_args(dataset=dataset)
                directory = Path(checkpoint_directory(args))
                self.assertEqual(
                    directory,
                    Path("checkpoints") / dataset / "VisionPosition__Transformer",
                )
                self.assertEqual(
                    Path(evaluation_checkpoint(args)),
                    directory / "phase_2_model.pth",
                )
                paths.append(directory)
        self.assertNotEqual(*paths)

    def test_checkpoint_directory_override_keeps_legacy_paths_usable(self):
        args = self.make_args(
            checkpoint_dir="~/ost-weights/VisionPosition__Transformer",
        )
        expected = Path.home() / "ost-weights" / "VisionPosition__Transformer"
        self.assertEqual(Path(checkpoint_directory(args)), expected)
        self.assertEqual(
            Path(evaluation_checkpoint(args)), expected / "phase_2_model.pth",
        )

    def test_explicit_checkpoint_takes_priority_over_directory(self):
        args = self.make_args(
            checkpoint_dir=str(self.directory),
            checkpoint="~/ost-weights/custom-trained-model.pth",
        )
        self.assertEqual(
            evaluation_checkpoint(args),
            os.path.expanduser("~/ost-weights/custom-trained-model.pth"),
        )

    def test_missing_phase_two_explains_training_and_checkpoint_override(self):
        with self.assertRaises(FileNotFoundError) as error:
            self.stopping.load_checkpoint(self.source, str(self.directory), phase=2)
        message = str(error.exception)
        self.assertIn(str(self.directory / "phase_2_model.pth"), message)
        self.assertIn("train.py train", message)
        self.assertIn("phases 1 and 2", message)
        self.assertIn("--checkpoint", message)
        self.assertIn(".pkl files do not contain model weights", message)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_plain_and_legacy_weights_load_into_plain_and_parallel_models(self):
        for legacy in (False, True):
            saved_model = nn.DataParallel(self.source) if legacy else self.source
            checkpoint = self.directory / ("legacy.pth" if legacy else "plain.pth")
            torch.save(saved_model.state_dict(), checkpoint)
            for parallel_target in (False, True):
                with self.subTest(legacy=legacy, parallel_target=parallel_target):
                    target = nn.Linear(2, 1)
                    with torch.no_grad():
                        target.weight.zero_()
                        target.bias.zero_()
                    if parallel_target:
                        target = nn.DataParallel(target)
                    self.stopping.load_checkpoint(
                        target, str(self.directory), phase=2,
                        checkpoint=str(checkpoint),
                    )
                    self.assert_model_restored(target)

    def test_parallel_saves_are_portable_and_reload_from_default_phase_path(self):
        self.stopping.save_checkpoint(
            0.25, nn.DataParallel(self.source), str(self.directory), phase=2,
        )
        checkpoint = self.directory / "phase_2_model.pth"
        self.assertTrue(checkpoint.is_file())
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.assertEqual(set(state_dict), {"weight", "bias"})
        for key, value in self.source.state_dict().items():
            torch.testing.assert_close(state_dict[key], value)
        target = nn.Linear(2, 1)
        self.stopping.load_checkpoint(target, str(self.directory), phase=2)
        self.assert_model_restored(target)


if __name__ == "__main__":
    unittest.main()
