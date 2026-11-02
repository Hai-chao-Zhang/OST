"""Regression checks for using released pickles without raw RAN directories."""

import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from config.config import Config


class ProcessedDataConfigTests(unittest.TestCase):
    def setUp(self):
        original_cwd = Path.cwd()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.addCleanup(os.chdir, original_cwd)
        self.directory = Path(directory.name).resolve()
        os.chdir(self.directory)

    def make_config(self, dataset='vifi', data_file=None):
        return Config(SimpleNamespace(
            dataset=dataset, data_file=data_file, recent_K=50,
            is_proc_data=False,
        ))

    def test_defaults_work_outside_repo_without_creating_raw_directories(self):
        repo_root = Path(__file__).resolve().parents[1]
        for dataset, filename in (
            ('vifi', 'vifi_dataset_gps/vifi_data.pkl'),
            ('jrdb', 'JRDB/jrdb.pkl'),
        ):
            with self.subTest(dataset=dataset):
                config = self.make_config(dataset)
                self.assertEqual(Path(config.data_file), repo_root / filename)
                self.assertEqual(config.BBX5_dim, 4)
                self.assertEqual(config.IMU19_dim, 19)
                self.assertEqual(config.recent_K, 50)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_relative_override_uses_callers_directory(self):
        config = self.make_config(data_file='downloads/custom.pkl')
        self.assertEqual(
            Path(config.data_file), self.directory / 'downloads/custom.pkl',
        )
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_home_override_is_expanded(self):
        config = self.make_config('jrdb', '~/datasets/jrdb.pkl')
        self.assertEqual(Path(config.data_file), Path.home() / 'datasets/jrdb.pkl')


if __name__ == '__main__':
    unittest.main()
