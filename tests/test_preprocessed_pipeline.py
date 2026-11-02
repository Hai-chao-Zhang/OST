"""Exercise released loader schemas and model inputs with small CPU fixtures."""

from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from config.config import Config
from data.dataloader import Dataset_ViFi
from JRDB.dataloader import Dataset_JRDB
from model.vision_postion import VisionPosition


class PreprocessedPipelineTests(unittest.TestCase):
    def make_config(self, dataset, path):
        return Config(SimpleNamespace(
            dataset=dataset, data_file=str(path), recent_K=50,
            is_proc_data=False, mask_ratio=0.5, is_pos=False,
            dec_model='Transformer',
        ))

    def test_missing_pickle_explains_how_to_set_the_path(self):
        with tempfile.TemporaryDirectory() as directory:
            for dataset, loader in [('vifi', Dataset_ViFi), ('jrdb', Dataset_JRDB)]:
                with self.subTest(dataset=dataset):
                    config = self.make_config(dataset, Path(directory) / 'missing.pkl')
                    with self.assertRaisesRegex(FileNotFoundError, '--data_file'):
                        loader(config)

    def test_processed_pickles_feed_both_models_without_raw_directories(self):
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(1)
        for dataset, loader in [('vifi', Dataset_ViFi), ('jrdb', Dataset_JRDB)]:
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                data_file = Path(directory) / 'input.pkl'
                if dataset == 'vifi':
                    payload = {
                        'video': list(range(10)), 'timestamp': list(range(10)),
                        'bbx': [torch.ones(200, 3, 4) for _ in range(10)],
                        'imu': [torch.ones(200, 3, 19) for _ in range(10)],
                        'nearst_gps': [torch.ones(200, 3, 2) for _ in range(10)],
                        'interl_gps': [torch.ones(200, 3, 2) for _ in range(10)],
                    }
                else:
                    payload = (np.ones((10, 200, 3, 4)), np.ones((10, 200, 3, 3)))
                with data_file.open('wb') as stream:
                    pickle.dump(payload, stream)

                config = self.make_config(dataset, data_file)
                train_data = loader(config, flag='train')
                valid_data = loader(config, flag='valid')
                # Preserve the release's existing split; this is a setup repair.
                self.assertEqual((len(train_data), len(valid_data)), (2, 8))
                gps, nearest, imu, boxes, other_boxes, other_gps, other_nearest = (
                    value.float().unsqueeze(0) for value in train_data[0]
                )
                model = VisionPosition(config).eval()
                with torch.no_grad():
                    prediction, denoised = model(
                        gps[:, :100], nearest[:, :100], imu[:, :100], boxes,
                        other_boxes, other_gps.reshape(1, 200, -1),
                        other_nearest.reshape(1, 200, -1), imu,
                    )
                self.assertEqual(tuple(prediction.shape), (1, 200, 4))
                self.assertEqual(tuple(denoised.shape), (1, 100, 4))
                self.assertTrue(torch.isfinite(prediction).all())
                self.assertTrue(torch.isfinite(denoised).all())
                self.assertEqual(list(Path(directory).iterdir()), [data_file])


if __name__ == '__main__':
    unittest.main()
