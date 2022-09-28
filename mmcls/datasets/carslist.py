import mmcv
import numpy as np
from .builder import DATASETS
# from .multi_label import MultiLabelDataset
from .base_dataset import BaseDataset

@DATASETS.register_module()
class CarsList(BaseDataset):
    CLASSES = ['sedan', 'suv', 'van', 'hatchback', 'mpv', 'pickup', 'bus', 'truck', 'estate']
    # CLASSES = ['yellow', 'orange', 'green', 'gray', 'red', 'blue', 'white', 'golden', 'brown', 'black']
    def load_annotations(self):
        assert isinstance(self.ann_file, str)

        data_infos = []
        with open(self.ann_file) as f:
            samples = [x.strip().split(' ') for x in f.readlines()]
            for sample in samples:
                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': sample[0]}
                info['gt_label'] = np.array(sample[1], dtype=np.int64)
                data_infos.append(info)
            return data_infos
