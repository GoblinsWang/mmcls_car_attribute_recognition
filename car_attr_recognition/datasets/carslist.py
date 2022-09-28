import mmcv
import numpy as np
from mmcls.datasets import DATASETS
from mmcls.datasets import MultiLabelDataset

@DATASETS.register_module()
class CarsList(MultiLabelDataset):
    def load_annotations(self):
        assert isinstance(self.ann_file, str)

        data_infos = []
        with open(self.ann_file) as f:
            samples = [x.strip().lsplit(' ') for x in f.readlines()]
            for sample in samples:
                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': sample[0]}
                info['gt_label'] = np.array(sample[1:], dtype=np.int64)
                data_infos.append(info)
            return data_infos


# if __name__ == "__main__":
#     pass
