import numpy
import mmcv
from mmcls.datasets import MultiLabelDataset

class CarsList(MultiLabelDataset):
    def load_annotations(self):
        """Load image paths and gt_labels."""
        if self.ann_file is None:
            samples = self._find_samples()
        elif isinstance(self.ann_file, str):
            lines = mmcv.list_from_file(
                self.ann_file, file_client_args=self.file_client_args)
            samples = [x.strip().rsplit(' ', 1) for x in lines]
        else:
            raise TypeError('ann_file must be a str or None')

        data_infos = []
        for filename, gt_label in samples:
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': filename}
            temp_label = np.zeros(len(self.CLASSES))
            
            if not self.multi_label:
                info['gt_label'] = np.array(gt_label, dtype=np.int64)
            else:
                ##multi-label classify
                if len(gt_label) == 1:
                    temp_label[np.array(gt_label, dtype=np.int64)] = 1
                    info['gt_label'] = temp_label
                else:
                    for i in range(np.array(gt_label.split(','), dtype=np.int64).shape[0]):
                        temp_label[np.array(gt_label.split(','), dtype=np.int64)[i]] = 1
                    info['gt_label'] = temp_label
            
            data_infos.append(info)
        return data_infos


# if __name__ == "__main__":
#     pass
