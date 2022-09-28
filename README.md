## 概述

- 该项目主要是利用`openmmlab`的`mmcls`框架，对车辆属性进行识别，数据集为`VeRi`车辆数据集（里面标注有**车辆颜色**和**车型**）。

- 直接继承`mmcls`中的`mobilenet_v2`的配置文件，在此基础上修改`num_classes`和数据集等。

## 安装依赖

下面是快速安装的步骤:

```bash
conda create -n open-mmlab python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision -c pytorch -y
conda activate open-mmlab
pip3 install openmim
mim install mmcv-full
```

```bash
git clone https://github.com/GoblinsWang/mmcls_car_attribute_recognition.git
cd mmcls_car_attribute_recognition
pip install -v -e .
# "-v" 表示输出更多安装相关的信息
# "-e" 表示以可编辑形式安装，这样可以在不重新安装的情况下，让本地修改直接生效
```

## 主要改动

- 在`mmcls/datasets`中添加自定义的数据集类`carslist.py`：

```python
import mmcv
import numpy as np
from .builder import DATASETS
# from .multi_label import MultiLabelDataset
from .base_dataset import BaseDataset

@DATASETS.register_module()
class CarsList(BaseDataset):
    # CLASSES = ['sedan', 'suv', 'van', 'hatchback', 'mpv', 'pickup', 'bus', 'truck', 'estate']
    CLASSES = ['yellow', 'orange', 'green', 'gray', 'red', 'blue', 'white', 'golden', 'brown', 'black']
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
```

- 在`mmcls_car_attribute_recognition/car_attr_recognition/configs/mobilenet_v2_finetune.py`中通过以下代码注册上述模块：

```python
# 添加自定义类
custom_imports = dict(
    imports=['mmcls.datasets.filelist', 'mmcls.datasets.carslist'],
    allow_failed_imports=False)
```

## 使用示例

- 训练：

```bash
python tools/train.py car_attr_recognition/configs/mobilenet_v2_finetune.py
```

## 注意

若重新自定义一些模块类，比如`mmcls/datasets`添加相应文件之后，需要重新编译mmcls：

```bash
pip install -v -e .
```