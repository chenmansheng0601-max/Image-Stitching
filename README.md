## A pytorch-reimplementation for [Unsupervised Deep Image Stitching via Semantic-Guided Alignment and Global-Local Reconstruction]
## Installation
[**Python>=3.6**](https://www.python.org/) is required with all
[requirements.txt](requirements.txt) installed including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/):

```bash
python3 -m pip install -r requirements.txt
```


## Data Preparation
Download the [UDIS-D](https://drive.google.com/drive/folders/1kC7KAULd5mZsqaWnY3-rSbQLaZ7LujTY?usp=sharing) and [WarpedCOCO](https://pan.baidu.com/s/1MVn1VFs_6-9dNRVnG684og) (code: 1234), and
make soft-links to the data directories:

```bash
ln -sf /path/to/UDIS-D UDIS-D
ln -sf /path/to/WarpedCOCO WarpedCOCO
```

Make sure the images are organized as follows:

```bash
UDIS-D/train/input1/000001.jpg  UDIS-D/train/input2/000001.jpg  UDIS-D/test/input1/000001.jpg  UDIS-D/test/input2/000001.jpg
WarpedCOCO/training/input1/000001.jpg  WarpedCOCO/training/input2/000001.jpg  WarpedCOCO/testing/input1/000001.jpg  WarpedCOCO/testing/input2/000001.jpg
```

## Training, Testing, and Inference
Run the commands below to go through the whole process of unsupervised deep image stitching. Some alternative commands are displayed in [main.sh](main.sh).

Download the pretrained backbones ([DINO](https://github.com/chenmansheng0601-max/Image-Stitching/releases/tag/v1.0), [YOLOv5x](https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5x.pt)) and put them to the `weights/` directory first. You can modify the `depth_multiple` and `width_multiple` in `models/*.yaml` to choose which backbone to use. 


