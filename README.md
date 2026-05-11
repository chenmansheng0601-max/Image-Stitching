# Image-Stitching
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
