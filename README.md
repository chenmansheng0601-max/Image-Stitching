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

Run the commands below to complete the whole pipeline of unsupervised deep image stitching. Some alternative commands are provided in [main.sh](main.sh).

Download the pretrained models:

- [DINO Backbone](https://github.com/chenmansheng0601-max/Image-Stitching/releases/download/v1.0/DINO_test.pth)
- [LeViT-UNet Reconstruction Model](https://github.com/chenmansheng0601-max/Image-Stitching/releases/download/v1.0/levit_test.pth)

and place them into the `weights/` directory before training or inference.

#### Step 1 (Alignment): Unsupervised pre-training on Stitched MS-COCO

```bash

python3 train.py --data data/warpedcoco.yaml --hyp data/hyp.align.scratch.yaml --cfg models/align_dino.yaml --weights weights/DINO_test.pth --batch-size 16 --img-size 128 --epochs 150 --adam --device 0 --mode align

mv runs/train/exp weights/align/warpedcoco
```

#### Step 2 (Alignment): Unsupervised finetuning on UDIS-D

```bash

python3 train.py --data data/udis.yaml --hyp data/hyp.align.finetune.udis.yaml --cfg models/align_dino.yaml --weights weights/align/warpedcoco/weights/best.pt --batch-size 16 --img-size 128 --epochs 50 --adam --device 3 --mode align
 
mv runs/train/exp weights/align/udis
```

#### Step 3 (Alignment): Evaluating and visualizing the alignment results

```bash
(RMSE) python3 inference_align.py --source data/warpedcoco.yaml --weights weights/align/warpedcoco/weights/best.pt --task val --rmse
(PSNR) python3 test.py --data data/warpedcoco.yaml --weights weights/align/warpedcoco/weights/best.pt --batch-size 64 --img-size 128 --task val --device 0 --mode align
(PSNR) python3 test.py --data data/udis.yaml --weights weights/align/udis/weights/best.pt --batch-size 64 --img-size 128 --task val --device 0 --mode align
(PLOT) python3 inference_align.py --source data/udis.yaml --weights weights/align/udis/weights/best.pt --task val --visualize
rm -r runs/infer/ runs/test/
```

#### Step 4 (Alignment): Generating the coarsely aligned image pairs

```bash
python3 inference_align.py --source data/udis.yaml --weights weights/align/udis/weights/best.pt --task train
python3 inference_align.py --source data/udis.yaml --weights weights/align/udis/weights/best.pt --task test
mkdir UDIS-D/warp
mv runs/infer/exp UDIS-D/warp/train
mv runs/infer/exp2 UDIS-D/warp/test
```

#### Step 5 (Reconstruction): Training the reconstrction model on UDIS-D

```bash

python3 train.py --data data/udis.yaml --hyp data/hyp.fuse.scratch.yaml --cfg models/fuse_yolo.yaml --weights weights/levit_test.pth --batch-size 4 --img-size 640 --epochs 30 --adam --device 0 --mode fuse --reg-mode crop

mv runs/train/exp weights/fuse/udis
```

#### Step 6 (Reconstruction): Generating the finally stitched results

```bash
python3 inference_fuse.py --weights weights/fuse/udis/weights/best.pt --source data/udis.yaml --task test --half --img-size 640 --reg-mode crop
```

