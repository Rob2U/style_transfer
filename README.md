# DL Projekt: Style Transfer

## Setup
### File Structure
```
.
├── data 
│   ╰── train2017
│       ╰── train2017
│           ├── 000000000009.jpg
│           ╽   ...
├── src 
│   ├── architecture.py         <- model architecture
│   ├── dataset.py              <- dataloading
│   ├── train.py                <- actual training
│   ├── trainer.py              <- specific training loop
│   ├── loss.py                 <- perceptual loss function
│   ├── config.py
│   ╰── utils.py                <- (TODO: the video stuff)
├── style_images                <- put all style images here
│   ├── style1.jpg
│   ╽   ...
├── test_images                 <- put all test images here
├── checkpoints                 <- create this directory (models will be saved here)
├── environment.yml             <- for conda (if desired)
├── demo.ipynb                  <- demonstration of our results
├── README.md
╽
```
### Setup
1. Download the [COCO Dataset](http://cocodataset.org/#download) and extract it into the `data` folder. (for `curl` use `curl http://images.cocodataset.org/zips/train2017.zip --output data/train2017.zip` and then `unzip data/train2017.zip -d train2017`, then the directory structure should be correct (if not just adjust the `DATA_DIR` in `src/config.py`))
2. 
- If you want to use the conda environment, run `conda env create -f environment.yml` and then `conda activate style-transfer` in order to activate it.
- NOTE: Depending on your OS, you may need to change the pytorch related packages and channels (see [here](https://pytorch.org/get-started/locally/) (channels are added in the command with `-c`))
- as it was not possible to install openCV with conda, it is installed with pip (see [here](https://pypi.org/project/opencv-python/))
- a list of all needed packages can be found below if anything goes wrong (or you just want to install them manually)

3. Specify the style and test images in `src/config.py`. (Check all other parameters as well (one may want to change the __len__ of the dataset to adjust the number of training images))
4. Run `python -m src.train` to start training.
5. The saved model can be found in `checkpoints` and the generated output images in `test_images`.
6. You can use `utils.py` to apply the model to your webcam stream (see `utils.py` for more information). Alternatively a very similar demo can be found in `demo.ipynb`.

## Exercise

Build a model, that takes a style image and mixes it with a content image. Demonstrate your results by creating a simple application, which takes a webcam frame and stylises it.

Paper: [https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf](https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf)

Dataset for content images: [http://cocodataset.org/#download](http://cocodataset.org/#download)

&nbsp;

## Packages
We used Python 3.10 and the following packages:
- torchaudio
- pytorch-cuda
- torchvision
- pytorch
- wandb
- pandas
- torchmetrics
- matplotlib
- tqdm
- numpy
- pillow
- datetime

