# river-semantic-segmentation-tf

## Overview

The repository contains a code that allows training models based on convolutional neural networks for segmenting river areas in satellite images composed of RGB visible bands.

## Results


Id. | Model | Accuracy | IoU 
--- | --- | --- | ---
1 | Input | - | -
2 | Ground truth | 1.0 | 1.0 
3 | VGG_UNET | 0.987783 | 0.872502
4 | RESNET50_UNET | 0.986252 | 0.858671
5 | VGG_SEGNET | 0.984189 | 0.83578 
6 | RESNET50_SEGNET | 0.982243 | 0.818463 
7 | UNET | 0.980608 | 0.801781 
8 | SEGNET | 0.977276 | 0.775857 

The numbering of the rows in the table above corresponds to the labels on the visualization of the results for the sample image:
![results.png](https://i.postimg.cc/y890Vgkn/results.png)


## Tools used
- TensorFlow - ML framework
- OpenCV - a library for image processing
- NumPy - a library for matrix operations
- neptune - logging tool

## Dataset

Dataset available for download from a separate repository: https://github.com/shocik/sentinel-river-segmentation-dataset

## Running the code
Uruchomienie kodu na własnym komputerze wymaga wykonania następujących kroków przygotowujących:

1. Configuration of neptune in file [config.cfg](config.cfg).
2. Modify the path to the working folder in the file [train_predict.ipynb](train_predict.ipynb):
	```python
	#set workdir
	os.chdir("/content/drive/MyDrive/RiverSemanticSegmentation/")
	```
3. Modifying the path to a dataset in a file [train_predict.ipynb](train_predict.ipynb):
	```python
	#dataset configuration
	dataset_dir = os.path.normpath("/content/drive/MyDrive/SemanticSegmentationV2/dataset/")
	```
