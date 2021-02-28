# river-semantic-segmentation-tf

## Informacje ogólne:

Repozytorium zawiera program pozwalający na trening modelów opartych na konwolucyjnych sieciach neuronowych służacych do segmentacji obszarów rzecznych na zdjęciach satelitarnych skomponowanych z pasm widzialnych RGB.


lp. | Model | Accuracy | IoU 
--- | --- | --- | ---
1 | Wejście | - | -
2 | Wzorowe wyjście | 1.0 | 1.0 
3 | UNET | 0.987783 | 0.872502
4 | RESNET50_UNET | 0.986252 | 0.858671
5 | VGG_SEGNET | 0.984189 | 0.83578 
6 | RESNET50_SEGNET | 0.982243 | 0.818463 
7 | UNET | 0.980608 | 0.801781 
8 | SEGNET | 0.977276 | 0.775857 

![results.png](https://i.postimg.cc/y890Vgkn/results.png)


## Użyte narzędzia:
- TensorFlow - framework ML
- OpenCV - biblioteka do przetwarzania obrazów
- NumPy - biblioteka do operacji na macierzach
- neptune - narzędzie logujące

## Uruchomienie:

1. Dataset do pobrania z oddzielnego repozytorium: https://github.com/shocik/sentinel-river-segmentation-dataset
2. Trenowanie odbywa się w pliku train_predict.ipynb. Należy w nim zmodyfikować:
	- ścieżkę do workdir:
	```python
	#set workdir
	os.chdir("/content/drive/MyDrive/RiverSemanticSegmentation/")
	```
	- ścieżkę do datasetu
	```python
	#dataset configuration
	dataset_dir = os.path.normpath("/content/drive/MyDrive/SemanticSegmentationV2/dataset/")
	```
