# river-semantic-segmentation-tf

## Informacje ogólne:

Repozytorium zawiera program pozwalający na trening modelów opartych na konwolucyjnych sieciach neuronowych służacych do segmentacji obszarów rzecznych na zdjęciach satelitarnych skomponowanych z pasm widzialnych RGB.

# Rezultaty

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
