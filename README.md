# PackFormer

Packformer contains four datasets, each with a separate code.

## Installation

Make sure you have Python 3.7 installed.

Install the required dependencies:

```
$ pip install -r requirements.txt
```

## NNI
For any of the datasets, there is an NNI tuning code that provides quick access to the optimal parameters of the Packformer model.
```
$ nnictl create --config config.yml
```
The results of the call are saved in the `/params_turner` 
folder

## Quick Start
If you want to try a quick training of all models, you can run the auto_train1.py file in each dataset folder.
```
$ python the_dataset_name/auto_train1.py
```
The trained models are saved in `/RESU_decoder_20_4`, `/MIT_decoder_20_4`, `/Oxford_20_4`, `/NASA_decoder3_attn0.5`.

## Results
The results of the various baseline and Packformer methods proposed in this paper can be obtained by running the `/plot.py` file under each dataset folder.
```
$ python the_dataset_name/plot.py
```
The results are stored with each dataset in the `/plot_data` folder.
## Ablation
The training code in Packformer without using the cell attention is as follows:
```
$ python the_dataset_name/Ablation.py
```
The trained models are saved in `/ablation` in each dataset. Then the results of ablation experiment is get by the following code:
```
$ python the_dataset_name/plot_ablation.py
```
## Plot
The results used in the thesis can be plotted by using the code under the `/figure` folder.