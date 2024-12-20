## Requirements
- PyTorch == 1.6.0
- TensorFlow == 1.15.0
- horovod == 0.19.5
- transformers == 3.0.2
- Additionally, see requirements.txt file
- The model was run on Python 3.11.4

## Notebooks
The modeling is mainly split up into two notebooks. The code made in the following notebooks builds on code in the example folder from the original code [REF]. Noticeable differences include the training step, hyperparameters, functions such as `ebnerd_from_path`, as well as the implementation for using the test set — such that we can get predictions on the hidden test set (in form of the predictions.txt file).

And of course, the model itself, stored in `nrmspy_1.py`.

### PyTorch Model Invocation
In the PyTorch implementation, the NRMS model is instantiated and wrapped in a custom `NRMSWrapper` class. Directories for logs and weights are created. A custom `PyTorchModelCheckpoint` class is defined to save the best model based on validation loss. The model is trained using a custom training loop within the `model.fit()` method, which provides greater control over the training process. After training, the best weights are loaded using `model.load_weights()`. Predictions are generated using the `model.predict()` method, which iterates over the validation data.

# Important code changes
The notebooks are found in `ebnerd-benchmark-copy/src/...`:
- [Notebook_Model__.ipynb](https://github.com/s204619/DeepL-Project/blob/114f3580d734a7f71576f8fc68ddef2ebfa9db94/ebnerd-benchmark-copy/src/NoteBook_Hyperparams_GOOD.ipynb) provides an overview of the model, its training, and the addition of features.
- [NoteBook_Hyperparams_GOOD.ipynb](https://github.com/s204619/DeepL-Project/blob/114f3580d734a7f71576f8fc68ddef2ebfa9db94/ebnerd-benchmark-copy/src/NoteBook_Hyperparams_GOOD.ipynb) does hyperparameter tuning.
- [Notebook_Model_testset.ipynb](https://github.com/s204619/DeepL-Project/blob/114f3580d734a7f71576f8fc68ddef2ebfa9db94/ebnerd-benchmark-copy/src/Notebook_Model_testset.ipynb) builds on Notebook_Model.ipynb, but includes the submission to the test set.

An alternative way to make the test set is shown in [Notebook_Alternative__testset__.ipynb](https://github.com/s204619/DeepL-Project/blob/114f3580d734a7f71576f8fc68ddef2ebfa9db94/ebnerd-benchmark-copy/src/Notebook_Alternative__testset__.ipynb). This is heavily inspired by the original code examples/reproducibility_scripts/[ebnerd_nrms.py](https://github.com/ebanalyse/ebnerd-benchmark/blob/main/examples/reproducibility_scripts/ebnerd_nrms.py).

Data has been downloaded from:
https://recsys.eb.dk/#leaderboard --> Download Dataset and is to be put into the `ebnerd_data` folder.

Many of the results were produced from `Notebook_MODEL.ipynb`.
