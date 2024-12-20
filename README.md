## changes from Original code:
Ignored: Make a MIND folder and add teh data stuff to it :)






## Requirements
- PyTorch == 1.6.0
- TensorFlow == 1.15.0
- horovod == 0.19.5
- transformers == 3.0.2
- Additionally see requirements.txt file
- The model was run on Python 3.11.4


## Notebooks
The modelling is Mainly split up into two notebooks
The code made in the following notebooks builds on code in the example folder form the original code [REF]
Noticeable differences include the training step, hyperparamters, funcitons such as {ebnerd_from_path, 
As well as the implementation for using the test set — such that we can get predictions on te hidden test set (in form of  the precitions.txt file)



And of course the model itself, stored in nrmspy_1.py
PyTorch Model Invocation
In the PyTorch implementation, the NRMS model is instantiated and wrapped in a custom NRMSWrapper class. Directories for logs and weights are created. A custom PyTorchModelCheckpoint class is defined to save the best model based on validation loss. The model is trained using a custom training loop within the model.fit() method, which provides greater control over the training process. After training, the best weights are loaded using model.load_weights(). Predictions are generated using the model.predict() method, which iterates over the validation data.


- Notebook_Model.ipynb trains the model

- NoteBook_Hyperprarams.ipynb does hyperparameter tuning,

- Notebook_Model_WithSubmission.ipynb builds on Notebook_Model.ipynb, but includes the submission to the test set.

Data has been downloaded from:
https://recsys.eb.dk/#leaderboard --> Download Dataset and is to be put into ebnerd_data folder


Many of the results were produced from Notebook_MODEL.ipynb

