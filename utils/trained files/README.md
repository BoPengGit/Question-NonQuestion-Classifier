
# Pretrained Files
There are three pre-trained files that are required to run the `predict.py` using pretrained models and tokenizer.

* The first two files are the GRU.h5 file and LSTM.h5 file. These two files contain the two pretrained keras models. The pretrained [GRU.h5 file can be found here](https://storage.googleapis.com/question_nonquestion_classifier/cloud%20files/GRU.h5). The pretrained [LSTM.h5 file can be found here](https://storage.googleapis.com/question_nonquestion_classifier/cloud%20files/LSTM.h5).
* Both the 'GRU.h5' file and 'LSTM.h5' file needs to be downloaded and stored into the 'pretrained files' folder for the `predict.py` file to run using these two pretrained models.
* The tokenizer.pickle file is a tokenizer file trained on the full `train_2m.csv`. The train_2m.csv file contains around 2 million questions and 2 million non-question sentences. The tokenizer file was built using max number of tokens = 30,000.
