# Question Non-Question Classifier

This is a Question Non-Question classifier that inputs a csv file of sentences and outputs a prediction on whether each sentence is a question or not a question.

This model is trained using around 2 million questions and around 2 million non-question sentences. This model uses a bidirectional GRU and a bidirectional LSTM. The predicted outputs are a simple average of the GRU and LSTM predicted probabilities. 

This model achieves over 99% accuracy tested on unseen data from the same distribution as the training data. 

# Requirements
*  Python (tested with v3.6.3)
*  Numpy  (tested with v1.14.0)
*  Keras  (tested with v2.1.3)
 
 # Data
 * The data was trained using around 2 million questions from the [Quora Questions Pairs dataset](https://www.kaggle.com/c/quora-question-pairs/data). The around 2 million non-question sentences used are a conglomeration of two datasets, the [Stanford Natural Language Inference Corpus](https://www.kaggle.com/stanfordu/stanford-natural-language-inference-corpus/data) and the [Amazon Fine Food Reviews
dataset](https://www.kaggle.com/snap/amazon-fine-food-reviews/data).
* The training data is a .csv file with three columns: 'id','questions', 'non-questions'. A smaller subset of the training data is provided under the train.csv file in the data folder.
<br>

* To make new predictions using the pretrained model, the test input data should be a .csv file with a column named 'sentences' that contains all of the sentences that the model is to classify.
* The model will output a new .csv file that will contain the original .csv file with a new column, 'prediction' with the predicted values for each sentence.

# Usage
* To make question/non-question predictions on a test set, run `python predict.py data/test.csv`. This will use the default test set. This test set can be replaced with your own test set to make new predictions.
* To retrain the full model, run `python train.py data/train_2m.csv "data/word vectors/glove.6B.300d.txt" -es 300`. This will train the model using the full 'train_2m.csv' file that contains the 2 million questions and 2 million non-question sentences. This will also train the model using the GloVe 300d word embeddings. The 'train_2m.csv' file and the 'glove.6B.300d.txt' file will both need to be downloaded for the model to train. The provided 'train.csv' in the data folder file is a smaller subset of the original data. The pretrained files, 'GRU.h5', 'LSTM.h5', 'tokenizer.pickle', in the pretrained files folder were trained using the 'train_2m.csv' file with the 'glove.6B.300d.txt' word embeddings.

# Notes
This project uses a significant portion of code from these two resources, [PavelOstyakov's Toxic Comment Classification Challenge](https://github.com/PavelOstyakov/toxic) and [Jeremy Howard's Improved LSTM baseline: Glove + dropout kernel](https://www.kaggle.com/jhoward/improved-lstm-baseline-glove-dropout).

