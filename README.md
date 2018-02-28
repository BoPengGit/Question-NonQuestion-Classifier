# Question-Non-Question-Classifier

This is a Question Non-Question classifier that inputs a csv file of sentences and outputs a prediction on whether each sentence is a question or not a question.

This model is trained using around 3 million question and around 3 million non-question sentences. This model uses a bidirectional GRU and a bidirectional LSTM. The predicted outputs are a simple average of the GRU and LSTM predicted probabilities. 

This model achieves over 99% accuracy tested on the same distribution as the training data. 

# Requirements
*  Python (tested with v3.6.3)
*  Numpy  (tested with v1.14.0)
*  Keras  (tested with v2.1.3)
 
 # Data
 * The data was trained using around 3 million questions from the [Quora Questions Pairs dataset](https://www.kaggle.com/c/quora-question-pairs/data). The non-question sentences are a conglomeration of two datasets, the [Stanford Natural Language Inference Corpus](https://www.kaggle.com/stanfordu/stanford-natural-language-inference-corpus/data) and the [Amazon Fine Food Reviews
dataset](https://www.kaggle.com/snap/amazon-fine-food-reviews/data).
* The training data is a .csv file with three columns: 'id','questions', 'non-questions'. A smaller subset of the training data is provided under the train.csv file in the data folder.
* To make new predictions using the pretrained model, the test input data should be a csv file with a column named 'sentences' that contains all of the sentences that the model is to classify.
* The model will output a new .csv file that will contain the original .csv file with a new column, 'prediction' with the predicted values for each sentence.


