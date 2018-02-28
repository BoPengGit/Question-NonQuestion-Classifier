# Question-Non-Question-Classifier

This is a Question Non-Question classifier that inputs a csv file of sentences and outputs a prediction on whether each sentence is a question or not a question.

This model is trained using around 3 million question and around 3 million non-question sentences. This model uses a bidirectional GRU and a bidirectional LSTM. The predicted outputs are a simple average of the GRU and LSTM predicted probabilities. 

This model achieves over 99% accuracy tested on the same distribution as the training data. 

# Requirements
*  Python (tested with v3.6.3)
*  Numpy  (tested with v1.14.0)
*  Keras  (tested with v2.1.3)
 
 # Data
 * The data was trained using around 3 million questions of the [Quora Questions Pairs dataset.](https://www.kaggle.com/c/quora-question-pairs/data). The non-questions are a conglomeration of two datasets, the [Stanford Natural Language Inference Corpus](https://www.kaggle.com/stanfordu/stanford-natural-language-inference-corpus/data) and the [Amazon Fine Food Reviews
datast](https://www.kaggle.com/snap/amazon-fine-food-reviews/data).


