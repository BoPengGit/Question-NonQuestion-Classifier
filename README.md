# Question-Non-Question-Classifier

This is a Question Non-Question classifier that inputs a csv file of sentences and outputs a prediction on whether each sentence is a question or not a question.

This model is trained using around 3 million questions and 3 million non question sentences. This model uses a bidirectional GRU and a bidirectional LSTM. The predicted outputs are a simple average of the GRU and LSTM predicted probabilities. 

This model achieves over 99% accuracy tested on the same distribution as the training data. 

# Requirements
*  


