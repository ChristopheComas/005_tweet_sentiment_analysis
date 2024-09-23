# Context
-----------
In this study, we explore different text classification approaches to determine the sentiment of reviews. We compare three models: a simple TF-IDF with logistic regression, a custom-built neural network with convolutional layers and LSTM, and the advanced BERT model. The goal is to assess which model provides the best balance of accuracy and complexity for deployment in a production environment, helping developers choose the optimal solution based on resource availability.

# Tech used
--------------
* Text classification models: TF-IDF with Logistic Regression, Conv1D Neural Network, biLSTM, BERT
* Model evaluation: accuracy, AUC score
* Model monitoring: Azure Application Insights
* CI / CD : github action (pytest)
* Interpretability and scalability:  MLflow
