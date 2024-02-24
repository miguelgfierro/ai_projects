[![Issues](https://img.shields.io/github/issues/miguelgfierro/ai_projects.svg)](https://github.com/miguelgfierro/ai_projects/issues)
[![Pull requests](https://img.shields.io/github/issues-pr/miguelgfierro/ai_projects.svg)](https://github.com/miguelgfierro/ai_projects/pulls)
[![Commits](https://img.shields.io/github/commit-activity/y/miguelgfierro/ai_projects.svg?color=success)](https://github.com/miguelgfierro/ai_projects/commits/master)
[![Last commit](https://img.shields.io/github/last-commit/miguelgfierro/ai_projects.svg)](https://github.com/miguelgfierro/ai_projects/commits/master)

[![Linkedin](https://img.shields.io/badge/Linkedin-Follow%20Miguel-blue?logo=linkedin)](https://www.linkedin.com/comm/mynetwork/discovery-see-all?usecase=PEOPLE_FOLLOWS&followMember=miguelgfierro)
[![Blog](https://img.shields.io/badge/Blog-Visit%20miguelgfierro.com-blue.svg)](https://miguelgfierro.com?utm_source=github&utm_medium=profile&utm_campaign=ai_projects)


# Sciblog support information and code
This repo contains the projects, additional information and code to support my blog: [sciblog](https://miguelgfierro.com/).

You can find a list of all the post I made in [this file](miguelgfierro_posts.txt).

## Notebook projects

* [Introduction to Convolutional Neural Networks](A_Gentle_Introduction_to_CNN/Intro_CNN.ipynb): In this project we explain what is a convolution and how to compute a CNN using MXNet deep learning library with the MNIST character recognition dataset. Here the [blog entry](https://miguelgfierro.com/blog/2016/a-gentle-introduction-to-convolutional-neural-networks/?utm_source=github&utm_medium=repo-entry&utm_campaign=cnn-intro).

* [Introduction to Transfer Learning](A_Gentle_Introduction_to_Transfer_Learning/Intro_Transfer_Learning.ipynb): In this project we use PyTorch to explain the basic methodologies of transfer learning (finetuning and freezing) and analyze in which case is better to use each of them. Here the [blog entry](https://miguelgfierro.com/blog/2017/a-gentle-introduction-to-transfer-learning-for-image-classification/?utm_source=github&utm_medium=repo-entry&utm_campaign=transfer-learning).

* [Cloud-Scale Text Classification With Convolutional Neural Networks](Cloud-Scale_Text_Classification_with_CNNs_on_Azure): In these notebooks we show how to perform character level convolutions for sentiment analysis using Char-CNN and VDCNN models. Here the [blog entry](https://miguelgfierro.com/blog/2019/cloud-scale-text-classification-with-convolutional-neural-networks/?utm_source=github&utm_medium=repo-entry&utm_campaign=charcnn).

* [Introduction to Data Generation](Data_Generation/data_generation.ipynb): In this notebook we show a number of simple techniques to generate new data in images, text and time series. Here the [blog entry](https://miguelgfierro.com/blog/2019/revisiting-the-revisit-of-the-unreasonable-effectiveness-of-data/?utm_source=github&utm_medium=repo-entry&utm_campaign=data-gen).

* [Introduction to Dimensionality Reduction with t-SNE](Dimensionality_Reduction_with_TSNE/dimensionality_reduction.ipynb): In this project we use sklearn and CUDA to show an example of t-SNE algorithm. We use a CNN to generate high-dimensional features from images and then show how they can be projected and visualized into a 2-dimensional space. Here the [blog entry](https://miguelgfierro.com/blog/2018/a-gentle-explanation-of-dimensionality-reduction-with-t-sne/?utm_source=github&utm_medium=repo-entry&utm_campaign=tsne).

* [Introduction to Distributed Training with DeepSpeed](Distributed_Training_with_DeepSpeed): In this project we show how to use DeepSpeed to perform distributed training with PyTorch. Here the [blog entry](https://miguelgfierro.com/blog/2022/a-gentle-introduction-to-distributed-training-with-deepspeed/?utm_source=github&utm_medium=repo-entry&utm_campaign=deepspeed).

* [Introduction to Fraud Detection](Intro_to_Fraud_Detection/fraud_detection.ipynb): In this notebook we design a real-time fraud detection model using LightGBM on GPU (also available on CPU). The model is then operationalized through an API using Flask and websockets. Here the [blog entry](https://github.com/miguelgfierro/ai_projects/blob/master/Intro_to_Fraud_Detection/fraud_detection.ipynb?utm_source=github&utm_medium=repo-entry&utm_campaign=fraud).

* [Introduction to Machine Learning API](Intro_to_Machine_Learning_API/Intro_to_Cloud_ML_with_Flask_and_CNTK.ipynb): In this notebook we show how to create an image classification API. The system works with a pretrained CNN using CNTK deep learning library. The API is setup with Flask for managing the end point services and CherryPy as the backend server. Here the [blog entry](https://miguelgfierro.com/blog/2017/how-to-deploy-an-image-classification-api-based-on-deep-learning/?utm_source=github&utm_medium=repo-entry&utm_campaign=ml-api).

* [Introduction to Recommendation Systems with Deep Autoencoders](Intro_to_Recommendation_Systems/Intro_Recommender.ipynb): In this notebook we make an overview to recommendation systems and implement a recommendation API using a deep autoencoder with PyTorch and the Netflix dataset. Here the [blog entry](https://miguelgfierro.com/blog/2018/introduction-to-recommendation-systems-with-deep-autoencoders/?utm_source=github&utm_medium=repo-entry&utm_campaign=reco-deep-autoencoder).

* [Introduction to Natural Language Processing with fastText](Intro_to_NLP_with_fastText/Intro_to_NLP.ipynb): In this project we show how to implement text classification, sentiment analysis and word embedding using the library fastText. We also show a way to represent the word embeddings in a reduced space using t-SNE algorithm. Here the [blog entry](https://miguelgfierro.com/blog/2017/a-gentle-introduction-to-text-classification-and-sentiment-analysis/?utm_source=github&utm_medium=repo-entry&utm_campaign=fasttext).

* [Time Series Forecasting of Stock Price](Time_Series_Forecasting_of_Stock_Price/Stock_Price_Forecasting.ipynb): In this tutorial we show how to implement a simple stock forecasting model using different variants of LSTMs and Keras. Here the [blog entry](https://miguelgfierro.com/blog/2018/stock-price-prediction-with-lstms/?utm_source=github&utm_medium=repo-entry&utm_campaign=stock-forecasting).

* [Visualization of Football Matches with Datashader](Visualization_of_Football_Matches/visualization_football.ipynb): In this notebook we explain how to visualize all matches in the UEFA Champions League since its beginning using the python library datashader. To create the project we use the Lean Startup method. Here the [blog entry](https://miguelgfierro.com/blog/2016/how-to-develop-a-data-science-project-using-the-lean-startup-method/?utm_source=github&utm_medium=repo-entry&utm_campaign=datashader).
