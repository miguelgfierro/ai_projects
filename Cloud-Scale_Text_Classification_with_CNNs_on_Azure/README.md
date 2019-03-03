# Text Classification

Here we show some text classification examples with Recurrent Neural Networks and Convolutional Neural Networks.


## Bidirectional LSTM using Keras

This example shows how to train a Bi-LSTM in the IMDB database for sentiment classification 
[source](https://github.com/fchollet/keras/blob/d73c8361725e550f83a36cdf322e40a695db3a84/examples/imdb_bidirectional_lstm.py):
    
	cd python/keras  
	python bilstm_imdb.py  

## CNN at character level using MXNet

Implementation of the papers ["Character-level Convolutional Networks for Text Classification", Zhang et al., 2016](http://arxiv.org/abs/1509.01626) and  ["Very Deep Convolutional Networks for Natural Language Processing", Conneau et al., 2016](http://arxiv.org/abs/1606.01781). The authors present an architecture for text processing which operates directly on the character level and uses only small convolutions and pooling operations. The authors claim that it is the ﬁrst time that very deep convolutional nets have been applied to NLP. They surpass the state of the art accuracy in several public databases. 

We are going to use the dataset of Amazon categories. This dataset consists of a training set of 2.38 million sentences, a test set of 420.000 sentences, divided in 7 categories: “Books”, “Clothing, Shoes & Jewelry”, “Electronics”, “Health & Personal Care”, “Home & Kitchen”, “Movies & TV” and “Sports & Outdoors”. 
	
	cd data
	python download_amazon_categories.py  

To run the code in R, with VDCNN network of 9 layers, with 4 GPUs, a batch size of 128 in each GPU, learning rate 0.01, learning rate scheduler with factor 0.94 during 10 epochs:

	cd R/mxnet
	Rscript text_classification_cnn.R --network vdcnn --depth 9 --batch-size 512 --lr 0.01 --lr-factor .94 --gpus 0,1,2,3 --train-dataset categories_train_big.csv --val-dataset categories_test_big.csv --num-examples 2379999 --num-classes 7 --num-round 10 --log-dir $PWD --log-file vdcnn.log --model-prefix vdcnn 

We also provide another version in python. This [jupyter notebook](python/mxnet/crepe_dbpedia_prefetch.ipynb) uses the crepe model with DBpedia dataset to classify among 14 different classes. 

