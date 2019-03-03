# Implementation of the paper "Very Deep Convolutional Networks for Natural 
# Language Processing", Conneau et al., 2016. 
# http://arxiv.org/abs/1606.01781

# Convolution block
convolution.block <- function(data, kernel, padding, residual, num_filter, act_type){
  conv <- mx.symbol.Convolution(data=data, kernel=kernel, pad=padding, num_filter=num_filter)
  norm <- mx.symbol.BatchNorm(data=conv)
  act <- mx.symbol.Activation(data=norm, act_type=act_type)
  
  if (residual){
    shortcut <- data
  } else{
    shortcut <- mx.symbol.Convolution(data=act, num_filter=num_filter, no_bias=TRUE,
                                      kernel=c(1,1), stride=c(1,1))
  }
  return(act + shortcut)
}

# Complete Network
get_symbol <- function(vocab.size, residual=FALSE, depth=9, num.output.classes=2){
  
  # Initial parameters
  num.filters1 <- 64
  num.filters2 <- num.filters1*2
  num.filters3 <- num.filters2*2
  num.filters4 <- num.filters3*2
  fully.connected.size <- 2048
  fully.connected.size2 <- fully.connected.size*2

  if(depth==9){
    units <- c(2,2,2,2)
  } else if(depth==17){
    units <- c(4,4,4,4)
  } else if(depth==29){
    units <- c(10,10,4,4)
  } else if(depth==49){
    units <- c(16,16,10,6)
  } else stop("Wrong depth")
  
  # Model
  data <- mx.symbol.Variable('data')

  # Changed embedding by convolution of vocabulary size 
  model <- mx.symbol.Convolution(data=data, num_filter=num.filters1, 
                                 pad=c(1,0), kernel=c(3, vocab.size))
  model <- mx.symbol.Activation(data=model, act_type="relu")
  
  # First convolution block of size num.filters1 
  for(i in 1:units[1]){
    model <- convolution.block(data = model, kernel=c(3,1), pad=c(1,0), residual=residual,
                               num_filter=num.filters1, act_type="relu")
  }
  
  # Pooling/2
  model <- mx.symbol.Pooling(data=model, pool_type="max", pad=c(1,0),
                             kernel=c(3,1), stride=c(2,1))
  
  # Second convolution block of size num.filters2
  for(i in 1:units[2]){
    if(i==1){
      #no residual in the first step to adapt the dimensions
      model <- convolution.block(data = model, kernel=c(3,1), pad=c(1,0), residual=FALSE,
                                  num_filter=num.filters2, act_type="relu") 
    } else {
      model <- convolution.block(data = model, kernel=c(3,1), pad=c(1,0), residual=residual,
                                 num_filter=num.filters2, act_type="relu")
    }
  }
  
  # Pooling/2
  model <- mx.symbol.Pooling(data=model, pool_type="max", pad=c(1,0),
                             kernel=c(3,1), stride=c(2,1))
  
  # Thrid convolution block of size num.filters3 
  for(i in 1:units[3]){
    if(i==1){
      model <- convolution.block(data = model, kernel=c(3,1), pad=c(1,0), residual=FALSE,
                                 num_filter=num.filters3, act_type="relu")
    } else {
      model <- convolution.block(data = model, kernel=c(3,1), pad=c(1,0), residual=residual,
                                 num_filter=num.filters3, act_type="relu")
    }
  }
  
  # Pooling/2
  model <- mx.symbol.Pooling(data=model, pool_type="max", pad=c(1,0),
                             kernel=c(3,1), stride=c(2,1))
  
  # Fourth convolution block of size num.filters4 
  for(i in 1:units[4]){
    if(i==1){
      model <- convolution.block(data = model, kernel=c(3,1), pad=c(1,0), residual=FALSE,
                                 num_filter=num.filters4, act_type="relu")
    } else {
      model <- convolution.block(data = model, kernel=c(3,1), pad=c(1,0), residual=residual,
                                 num_filter=num.filters4, act_type="relu")
    }
  }
  
  # Pooling (normal padding instead of k-max padding)
  model <- mx.symbol.Pooling(data=model, pool_type="max", pad=c(1,0),
                             kernel=c(8,1), stride=c(2,1))
  
  # First fully connected
  flatten <- mx.symbol.Flatten(data=model)
  fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=fully.connected.size2) 
  act_fc1 <- mx.symbol.Activation(data=fc1, act_type="relu")
  
  # Second fully connected
  fc2 <- mx.symbol.FullyConnected(data=flatten, num_hidden=fully.connected.size)
  act_fc1 <- mx.symbol.Activation(data=fc2, act_type="relu")
  
  # Third fully connected
  fc3 <- mx.symbol.FullyConnected(data=flatten, num_hidden=num.output.classes)
  network <- mx.symbol.SoftmaxOutput(data=fc3) # loss
  network
}