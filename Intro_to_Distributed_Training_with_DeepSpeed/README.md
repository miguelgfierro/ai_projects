# Introduction to Distributed Training with DeepSpeed

# Introduction

[DeepSpeed](https://github.com/microsoft/DeepSpeed) is an open-source library that facilitate the training of large deep learning models based on PyTorch. With minimal code changes, a developer can train a model on a single GPU machine, a single machine with multiple GPUs or on multiple machines in a distributed fashion. 

One of the advantages is that it enables massive models. When the library was first released, it was able to train a model of 200B parameters, by the end of 2021, the team was able to train [Megatron-Turing NLG 530B](https://www.microsoft.com/en-us/research/blog/using-deepspeed-and-megatron-to-train-megatron-turing-nlg-530b-the-worlds-largest-and-most-powerful-generative-language-model/), the largest generative language model to date. They are working to supporting soon a model of 1 trillion parameters.

The other important feature is its speed. According the [their experiments](https://www.deepspeed.ai/), DeepSpeed trains 2–7x faster than other solutions by reducing communication volume during distributed training. 

Last, but not least, the library only requires minimal code changes to use. In comparison to other distributed training libraries, DeepSpeed does not require a code redesign or model refactoring.

# Installation
