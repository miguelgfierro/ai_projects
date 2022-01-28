# Introduction to Distributed Training with DeepSpeed

## Introduction

[DeepSpeed](https://github.com/microsoft/DeepSpeed) is an open-source library that facilitate the training of large deep learning models based on PyTorch. With minimal code changes, a developer can train a model on a single GPU machine, a single machine with multiple GPUs or on multiple machines in a distributed fashion. 

One of the advantages is that it enables massive models. When the library was first released, it was able to train a model of 200B parameters, by the end of 2021, the team was able to train [Megatron-Turing NLG 530B](https://www.microsoft.com/en-us/research/blog/using-deepspeed-and-megatron-to-train-megatron-turing-nlg-530b-the-worlds-largest-and-most-powerful-generative-language-model/), the largest generative language model to date. They are working to supporting soon a model of 1 trillion parameters.

The other important feature is its speed. According the [their experiments](https://www.deepspeed.ai/), DeepSpeed trains 2–7x faster than other solutions by reducing communication volume during distributed training. 

Last, but not least, the library only requires minimal code changes to use. In comparison to other distributed training libraries, DeepSpeed does not require a code redesign or model refactoring.

## Installation

The installation is very simple, for a basic test of the library we can install DeepSpeed, PyTorch and Transformers.

```bash
conda create -n deepspeed python=3.7 -y
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip install deepspeed transformers datasets fire loguru sh pytz
```

The versions installed are from the [requirements.txt](requirements.txt) file.

```python
print(f"Numpy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"DeepSpeed version: {deepspeed.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"Datasets version: {datasets.__version__}")
```
```bash
Numpy version: 1.21.2
PyTorch version: 1.10.2
DeepSpeed version: 0.5.10
Transformers version: 4.16.0
Datasets version: 1.18.1
```

