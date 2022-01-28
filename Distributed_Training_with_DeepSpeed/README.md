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

## Implementation of DeepSpeed in a PyTorch model

One of the first [tutorials](https://github.com/microsoft/DeepSpeedExamples/tree/d1452c9d6c48f7586b1d1b734f89751c1585d25e/HelloDeepSpeed) that can be found in the repository explains how to create and train a Transformer encoder on the Masked Language Modeling (MLM) task. It also shows the code changes that need to be made to transform a PyTorch solution into a DeepSpeed one.

In the file [train_bert.py](train_bert.py) we can see how to train a Transformer encoder on the MLM task using standard PyTorch and in the file [train_bert_ds.py](train_bert_ds.py) we can see how to train the same model using DeepSpeed.

### Initialization

Replace the original PyTorch code:
```python
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```
with DeepSpeed alternative:

```python
ds_config = {
  "train_micro_batch_size_per_gpu": batch_size,
  "optimizer": {
      "type": "Adam",
      "params": {
          "lr": 1e-4
      }
  },
  "fp16": {
      "enabled": True
  },
  "zero_optimization": {
      "stage": 1,
      "offload_optimizer": {
         "device": "cpu"
      }
  }
}
model, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)
```

### Training

Replace the original PyTorch code:
```python
for step, batch in enumerate(data_iterator, start=start_step):
    loss.backward()
    optimizer.step()
```
with DeepSpeed alternative:
```python
for step, batch in enumerate(data_iterator, start=start_step):
    model.backward(loss)
    model.step()
```

### Model Checkpointing

Replace the original PyTorch code:
```python
if step % checkpoint_every != 0:
    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(obj=state_dict, f=str(exp_dir / f"checkpoint.iter_{step}.pt"))
```

with DeepSpeed alternative:
```python
if step % checkpoint_every != 0:
    model.save_checkpoint(save_dir=exp_dir, client_state={'checkpoint_step': step})
```

## Execution

In order to train the standar PyTorch model, and assuming we are on a machine with at least one GPU, we can run the following command:

```bash
python train_bert.py --checkpoint_dir experiments --num_iterations 1000 --local_rank 0 --log_every 500
```

For running the same model using DeepSpeed, we can run the following command. This command will take by default all the GPUs available on the machine.
```bash
deepspeed train_bert_ds.py --checkpoint_dir experiments --num_iterations 1000 --log_every 500
```

## References

* [DeepSpeed GitHub](https://github.com/microsoft/DeepSpeed)
* [DeepSpeed Examples](https://github.com/microsoft/DeepSpeedExamples)
* Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. (2020) DeepSpeed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters. [In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '20, Tutorial)](https://dl.acm.org/doi/10.1145/3394486.3406703).

