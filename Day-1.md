# Day1: The Ultra-Scale Playbook: Training LLMs on GPU Clusters - Introduction

## Intro

How to get GPUs get burr?

This open-source book tries to let it open.

Various technique, such as data parallelism, tensor parallelism, pipeline parallelism and context parallelism as well as Zero and kernel functions.

Try to maintain a single storyline to help the developer understand where each method comes from.

Check `Deeplearning.ai` or `Pytorch tutorial` for basics.

Check `FineWeb blog post` on processing data for pre-training.

This book is built on the following three general foundations:

1. Quick intro on theory and concept

2. Clear code implementation 

     - [Picotron repo](https://github.com/huggingface/picotron)

     - [Nanotron](https://github.com/huggingface/nanotron)

     - [Ferdinand’s’ YouTube channel](https://www.youtube.com/watch?v=u2VSwDDpaBM)

3. Real training efficiency benchmarks 

The authors ran over 4,000 scaling experiments on up to 512 GPUs and measured throughput (size of markers) and GPU utilization (color of markers). Note that both are normalized per model size in this visualization.

### High-level overview

1. Memory usage

2. Compute efficiency

3. Communication overheads



### Cheatsheet

![ultra-cheatsheet.svg](images/ultra-cheatsheet.svg)

## Code Snippet

```python
import torch; 
torch.ones((1, 1)).to("cuda")
```

```bash
### and then checking the GPU memory with .
!nvidia-smi
```

## Suggested Readings

<https://www.deeplearning.ai/> 

<https://pytorch.org/tutorials/beginner/basics/intro.html> 

<https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1>

<https://huggingface.co/spaces/nanotron/predict_memory>

<https://github.com/huggingface/picotron>

<https://github.com/huggingface/nanotron>

<https://www.youtube.com/watch?v=u2VSwDDpaBM&list=PL-_armZiJvAnhcRr6y> 