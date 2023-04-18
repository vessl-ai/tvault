# `tvault` by VESSL

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://user-images.githubusercontent.com/97027715/232697803-3571bd58-8d4a-4c42-adba-96f300ef72c4.png" width="35%">
    <img alt="tvault" src="https://user-images.githubusercontent.com/97027715/232697811-f0a666a6-acbd-43a9-8af9-dea3e7cc0936.png" width="35%">
  </picture>
</p>

<p align="center">
    <a target="_blank" href="https://www.linkedin.com/company/vesslai"><img src="https://img.shields.io/badge/style--5eba00.svg?label=LinkedIn&logo=linkedin&style=social"></a>&nbsp;
    <a target="_blank" href="https://vesslai.medium.com/"><img src="https://img.shields.io/badge/style--5eba00.svg?label=Medium&logo=medium&style=social"></a>&nbsp;
    <a target="_blank" href="https://www.youtube.com/@vesslai4254"><img src="https://img.shields.io/badge/style--5eba00.svg?label=YouTube&logo=youtube&style=social"></a>&nbsp;
    <a target="_blank" href="https://join.slack.com/t/vessl-ai-community/shared_invite/zt-1a6schu04-NyjRKE0UMli58Z_lthBICA"><img src="https://img.shields.io/badge/Slack-Join-4A154B?logo=slack&style=social"></a>&nbsp;  
</p>

<h3 align="center">
    Compare and store Pytoch models in a local, lightweight registry
</h3>

----

## The challenges we are tackling

`tvault` is designed to help academic researchers iterate their models faster without the logging overhead. 

Many of the academic researchers we encounter simply want to *get going* with minimum setup and configurations. This often means using local codebase as opposed to integrating tools like [VESSL Experiments](https://docs.vessl.ai/api-reference/python-sdk/utils/vessl.log) and Weights & Biases to git-committed code. 

`tvault` is "git diff for ML" &mdash; a simple, lightweight framework for quickly tracking and comparing ML experiments in a local model registry. 

* Track and version models locally with `tvault.log_all()`
* Get a birds-eye differences of two experiments with `tvault --diff_flag`

<img alt="tvault-model_log" src="">

Get started with pip install:
```
pip install tvault
```

For those who are already using VESSL Python SDK and CLI,
```
pip install "vessl[tvault]"
```

Follow our guide below with our [MNIST example code](https://github.com/saeyoon17/mnist-tvault-example/blob/main/train.py). 

## Getting started with `tvault.log()`

Insert `tvault.log()` in your code's training loop with the metrics you want to track as tags.
https://github.com/vessl-ai/tvault/blob/4dd2f972a2c393b9ea3e70683c57b501a58c264f/setup.py#L5-L7

`tvault.log()` will automatically create the following:

* A folder `model_log` under your current directory - a tracking dashboard or model registry for your code
* A unique hash for the model and model ID for each training run with the key metrics

<img alt="tvault-model_log" src="">

## Look up experiments with `tvault --flag_flag`

`tvault`'s `find_flag` option allows you to look up different expereiments with simple cli. find_flag offers three different ways of exploring results:

  1. Search by hash
The command below shows all experiments with the hash value of  `2ba4adf`. 
```
tvault --find_flag --condition hash --hash 2ba4adf shows all experiments with hash 2ba4adf.
```
<img alt="tvault-model_log" src="">

  2. Search by result
The command below shows all experiments with the result between `50` and `100`.
```
tvault --find_flag --condition result --min 90 --max 100
```
<img alt="tvault-model_log" src="">

  3. Search by tags
The command below shows all experiments tagged as `0.5x`.
```
tvault --find_flag --condition tag --tag_type size --tag 0.5x
```
<img alt="tvault-model_log" src="">

## Compare models with `tvault --diff_flag`

`tvault`'s `diff_flag` option allows you to look up difference of two models by specifying model hash and index. tvault automatically detects functions that are realated to model, which removes git diffs that are not related to model.

This is useful when you have a baseline model that you want to iterate with different hyperparameters and higher-level architectures without digging through your code line-by-line. 
