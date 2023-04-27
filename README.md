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
    Quickly compare PyTorch models in a local, lightweight registry
</h3>

----

`tvault` is designed to help academic researchers iterate their models faster without logging overheads. 

Many of the academic researchers we encounter simply want to *get going* with minimum setup and configurations. This often means using local codebase as opposed to integrating tools like [VESSL Experiments](https://docs.vessl.ai/api-reference/python-sdk/utils/vessl.log) and Weights & Biases to git-committed codes. 

`tvault` is "git diff for machine learning" &mdash; a simple, lightweight framework for quickly tracking and comparing ML experiments in a local model registry. 

* Track and version models locally with `tvault.log_all()`
* Get a birds-eye differences of two experiments with `tvault --diff_flag`

<img alt="tvault-diff" src="https://user-images.githubusercontent.com/97027715/232963478-b4dbed5a-b380-4929-b71b-c01121899574.gif">

Get started with pip install:
```
pip install tvault
```

For those who are already using [VESSL Python SDK and CLI](https://docs.vessl.ai/api-reference/what-is-the-vessl-cli-sdk),
```
pip install "vessl[tvault]"
```

Follow our guide below with our [MNIST example code](https://github.com/vessl-ai/tvault/blob/1a6b5e038ff3fd4780a186bec7a555215a5e3c31/assets/mnist-train.py).

----
<br>

## Getting started with `tvault.log()`

Insert `tvault.log()` in your code's training loop with the metrics you want to track as tags.
https://github.com/vessl-ai/tvault/blob/4bebfe48594b00597889b2696dcec532d72b33d4/assets/mnist-train.py#L145-L146

`tvault.log()` will automatically create the following:

* A folder `model_log` under your current directory - a tracking dashboard or model registry for your code
* A unique hash for the model and model ID for each training run with the key metrics

<img alt="tvault-model" src="https://user-images.githubusercontent.com/97027715/232966704-b01ae877-a39e-4f9e-be19-148c69259485.gif">

## Look up experiments with `tvault --find_flag`

`tvault`'s `find_flag` option allows you to look up different experiments with a simple CLI command. `find_flag` offers three different ways of exploring results:

  1. Search by hash
The command below shows all experiments with the hash value of  `2ba4adf`. 
```
tvault --find_flag --condition hash --hash 2ba4adf shows all experiments with hash 2ba4adf.
```
<img alt="tvault-diff-hash" src="https://user-images.githubusercontent.com/97027715/232963946-ca830858-c215-4c6c-a798-3266df76ba37.gif">

  2. Search by result
The command below shows all experiments with the result between `50` and `100`.
```
tvault --find_flag --condition result --min 90 --max 100
```
<img alt="tvault-diff-result" src="https://user-images.githubusercontent.com/97027715/232963947-4f8b222c-b573-4cb4-aabe-5a4589614ec8.gif">

  3. Search by tags
The command below shows all experiments tagged as `0.5x`.
```
tvault --find_flag --condition tag --tag_type size --tag 0.5x
```
<img alt="tvault-diff-tag" src="https://user-images.githubusercontent.com/97027715/232963949-57cc88de-f5a9-440e-a447-8a8d412e8a5b.gif">

## Compare models with `tvault --diff_flag`

`tvault`'s `diff_flag` option allows you to look up the difference of two models by specifying the model hash and index. `tvault` automatically detects and displays the changes in functions while removing git diffs that are not related to the model.

This is useful when you have a baseline model that you want to iterate with different hyperparameters and higher-level architectures without digging through your code line-by-line. 

The following command, for example, provides the model difference between models in between two commits.
```
tvault --diff_flag --sha1 f407ed0 --index1 0 --sha2 737b47a --index2 0
```
<img alt="tvault-diff" src="https://user-images.githubusercontent.com/97027715/232963478-b4dbed5a-b380-4929-b71b-c01121899574.gif">

`tvault` can also get the diff related to the optimizers.

## Issues, feature requests, and questions

We are excited to hear your feedback!
* For issues and feature requests, please open a GitHub issue.
* For questions, please use GitHub Discussions.
* For general discussions, join our community Slack.

## VESSL for Academics

<img alt="vessl-for-academics" src="https://user-images.githubusercontent.com/97027715/232964554-6af7dabe-8409-4158-bafb-97e19441fac6.jpeg">

Our free academic plan is dedicated to help graduate students set up a modern ML research workflow with zero maintenance overheads. [Learn more](https://vesslai.notion.site/VESSL-for-Academics-fa47bf5e69b44e92b5daaead758cb057) about VESSL's full-fledged, fully-managed job scheduler and experiment tracking.
