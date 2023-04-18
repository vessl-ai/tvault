# `tvault` by VESSL

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://user-images.githubusercontent.com/97027715/232697803-3571bd58-8d4a-4c42-adba-96f300ef72c4.png" width="35%">
    <img alt="SkyPilot" src="https://user-images.githubusercontent.com/97027715/232697811-f0a666a6-acbd-43a9-8af9-dea3e7cc0936.png" width="35%">
  </picture>
</p>

<p align="center">
    <a target="_blank" href="https://www.linkedin.com/company/vesslai"><img src="https://img.shields.io/badge/style--5eba00.svg?label=LinkedIn&logo=linkedin&style=social"></a>&nbsp;
    <a target="_blank" href="https://vesslai.medium.com/"><img src="https://img.shields.io/badge/style--5eba00.svg?label=Medium&logo=medium&style=social"></a>&nbsp;
    <a target="_blank" href="https://www.youtube.com/@vesslai4254"><img src="https://img.shields.io/badge/style--5eba00.svg?label=YouTube&logo=youtube&style=social"></a>&nbsp;
    <a target="_blank" href="https://join.slack.com/t/vessl-ai-community/shared_invite/zt-1a6schu04-NyjRKE0UMli58Z_lthBICA"><img src="https://img.shields.io/badge/Slack-Join-4A154B?logo=slack&style=social"></a>&nbsp;  
</p>

<h3 align="center">
    Compare and store models in a local, lightweight registry
</h3>

----

### The challenges we are tackling

`tvault` is designed to help academic researchers iterate their models faster without the logging overhead. 

Many of the academic researchers we encounter simply want to *get going* with minimum setup and configurations. This often means using local codebase as opposed to integrating tools like [VESSL Experiments](https://docs.vessl.ai/api-reference/python-sdk/utils/vessl.log) and Weights & Biases to git-committed code. 

`tvault` is "git diff for ML" &mdash; a simple, lightweight framework for quickly tracking and comparing ML workloads in a local model registry. 

* Track and version models locally with `tvault.log_all()`
* Get a birds-eye differences of two workloads with `tvault --diff_flag`

Get started with pip install:
```
pip install tvault
```

For those who are already using VESSL Python SDK and CLI,
```
pip install "vessl[tvault]"
```

Follow our guide below with our [MNIST example code](https://github.com/saeyoon17/mnist-tvault-example/blob/main/train.py). 

----

### Getting started

Start by inserting `tvault.log()` in your code's training loop. 


This creates a folder `model_log` under your current directory
