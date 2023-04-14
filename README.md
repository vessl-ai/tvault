# tvault
Local ml model registry with model diff for efficient model development.

# QuickStart
Install tvault using PyPI:
`pip install tvault`

## Logging

After your model training, make your model local registry using:

`tvault.log_all(model, tags=tags, result=acc.item(), optimizer=optimizer)`

tvault will automatically detect necessary components (related source code) and build model registry.

You can add your custom tags which will reside within model registry.

(i.e. `tags = {"language": "pytorch", "learning_rate": learning_rate}`)

## Searching

After model registries have been made, there are three ways to look desired experiments.

- `tvault --find_flag --condition hash --hash yourhash`

Provides all model registries under the git commit hash. Multiple registries under same git commit are differentiated using model index.

- `tvault --find_flag --condition result --min yourmin --max yourmax`

Provides all model registries that have result value between certain boundary.

- `tvault --find_flag --condition tag --tag_type yourtag --tag tagcontent`

Provies all model registries that have value `tagcontent` under tag `yourtag`.

## Comparing diffs
A naive approach for debugging model performance is to look at the git hash, and use git diff to see what parts have been changed. This causes time overhead. At the same time, git diff across time usually contains changes that are not directly connected to model performance.

When creating model registry, tvault filters source parts that are related to model itself. Therefore, it can show diffs of source codes that are directly related to your model. 

Once you have located two models (commit hash, model index)s, use following command:

`tvault --diff_flag --sha1 hash1 --index1 index1 --sha2 hash2 --index2 index2`

The command will automatically provide diffs that are related to model. Specifically, we provide diffs between:

- model architecture (using pytorch model __repr__)
- model class source
- external functions defined in each model class

Visit the following links for more detailed example usage.
- [MNIST using ResNet-18](https://github.com/saeyoon17/mnist-tvault-example)
- [CIFAR-100 using MobilenetV2](https://github.com/saeyoon17/cifar-100-tvault-example)



