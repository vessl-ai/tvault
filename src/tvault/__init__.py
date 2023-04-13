from .torchvault import TorchVault
import click

"""
Logging Functions
"""


def log(model, log_dir="./model_log", model_dir="./"):
    vault = TorchVault(log_dir, model_dir)
    vault.log_model(model)


def log_scheduler(scheduler, log_dir="./model_log", model_dir="./"):
    vault = TorchVault(log_dir, model_dir)
    vault.log_scheduler(scheduler)


def log_optimizer(optimizer, log_dir="./model_log", model_dir="./"):
    vault = TorchVault(log_dir, model_dir)
    vault.log_optimizer(optimizer)


def diff(sha1="", index1=-1, sha2="", index2=-1, ask_gpt=False, log_dir="./model_log"):
    # def diff(self, sha1="", index1=-1, sha2="", index2=-1, out=False, ask_gpt=False)
    vault = TorchVault(log_dir)
    vault.diff(sha1, index1, sha2, index2, ask_gpt)


def add_tag(tag_type="", tag="", sha="", log_dir="./model_log"):
    # def add_tag(self, sha="", tag_type="", tag="", idx=None)
    vault = TorchVault(log_dir)
    vault.add_tag(sha, tag_type, tag)


def add_result(result=0, sha="", log_dir="./model_log"):
    vault = TorchVault(log_dir)
    vault.add_result(sha, result)


# tags should be a dictionary of key, value pairs
def log_all(model, tags=dict(), result=-1, optimizer=None, log_dir="./model_log", model_dir="./"):
    vault = TorchVault(log_dir, model_dir)
    vault.log_model(model)
    if optimizer is not None:
        vault.log_optimizer(optimizer)
    for tag_type, tag in tags.items():
        vault.add_tag("", tag_type, tag)
    if result is not -1:
        vault.add_result("", result)


"""
Other utils
"""


def find(
    log_dir="./model_log",
    model_dir="./",
    condition="hash",
    hash="",
    tag_type="",
    tag="",
    min=0,
    max=100,
):
    vault = TorchVault(log_dir, model_dir)
    target_models = vault.find(condition, hash, tag_type, tag, min, max)
    vault.show_result(target_models)


"""
cli utils
"""


@click.command()
@click.option("--find_flag", is_flag=True, default=False, help="tvault cli for tvault.find")
@click.option("--diff_flag", is_flag=True, default=False, help="tvault cli for tvault.diff")
# options for find
@click.option("--log_dir", type=str, default="./model_log")
@click.option("--model_dir", type=str, default="./")
@click.option("--condition", type=str, default="hash")
@click.option("--hash", type=str, default="")
@click.option("--tag_type", type=str, default="")
@click.option("--tag", type=str, default="")
@click.option("--min", type=int, default=0)
@click.option("--max", type=int, default=100)
# options for diff
@click.option("--sha1", type=str, default="")
@click.option("--index1", type=int, default=0)
@click.option("--sha2", type=str, default="")
@click.option("--index2", type=int, default=0)
def cli_main(
    find_flag,
    diff_flag,
    log_dir,
    model_dir,
    condition,
    hash,
    tag_type,
    tag,
    min,
    max,
    sha1,
    index1,
    sha2,
    index2,
):
    if find_flag:
        find(log_dir, model_dir, condition, hash, tag_type, tag, min, max)
    elif diff_flag:
        diff(sha1, index1, sha2, index2, ask_gpt=False, log_dir=log_dir)
    else:
        print("tvault: not implemented")
