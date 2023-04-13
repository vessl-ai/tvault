import os
import sys
import ast
import git
import pickle
import astunparse
from prettytable import PrettyTable
from collections import defaultdict

from .parse_utils import match_external_funcs, extract_info_from_model, extract_diff


class TorchVaultError(Exception):
    pass


class TorchVault:
    def __init__(self, log_dir="./model_log", model_dir="./"):
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.use_astunparse = True if sys.version_info.minor < 9 else False

        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        short_sha = sha[:7]
        self.sha = short_sha
        os.makedirs(self.log_dir, exist_ok=True)

    """
    reads model log from git hash
    returns empty model_log if nothing logged
    """

    def read_model_log(self, sha=""):
        if sha == "":
            sha = self.sha

        if os.path.exists(f"{self.log_dir}/model_{sha}"):
            with open(f"{self.log_dir}/model_{sha}", "rb") as f:
                model_log = defaultdict(lambda: dict(), pickle.load(f))
        else:
            model_log = defaultdict(lambda: dict())
        return model_log

    """
    write model log to git hash
    """

    def write_model_log(self, sha="", model_log=defaultdict(lambda: dict())):
        if sha == "":
            sha = self.sha
        with open(f"{self.log_dir}/model_{sha}", "wb") as f:
            pickle.dump(dict(model_log), f)

    """
    log torch scheduler 
    """

    def log_scheduler(self, scheduler):
        model_log = self.read_model_log()
        model_idx = len(model_log.keys()) - 1
        model_log[model_idx]["scheduler"] = scheduler.__str__()
        self.write_model_log("", model_log)

    """
    log torch optimizer
    """

    def log_optimizer(self, optimizer):
        model_log = self.read_model_log()
        model_idx = len(model_log.keys()) - 1
        model_log[model_idx]["optimizer"] = optimizer.__str__()
        self.write_model_log("", model_log)

    """
    add tag to model log, commit sha may be from previous results.
    if idx is set to -1, all models in the commit hash are tagged.
    if idx is set to None, tag of most recent model is changed.
    """

    def add_tag(self, sha="", tag_type="", tag="", idx=None):
        # if commit hash is not given, use current commit hash
        model_log = self.read_model_log(sha)
        if len(model_log.keys()) == 0:
            print(f"tvault error: model log with commit hash {sha} does not exist.")
            raise TorchVaultError

        if idx == -1:
            target_idxs = list(range(len(model_log.keys())))
        elif idx == None:
            target_idxs = [len(model_log.keys()) - 1]
            print(f"add tags: {target_idxs}")
        else:
            target_idxs = [idx]

        for model_idx in target_idxs:
            if f"tag-{tag_type}" in model_log[model_idx].keys():
                print(
                    f"tvault: changing tag from {model_log[model_idx][f'tag-{tag_type}']} to {tag} for model {sha}"
                )
            else:
                print(f"tvault: setting tag {tag} for model {sha}")
            model_log[model_idx][f"tag-{tag_type}"] = tag
        self.write_model_log(sha, model_log)

    """
    add result to model log, commit sha may be from previous results.
    if idx is set to -1, all models in the commit hash are tagged.
    if idx is set to None, tag of most recent model is changed.
    """

    def add_result(self, sha="", result=0, idx=None):
        # if commit hash is not given, use current commit hash
        model_log = self.read_model_log(sha)
        if len(model_log.keys()) == 0:
            print(f"tvault error: model log with commit hash {sha} does not exist.")
            raise TorchVaultError

        if idx == -1:
            target_idxs = list(range(len(model_log.keys())))
        elif idx == None:
            target_idxs = [len(model_log.keys()) - 1]
            print(f"add results: {target_idxs}")
        else:
            target_idxs = [idx]

        for model_idx in target_idxs:
            if "result" in model_log[model_idx].keys():
                print(
                    f"tvault: changing result from {model_log[model_idx]['result']} to {result} for model {sha}"
                )
            else:
                print(f"tvault: setting result {result} for model {sha}")
            model_log[model_idx]["result"] = result
        self.write_model_log(sha, model_log)

    """
    Basic logging for pytorch model.
    1. Retrives target modules from pytorch model representation.
    2. Get class definition of target modules.
    3. Get external function definition of those used in target model.

    Each logged model is stacked using index.
    """

    def log_model(self, model):
        class_defs, function_defs, target_modules = extract_info_from_model(model, self.model_dir)

        # get target module defs.
        filter_class_defs = defaultdict(lambda: "")
        for k, v in class_defs.items():
            if k.split(":")[-1] in target_modules:
                filter_class_defs[k] = v

        # find functions that we only want to track
        target_funcs = match_external_funcs(filter_class_defs)

        # unparse
        filter_target_class = defaultdict(lambda: "")
        for k, v in class_defs.items():
            if k.split(":")[-1] in target_modules:
                if self.use_astunparse:
                    filter_target_class[k] = astunparse.unparse(v)
                else:
                    filter_target_class[k] = ast.unparse(v)

        filter_target_funcs = defaultdict(lambda: "")
        for k, v in function_defs.items():
            if k.split(":")[-1] in target_funcs:
                if self.use_astunparse:
                    filter_target_funcs[k] = astunparse.unparse(v)
                else:
                    filter_target_funcs[k] = ast.unparse(v)

        model_log = self.read_model_log()
        model_idx = len(model_log.keys())
        model_log[model_idx]["model"] = model.__str__()
        model_log[model_idx]["src"] = dict(filter_target_class)
        model_log[model_idx]["external_func"] = dict(filter_target_funcs)
        self.write_model_log("", model_log)

    """
    Basic diff calculator between two pytorch models.
    sha1: commit hash of previous model, must be set. (for now)
    index1: model index of model in commit hash sha1. If not set, use latest.
    sha2: commit hash of current model, must be set. (for now)
    index2: model index of model in commit hash sha2. If not set, use latest.
    out: if out flag is set, writes out
    ask_gpt: if ask_gpt is set, asks gpt for difference summary.

    0412: Custom keys should not be considered when calculating diff.
    """

    def diff(self, sha1="", index1=-1, sha2="", index2=-1, ask_gpt=False):
        prev_model = self.read_model_log(sha1)
        cur_model = self.read_model_log(sha2)
        if len(prev_model.keys()) == 0:
            print(f"tvault error: sha1 argument must be provided.")
        if len(cur_model.keys()) == 0:
            print(f"tvault error: sha2 argument must be provided.")
        if index1 == -1:
            index1 = len(prev_model.keys()) - 1
        if index2 == -1:
            index2 = len(cur_model.keys()) - 1

        prev_model = prev_model[index1]
        cur_model = cur_model[index2]

        ret_str, diff_dict = extract_diff(prev_model, cur_model)

        print(ret_str)
        if ask_gpt:
            # self.ask_diff(diff_dict)
            # deprecated for now
            raise NotImplementedError

        return

    """
    find models using either commit hash, tag, or result
    should find suitable models and return list of [hash, model index, tag, result].
    Find models by that with custom keys, show result with such custom key
    """

    def find(self, condition="hash", hash="", tag_type="", tag="", min=0, max=100):
        target_models = []
        if os.path.exists(self.log_dir):
            if len(os.listdir(self.log_dir)) == 0:
                print(f"tvault error: log dir is empty")
                raise TorchVaultError
            if condition == "hash":
                if hash == "":
                    print(f"tvault error: hash is not set for hash finding")
                    raise TorchVaultError
                if os.path.exists(f"{self.log_dir}/model_{hash}"):
                    model_log = self.read_model_log(hash)
                    print(
                        f"tvault: model {hash} exists! - contains {len(model_log.keys())} experiments"
                    )
                    for model_idx, model in model_log.items():
                        model_info = {"HASH": hash, "MODEL-IDX": model_idx}
                        for k, v in model.items():
                            if "tag-" in k:
                                model_info[k] = v
                        if "result" in model.keys():
                            model_info["RESULT"] = model["result"]

                        target_models.append(model_info)

                else:
                    print(f"tvault: model {hash} does not exist.")
            elif condition == "tag":
                for model in os.listdir(self.log_dir):
                    with open(f"{self.log_dir}/{model}", "rb") as f:
                        model_log = pickle.load(f)
                    for model_idx, v in model_log.items():
                        model_info = dict()

                        if f"tag-{tag_type}" in v.keys() and v[f"tag-{tag_type}"] == tag:
                            model_info = {
                                "HASH": model.split("_")[1],
                                "MODEL-IDX": model_idx,
                            }
                            if "result" in v.keys():
                                model_info["RESULT"] = v["result"]

                            # other tags
                            for m_k, m_v in v.items():
                                if "tag-" in m_k:
                                    model_info[m_k] = m_v
                            target_models.append(model_info)
            elif condition == "result":
                for model in os.listdir(self.log_dir):
                    with open(f"{self.log_dir}/{model}", "rb") as f:
                        model_log = pickle.load(f)
                    for model_idx, v in model_log.items():
                        model_info = dict()

                        if "result" in v.keys() and min <= v["result"] <= max:
                            model_info = {
                                "HASH": model.split("_")[1],
                                "MODEL-IDX": model_idx,
                                "RESULT": v["result"],
                            }
                            # other tags
                            for m_k, m_v in v.items():
                                if "tag-" in m_k:
                                    model_info[m_k] = m_v

                            target_models.append(model_info)
            else:
                print(f"tvault error:condition other than [hash, tag, result] is not supported.")
                raise TorchVaultError
        return target_models

    def show_result(self, target_models):
        if len(target_models) == 0:
            print(f"tvault: no model satisfying the conditions")
            return

        # find model log with most tags.
        table = []
        for e in target_models:
            if len(list(e.keys())) > len(table):
                table = list(e.keys())

        # table should be sorted here
        sorted_table_out = ["HASH", "MODEL-IDX"]
        sorted_table = ["HASH", "MODEL-IDX"]
        for e in table:
            if e.startswith("tag-"):
                sorted_table_out.append(e.split("tag-")[1])
                sorted_table.append(e)
        sorted_table.append("RESULT")
        sorted_table_out.append("RESULT")

        # table contains model_log keys with most tags
        # should represent [hash, model idx, tag1, tag2, tag3, result] order.
        tab = PrettyTable(sorted_table_out)
        for e in target_models:
            row = []
            for k in sorted_table:
                if k in e.keys():
                    row.append(e[k])
                else:
                    row.append("")
            tab.add_row(row)
        print(tab)
