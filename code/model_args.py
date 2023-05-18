import torch
import os, json, sys, warnings
from torch.utils.data import Dataset
from dataclasses import asdict, dataclass, field, fields


@dataclass
class BoundaryExtractionArgs:
    """
    model args for bert.
    """
    manual_seed: int = 42  # 宇宙的真理42
    model_name: str = None
    tokenizer_name: str = None
    max_length: int = 512
    optimizer: str = "AdamW"
    epoch: int = 0
    lr: float = 1e-5
    batch_size: int = 8
    cuda_index: int = 0
    n_gpu: int = 0
    num_labels: int = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def update_from_dict(self, new_values):
        """
        Adding parameters customized.
        :param new_values:
        :return:
        """
        # print("Add Values:", new_values)
        if isinstance(new_values, dict):
            for key, value in new_values.items():
                setattr(self, key, value)
        else:
            raise (TypeError(f"{new_values} is not a Python dict."))

    def get_args_for_saving(self):
        args_for_saving = {
            key: value
            for key, value in asdict(self).items()
            if key not in self.not_saved_args
        }
        if "settings" in args_for_saving["wandb_kwargs"]:
            del args_for_saving["wandb_kwargs"]["settings"]
        return args_for_saving

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "model_args.json"), "w") as f:
            args_dict = self.get_args_for_saving()
            if args_dict["tokenizer_type"] is not None and not isinstance(
                    args_dict["tokenizer_type"], str
            ):
                args_dict["tokenizer_type"] = type(args_dict["tokenizer_type"]).__name__
            json.dump(args_dict, f)

    def load(self, input_dir):
        if input_dir:
            model_args_file = os.path.join(input_dir, "model_args.json")
            if os.path.isfile(model_args_file):
                with open(model_args_file, "r") as f:
                    model_args = json.load(f)

                self.update_from_dict(model_args)
