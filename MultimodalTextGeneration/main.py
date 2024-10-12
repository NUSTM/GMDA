import os
import random

import numpy as np
import torch
import argparse

import yaml
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from Collator import Collator
from data_utils import FMNERG_Dataset, get_dataset
from transformers import (
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
    InstructBlipConfig,
)
from Trainer import Trainer
from torch.utils.data import DataLoader


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # Reading YAML configuration file
    with open("config.yaml", "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
        config["text_dir"] = os.path.join(config["text_dir"], config["dataset"])
        config["training_argument"]["output_dir"] = os.path.join(
            config["training_argument"]["output_dir"], config["dataset"]
        )

    set_seed(config["seed"])

    device = torch.device("cuda:0")
    processor = InstructBlipProcessor.from_pretrained(config["model_name_or_path"])
    test_dataset = get_dataset("train", config)
    dev_dataset = get_dataset("dev", config)
    collator = Collator(
        processor=processor, task=config["task"], img_file_path=config["image_dir"]
    )

    aug_data = None

    if config["run_mode"] == "do_train":
        model = InstructBlipForConditionalGeneration.from_pretrained(
            config["model_name_or_path"],
            torch_dtype = torch.bfloat16
        ).to(device)

        for name, param in model.named_parameters():
            if (
                    any(prefix in name for prefix in ["vision_model"])
                    and param.requires_grad
            ):
                param.requires_grad = False

        if config["trainable_mode"] == "lora":
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=[
                    "v",
                    "q",
                    "o",
                    "wi_0",
                    "wi_1",
                    "dense",
                    "value",
                    "key",
                    "qkv",
                    "word_embeddings",
                    "fc1",
                    "fc2",
                    "projection",
                    "relative_attention_bias",
                ],
            )
            model = get_peft_model(model, peft_config)
            model = model.to(device)
            config["training_argument"]["output_dir"] = os.path.join(
                config["training_argument"]["output_dir"], "lora"
            )


        total_params = sum(p.numel() for p in model.parameters())
        print("total_params:", total_params)

        trainable_total_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print("trainable_total_params:", trainable_total_params)

        train_dataset = get_dataset("train", config)

        # Creating Trainer object
        trainer = Trainer(
            model=model,
            processor=processor,
            data_collator=collator,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            test_dataset=test_dataset,
            **config["training_argument"],
        )

        # Start the training process
        trainer.train()

    elif config["run_mode"] == "do_inference":
        model_config = InstructBlipConfig.from_pretrained(config["model_name_or_path"])
        model = InstructBlipForConditionalGeneration(model_config)
        if config["trainable_mode"] == "lora":
            config["training_argument"]["output_dir"] = os.path.join(
                config["training_argument"]["output_dir"], "lora_inference"
            )
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=[
                    "v",
                    "q",
                    "o",
                    "wi_0",
                    "wi_1",
                    "dense",
                    "value",
                    "key",
                    "qkv",
                    "word_embeddings",
                    "fc1",
                    "fc2",
                    "projection",
                    "relative_attention_bias",
                ],
            )
            model = get_peft_model(model, peft_config)
            model.model.load_state_dict(torch.load(config["load_checkpoint"]))
            model = model.to(device)

        # Creating Trainer object
        trainer = Trainer(
            model=model,
            processor=processor,
            data_collator=collator,
            test_dataset=test_dataset,
            eval_dataset=dev_dataset,
            **config["training_argument"],
        )

        # generate aug data for the training set
        trainer.aug_data_generate(sub_data="train", dataset=config["dataset"], num_return_sequences=config["training_argument"]["num_return_sequences"])


if __name__ == "__main__":
    main()
