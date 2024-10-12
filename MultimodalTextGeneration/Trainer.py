import glob
import os.path
import re

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime


class Trainer:
    def __init__(
        self,
        model,
        processor,
        data_collator,
        train_dataset=None,
        eval_dataset=None,
        test_dataset=None,
        neptune_run=None,
        **kwargs,
    ):

        self.train_batch_size = kwargs["train_batch_size"]
        self.trainable_mode = kwargs["trainable_mode"]
        if self.trainable_mode == "lora":
            self._model = model.model
        else:
            self._model = model
        self.processor = processor

        if train_dataset is not None:
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=kwargs["train_batch_size"],
                shuffle=True,
                collate_fn=data_collator,
            )
        if eval_dataset is not None:
            self.eval_loader = DataLoader(
                eval_dataset,
                batch_size=kwargs["eval_batch_size"],
                collate_fn=data_collator,
            )
        if test_dataset is not None:
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=kwargs["eval_batch_size"],
                collate_fn=data_collator,
            )
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=kwargs["learning_rate"]
        )
        self.num_train_epochs = kwargs["num_train_epochs"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self.device)
        self.current_step = 0
        self.step_interval = kwargs["step_interval"]

        self.saved_model_dir = os.path.join(kwargs["output_dir"], "models")
        self.generate_text_dir = os.path.join(kwargs["output_dir"], "aug_text")
        self.eval_delay = kwargs["eval_delay"]
        self.best_eval_loss = float("inf")
        self.best_model_path = None
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if not os.path.exists(self.saved_model_dir):
            os.makedirs(self.saved_model_dir)

        if not os.path.exists(self.generate_text_dir):
            os.makedirs(self.generate_text_dir)

    def train(self):

        for epoch in range(self.num_train_epochs):
            self._model.train()
            total_loss = 0

            # Training loop
            train_loop = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")
            for batch_data in train_loop:
                self.current_step += 1
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch_data["inputs"].items()
                }

                self.optimizer.zero_grad()
                outputs = self._model(**batch)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                train_loop.set_description(
                    f"[Epoch {epoch + 1}/{self.num_train_epochs}]"
                )
                train_loop.set_postfix(train_loss=loss.item())

            avg_loss = total_loss / len(self.train_loader)
            avg_eval_loss = self.evaluate()
            print(
                f"Epoch {epoch + 1}: Average Train Loss: {avg_loss:.4f} "
                f"Average Eval Loss: {avg_eval_loss:.4f}"
            )

            if avg_eval_loss < self.best_eval_loss:
                self.best_eval_loss = avg_eval_loss
                if epoch + 1 >= self.eval_delay:
                    self.save_best_model(avg_eval_loss, epoch + 1)

    def evaluate(self):
        self._model.eval()
        total_loss = 0.0
        evaluate_loop = tqdm(self.eval_loader, desc="Evaluation")

        with torch.no_grad():
            for batch_data in evaluate_loop:
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch_data["inputs"].items()
                }
                outputs = self._model(**batch)

                loss = outputs.loss
                total_loss += loss.item()

                evaluate_loop.set_postfix(evaluate_loss=loss.item())

        average_loss = total_loss / len(self.eval_loader)

        print(f"Evaluation Average Loss: {average_loss:.4f}")
        return average_loss

    def save_best_model(self, eval_loss, epoch):
        if self.best_model_path is not None and os.path.exists(self.best_model_path):
            os.remove(self.best_model_path)

        # Create a name for the current model file based on timestamp and current loss
        model_name = (
            f"best_model_lr-{self.optimizer.param_groups[0]['lr']}"
            f"_epochs-{epoch}_eval_loss-{eval_loss:.4f}"
            f"{self.trainable_mode}_{self.timestamp}.pt"
        )
        current_model_path = os.path.join(self.saved_model_dir, model_name)
        self.best_model_path = current_model_path

        # Save the current model and update the best model path and loss
        torch.save(self._model.state_dict(), current_model_path)
        print(f"Model saved with evaluation loss: {eval_loss:.4f}")

    def aug_data_generate(self, sub_data="test", dataset="FMNERG", num_return_sequences=1):
        if sub_data == "eval":
            if not hasattr(self, "eval_loader"):
                raise ValueError
            dataloader = self.eval_loader
        elif sub_data in ["train", "test"]:
            if not hasattr(self, "test_loader"):
                raise ValueError
            dataloader = self.test_loader
        else:
            raise ValueError

        aug_text_output_path = os.path.join(
            self.generate_text_dir,
            f"{dataset}_aug_text_{sub_data}-{self.optimizer.param_groups[0]['lr']}_epochs-{self.num_train_epochs}.txt",
        )
        with open(aug_text_output_path, "w", encoding="utf-8") as output_file:
            for batch_data in tqdm(dataloader):
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch_data["inputs"].items()
                }
                with torch.no_grad():
                    generated_ids = self._model.generate(
                        **batch,
                        do_sample=True,
                        max_length=512,
                        top_k=50,
                        top_p=0.95,
                        num_return_sequences=num_return_sequences,
                        min_length=5
                        # max_new_tokens=512,
                    )
                    
                generated_text_list = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                for generated_text in generated_text_list:
                    if "post_process_generation" in dir(self.processor):
                        processed_text = self.processor.post_process_generation(
                            generated_text, cleanup_and_extrxact=False
                        )
                    else:
                        processed_text = generated_text
                    output_file.write(processed_text + "\n")

        print(f"Processed texts saved to: {aug_text_output_path}")
