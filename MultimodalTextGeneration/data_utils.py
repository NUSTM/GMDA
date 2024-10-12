import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET


def get_dataset(mode, config):
    if config["task"] in ["fmnerg", "gmner"]:
        return FMNERG_Dataset(
            text_file_path=config["text_dir"],
            img_file_path=config["image_dir"],
            image_annotation_path=config["image_annotation_path"],
            mode=mode,
            data_format=config["data_format"],
        )
    else:
        raise NotImplementedError


class FMNERG_Dataset(Dataset):
    def __init__(
        self, text_file_path, img_file_path, image_annotation_path, mode, data_format
    ):
        self.image_annotation_path = image_annotation_path
        self.img_file_path = img_file_path

        self.data = []
        if data_format == "json":
            with open(os.path.join(text_file_path, mode + ".json"), "r") as file:
                self.data = json.load(file)
        else:
            raise ValueError(
                "Unsupported data format. Currently, the program only supports JSON format."
            )

    def __len__(self):
        return len(self.data)

    def _read_image_label(self, img_id):
        if not os.path.exists(
            os.path.join(self.image_annotation_path, img_id + ".xml")
        ):
            return -1, -1, [], []
        fn = os.path.join(self.image_annotation_path, img_id + ".xml")

        tree = ET.parse(fn)
        root = tree.getroot()

        width = int(root.find("size/width").text)
        height = int(root.find("size/height").text)

        entities = []
        boxes = []

        for object_container in root.findall("object"):
            for names in object_container.findall("name"):
                box_name = names.text
                box_container = object_container.findall("bndbox")
                if len(box_container) > 0:
                    xmin = int(box_container[0].findall("xmin")[0].text)
                    ymin = int(box_container[0].findall("ymin")[0].text)
                    xmax = int(box_container[0].findall("xmax")[0].text)
                    ymax = int(box_container[0].findall("ymax")[0].text)
                    entities.append(box_name)
                    boxes.append([xmin, ymin, xmax, ymax])
        return width, height, entities, boxes

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample
