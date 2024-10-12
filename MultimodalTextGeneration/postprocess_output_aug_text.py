import json
import os
import re

import yaml

from data_utils import FMNERG_Dataset, get_dataset
from remove_label_in_raw_sentence import process_sentence, extract_entities

coarse_fine_tree = {
    "location": [
        "city",
        "country",
        "state",
        "continent",
        "location_other",
        "park",
        "road",
    ],
    "building": [
        "building_other",
        "cultural_place",
        "entertainment_place",
        "sports_facility",
    ],
    "organization": [
        "company",
        "educational_institution",
        "band",
        "government_agency",
        "news_agency",
        "organization_other",
        "political_party",
        "social_organization",
        "sports_league",
        "sports_team",
    ],
    "person": [
        "politician",
        "musician",
        "actor",
        "artist",
        "athlete",
        "author",
        "businessman",
        "character",
        "coach",
        "director",
        "intellectual",
        "journalist",
        "person_other",
    ],
    "other": ["animal", "award", "medical_thing", "website", "ordinance"],
    "art": [
        "art_other",
        "film_and_television_works",
        "magazine",
        "music",
        "written_work",
    ],
    "event": ["event_other", "festival", "sports_event"],
    "product": ["brand_name_products", "game", "product_other", "software"],
}


def check():
    with open("config.yaml", "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
    directory = "./output_dir"
    # for root, dirs, files in os.walk(directory):
    #     for file in files:
    #         # 检查文件名是否以.txt结尾
    #         if file.endswith(".txt") and "eval" in file:
    #             # 拼接文件的完整路径
    # check_txt = os.path.join(root, file)
    check_txt = "/root/data1/brick/BLIP2_data_augmentation/augCodes/output_dir/50GMNER/lora_inference/aug_text/aug_text-5e-05_epochs-5.txt"

    # out_json = os.path.join("/root/data1/brick/BLIP2_data_augmentation/augText/json", file[:-4] + ".json")

    # config["dataset"] = file.split("_")[0]
    config["dataset"] = "50GMNER"
    config["text_dir"] = (
        "/root/data1/brick/dataset/original_MNER_4label"
        if "GMNER" in config["dataset"]
        else "/root/data1/brick/dataset/fine-grained"
    )
    config["text_dir"] = os.path.join(config["text_dir"], config["dataset"])
    config["training_argument"]["output_dir"] = os.path.join(
        config["training_argument"]["output_dir"], config["dataset"]
    )

    test_dataset = get_dataset("train", config)

    aug_text = []
    with open(check_txt, "r", encoding="utf-8") as f:
        for line in f:
            aug_text.append(line.strip())

    aug_json_list = []
    useful_items = 0
    sum_items = 0
    for test_data, aug_text_item in zip(test_dataset, aug_text):
        entities_appear = True
        sum_items += 1

        for entity in test_data["entities"]:
            if entity["text"] not in aug_text_item:
                entities_appear = False
                break

        if entities_appear:
            aug_json = dict()
            aug_json["img_id"] = test_data["img_id"]
            aug_json["entities"] = test_data["entities"]
            aug_json["sentence"] = aug_text_item
            aug_json_list.append(aug_json)
            useful_items += 1

    print(
        f"{check_txt}\n"
        f"useful aug num: {useful_items} || sum aug num: {sum_items} || Proportion: {useful_items * 100.0 / sum_items}%"
    )
    # with open(out_json, "w") as f:
    #     json.dump(aug_json_list, f, indent=4)


def is_match_case_insensitive(string, match):
    """
    判断一个字符串（match）是否在另一个字符串（string）中存在，不区分大小写。

    参数：
    - string (str): 要搜索的原始字符串。
    - match (str): 要检查是否存在的子串。

    返回：
    - bool: 如果匹配存在，则返回 True，否则返回 False。
    """
    # 通过 re.escape(match) 转义 match 中的特殊字符，创建正则表达式模式
    # re.IGNORECASE 标志使搜索不区分大小写
    pattern = re.compile(re.escape(match), re.IGNORECASE)

    # 使用 re.search() 搜索模式在字符串中的位置
    match_found = pattern.search(string)

    # 如果找到匹配，返回 True，否则返回 False
    return bool(match_found)


def replace_case_insensitive(string, span):
    """
    替换字符串中的指定子串（span），不区分大小写。

    参数：
    - string (str): 要进行替换的原始字符串。
    - span (str): 要被替换的子串。

    返回：
    - str: 替换后的新字符串。
    """
    # 通过 re.escape(span) 转义 span 中的特殊字符，创建正则表达式模式
    # re.IGNORECASE 标志使匹配不区分大小写
    pattern = re.compile(re.escape(span), re.IGNORECASE)

    # 使用 pattern.sub() 方法将匹配到的 span 替换为原始的 span
    result = pattern.sub(span, string)

    # 返回替换后的新字符串
    return result


def add_space_between_symbols_and_letters(text):
    # 使用正则表达式匹配非字母符号和字母符号之间的位置，只添加一个空格
    pattern = re.compile(r"(?<=[^\w\s])\s*|\s*(?=[^\w\s])")

    # 在匹配的位置添加一个空格
    spaced_text = re.sub(pattern, " ", text)

    return spaced_text


def json_to_txt_fmnerg(json_data, output_file, aug=False):
    with open(output_file, "w", encoding="utf-8") as txt_file:
        for entry in json_data:

            img_id = entry["img_id"]
            sentence = add_space_between_symbols_and_letters(entry["sentence"])
            if len(sentence) == 0:
                continue
            entities = entry.get("entities", [])

            if aug is False:
                txt_file.write(f"IMGID:{img_id}\n")
            else:
                txt_file.write(f"IMGID:{img_id}_aug\n")

            if sentence:
                tokens = sentence.split()
                tags = [["O", "O"] for _ in range(len(tokens))]
                for entity in entities:
                    text = entity["text"]
                    entity_tokens = text.split()
                    iter_entity_token = 0
                    begin_pos = None
                    for i, token in enumerate(tokens):
                        if token == entity_tokens[iter_entity_token]:
                            if iter_entity_token == 0:
                                begin_pos = i
                            iter_entity_token += 1
                        else:
                            iter_entity_token = 0
                            begin_pos = None
                        if iter_entity_token == len(entity_tokens):
                            tags[begin_pos] = [
                                f'B-{entity["tag_coarse"]}',
                                f'B-{entity["tag_fine"]}',
                            ]
                            for j in range(begin_pos + 1, i + 1):
                                tags[j] = [
                                    f'I-{entity["tag_coarse"]}',
                                    f'I-{entity["tag_fine"]}',
                                ]
                            iter_entity_token = 0

                for i, token in enumerate(tokens):
                    txt_file.write(f"{token}\t{tags[i][0]}\t{tags[i][1]}\n")

            txt_file.write("\n")


def json_to_txt_gmner(json_data, output_file, aug=False):
    with open(output_file, "w", encoding="utf-8") as txt_file:
        for entry in json_data:
            img_id = entry["img_id"]
            sentence = add_space_between_symbols_and_letters(entry["sentence"])
            if len(sentence) == 0:
                continue
            entities = entry.get("entities", [])
            if aug is False:
                txt_file.write(f"IMGID:{img_id}\n")
            else:
                txt_file.write(f"IMGID:{img_id}_aug\n")

            if sentence:
                tokens = sentence.split()
                tags = ["O" for _ in range(len(tokens))]
                for entity in entities:
                    text = entity["text"]
                    entity_tokens = text.split()
                    iter_entity_token = 0
                    begin_pos = None
                    for i, token in enumerate(tokens):
                        if token == entity_tokens[iter_entity_token]:
                            if iter_entity_token == 0:
                                begin_pos = i
                            iter_entity_token += 1
                        else:
                            iter_entity_token = 0
                            begin_pos = None
                        if iter_entity_token == len(entity_tokens):
                            tags[begin_pos] = f'B-{entity["tag"]}'

                            for j in range(begin_pos + 1, i + 1):
                                tags[j] = f'I-{entity["tag"]}'
                            iter_entity_token = 0

                for i, token in enumerate(tokens):
                    txt_file.write(f"{token}\t{tags[i]}\n")

            txt_file.write("\n")


def check_and_turn_to_json(aug_txt_lines, raw_json_data, sample_number=5):
    count_do_remain_useful_item = 0

    img_ids = [item["img_id"] for item in raw_json_data]
    aug_text_list_with_img_ids = []
    index = 0

    for i, text in enumerate(aug_txt_lines):
        if len(text) == 0:
            continue
        if (
                len(aug_text_list_with_img_ids) == 0
                or len(aug_text_list_with_img_ids[-1]["text"]) == sample_number
        ):
            aug_text_list_with_img_ids.append({"img_id": img_ids[index], "text": []})
            index += 1

        aug_text_list_with_img_ids[-1]["text"].append(text)

    aug_json_out_list = []

    for dict_item, raw_json_item in zip(aug_text_list_with_img_ids, raw_json_data):
        assert dict_item["img_id"] == raw_json_item["img_id"]
        entities = raw_json_item["entities"]

        do_remain_useful_item = False

        for text in dict_item["text"]:
            augmented_entities = []

            if len(text.split()) < 5 :
                continue

            do_have_entities = False
            for entity in entities:
                if is_match_case_insensitive(text, entity["text"]):
                    text = replace_case_insensitive(text, entity["text"])
                    do_have_entities = True
                    augmented_entities.append(entity)

            if do_have_entities is False and len(entities) > 0:
                continue

            do_remain_useful_item = True
            aug_json_out_list.append(
                {
                    "img_id": raw_json_item["img_id"],
                    "entities": augmented_entities,
                    "sentence": text,
                }
            )

        if do_remain_useful_item:
            count_do_remain_useful_item += 1

    total_images = len(img_ids) * sample_number
    percentage_useful_images = count_do_remain_useful_item / len(img_ids) * 100

    output_message = (
        f"All Text: {total_images}\n"
        f"Total Remain Useful Item: {count_do_remain_useful_item}/{len(img_ids)} || {percentage_useful_images:.2f}%"
    )

    print(output_message)

    return aug_json_out_list


def process_labeled_file_fmnerg(aug_txt_lines, raw_json_data, sample_number=5):
    unavailable_entity_count = 0

    img_ids = [item["img_id"] for item in raw_json_data]
    img_ids_with_sample_number_repeat = [
        item for item in img_ids for _ in range(sample_number)
    ]

    aug_txt_lines_sample_number_samples = [
        item for item in aug_txt_lines if len(item) > 0
    ]
    aug_text_list_with_img_ids = []
    total_number = len(aug_txt_lines_sample_number_samples)

    for (text, img_id) in zip(
            aug_txt_lines_sample_number_samples, img_ids_with_sample_number_repeat
    ):
        aug_text_dict = process_sentence(text, task="fmnerg")

        # check if available
        if len(aug_text_dict["sentence"].split(" ")) < 5:
            continue

        is_available = True
        entity_text_list = []
        for entity in aug_text_dict["entities"]:
            if (
                    entity["tag_coarse"] not in coarse_fine_tree.keys()
                    or entity["tag_fine"] not in coarse_fine_tree[entity["tag_coarse"]]
                    or entity["text"] in entity_text_list
                    or len(entity["text"]) == 0
                    or "-" in entity["text"]
            ):
                is_available = False
            entity_text_list.append(entity["text"])

        if not is_available:
            unavailable_entity_count += 1
            continue
        aug_text_dict["img_id"] = img_id

        aug_text_list_with_img_ids.append(aug_text_dict)

    print(f"unavailable_entity_count : {unavailable_entity_count}")
    print(f"count useful items : {len(aug_text_list_with_img_ids)} {total_number}")

    return aug_text_list_with_img_ids


def process_labeled_file_gmner(aug_txt_lines, raw_json_data, sample_number=5):
    unavailable_entity_count = 0

    img_ids = [item["img_id"] for item in raw_json_data]
    img_ids_with_sample_number_repeat = [
        item for item in img_ids for _ in range(sample_number)
    ]

    aug_txt_lines_sample_number_samples = [
        item for item in aug_txt_lines if len(item) > 0
    ]
    aug_text_list_with_img_ids = []
    total_number = len(aug_txt_lines_sample_number_samples)

    for (text, img_id) in zip(
            aug_txt_lines_sample_number_samples, img_ids_with_sample_number_repeat
    ):
        aug_text_dict = process_sentence(text, task="gmner")

        # check if available
        if len(aug_text_dict["sentence"].split(" ")) < 5:
            continue
        
        if len(aug_text_dict["sentence"].split(" ")) >= 30:
            continue
        is_available = True
        entity_text_list = []
        for entity in aug_text_dict["entities"]:
            if (
                    entity["tag"] not in ["LOC", "PER", "ORG", "OTHER"]
                    or entity["text"] in entity_text_list
                    or len(entity["text"]) == 0
                    or "-" in entity["text"]
            ):
                is_available = False
            entity_text_list.append(entity["text"])

        if not is_available:
            unavailable_entity_count += 1
            continue
        aug_text_dict["img_id"] = img_id

        aug_text_list_with_img_ids.append(aug_text_dict)

    print(f"unavailable_entity_count : {unavailable_entity_count}")
    print(f"count useful items : {len(aug_text_list_with_img_ids)} {total_number}")

    return aug_text_list_with_img_ids




def _bio_tag_to_spans(tags, ignore_labels=None):
    r"""
    给定一个tags的list，比如['O', 'B-singer', 'I-singer', 'I-singer', 'O', 'O']。
        返回[('singer', (1, 4))] (左闭右开区间)

    :param tags: List[str],
    :param ignore_labels: List[str], 在该list中的label将被忽略
    :return: List[Tuple[str, List[int, int]]]. [(label，[start, end])]
    """
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bio_tag = None
    for idx, tag in enumerate(tags):
        tag = tag.lower()
        bio_tag, label = tag[:1], tag[2:]
        if bio_tag == "b":
            spans.append((label, [idx, idx]))
        elif bio_tag == "i" and prev_bio_tag in ("b", "i") and label == spans[-1][0]:
            spans[-1][1][1] = idx
        elif bio_tag == "o":  # o tag does not count
            pass
        else:
            spans.append((label, [idx, idx]))
        prev_bio_tag = bio_tag
    return [
        (span[0], (span[1][0], span[1][1] + 1))
        for span in spans
        if span[0] not in ignore_labels
    ]


def turn_txt_to_t5_data(input_file, output_file):
    f = open(os.path.join(input_file), "r")

    new_lines = []

    sentence = []
    label = []
    img_id = ""
    for line in f:
        if line.startswith("IMGID:"):
            img_id = line.strip()[6:]
        elif line == "\n":
            ## Despite the cold, the crowd soon warmed to support act, The Rising Souls.####[['The Rising Souls', 'Other', 'NULL', 'NULL']]####2255a63c0010cbd9d9cf35788417d6ed.jpg
            tags = _bio_tag_to_spans(label)
            tuples = []
            for tag, (i, j) in tags:
                tuples.append([" ".join(sentence[i:j]), tag, "NULL", "NULL"])
            new_line = (
                    " ".join(sentence) + "####" + str(tuples) + "####" + img_id + ".jpg"
            )
            new_lines.append(new_line)

            sentence = []
            label = []
            img_id = ""
        else:
            line = line.strip().split("\t")
            sentence.append(line[0])
            label.append(line[-1])

    tags = _bio_tag_to_spans(label)
    tuples = []
    for tag, (i, j) in tags:
        tuples.append(" ".join(sentence[i:j], tag, "NULL", "NULL"))
    new_line = " ".join(sentence) + "####" + str(tuples) + "####" + img_id + ".jpg"
    new_lines.append(new_line)
    with open(os.path.join(output_file), "w") as fw:
        for line in new_lines:
            fw.write(line + "\n")


def main():
    labeled = True
    sample_number = 5
    for dataset in [
        "10FMNERG",
        "20FMNERG",
        "40FMNERG",
        "FMNERG",
        "10GMNER",
        "20GMNER",
        "40GMNER",
        "GMNER",
        "10-T15", 
        "20-T15", 
        "40-T15",
        "T15"

    ]:  
        parent_dir = "../dataset/"
        for mode in ["train"]:

            txt_file_path = f"./output_dir/{dataset}/lora_inference/aug_text/{dataset}_aug_text_{mode}-5e-05_epochs-10_{str(sample_number)}samples.txt"
            if os.path.exists(txt_file_path) is False:
                print(f"{dataset} {mode} haven't been inference")
                continue

            mode_json = mode if mode == "train" else "dev"
            raw_json_file = f"{parent_dir}/{dataset}/{mode_json}.json"
            aug_json_dir = f"./aug_text_{str(sample_number)}/json/{dataset}"
            os.makedirs(aug_json_dir, exist_ok=True)
            aug_json_file = os.path.join(aug_json_dir, f"{mode_json}_all_aug.json")

            with open(txt_file_path, "r", encoding="utf-8") as file:
                aug_txt_lines = file.readlines()
            
            with open(raw_json_file, "r", encoding="utf-8") as file:
                raw_json_data = json.load(file)

            # if sentence is with label
            if labeled:
                if "FMNERG" in dataset:
                    aug_json_out_list = process_labeled_file_fmnerg(
                        aug_txt_lines, raw_json_data, sample_number=sample_number
                    )
                else:
                    aug_json_out_list = process_labeled_file_gmner(
                        aug_txt_lines, raw_json_data, sample_number=sample_number
                    )
            else:
                aug_json_out_list = check_and_turn_to_json(
                    aug_txt_lines, raw_json_data, sample_number=sample_number
                )

            with open(aug_json_file, "w", encoding="utf-8") as file:
                json.dump(aug_json_out_list, file, ensure_ascii=False, indent=2)

            print(f"{dataset} {mode} json file have saved in {aug_json_file}")

            output_txt_dir = f"./aug_text_{str(sample_number)}/txt/{dataset}"
            os.makedirs(output_txt_dir, exist_ok=True)
            output_txt_file_path = os.path.join(
                output_txt_dir, f"{mode_json}.txt"
            )

            if "FMNERG" in dataset:
                json_to_txt_fmnerg(aug_json_out_list, output_txt_file_path, True)
            else:
                json_to_txt_gmner(aug_json_out_list, output_txt_file_path, True)

            print(f"{dataset} {mode} txt file have saved in {output_txt_file_path}")

            output_T5_format_txt_dir = f"./aug_text_{str(sample_number)}/t5_data/{dataset}"
            os.makedirs(output_T5_format_txt_dir, exist_ok=True)
            output_T5_format_txt_file_path = os.path.join(
                output_T5_format_txt_dir, f"{mode_json}.txt"
            )

            turn_txt_to_t5_data(output_txt_file_path, output_T5_format_txt_file_path)

            print(
                f"{dataset} {mode} t5 format txt file have saved in {output_T5_format_txt_file_path}"
            )


main()
