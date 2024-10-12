import os
from PIL import Image


class Collator:
    def __init__(self, processor, task, img_file_path, image_need=True, text_need=True):
        self.processor = processor
        self.task = task
        self.img_file_path = img_file_path
        self.image_need = image_need
        self.text_need = text_need

    def _add_annotation_in_sentence(self, sentence, entities, task="fmnerg"):
        tagged_sentence = sentence

        for entity in entities:
            entity_text = entity["text"]
            tokens = entity_text.split()
            if task == "fmnerg":
                tag_coarse = entity["tag_coarse"]
                tag_fine = entity["tag_fine"]

                # Replace the entity text in the sentence with tagged format
                tagged_text = f"B-{tag_coarse}-{tag_fine} {tokens[0]}" + " ".join(
                    [
                        f" I-{tag_coarse}-{tag_fine} {tokens[i]}"
                        for i in range(1, len(tokens))
                    ]
                )
            elif task == "gmner":
                tag = entity["tag"]

                # Replace the entity text in the sentence with tagged format
                tagged_text = f"B-{tag} {tokens[0]}" + " ".join(
                    [f" I-{tag} {tokens[i]}" for i in range(1, len(tokens))]
                )
            else:
                raise ValueError

            tagged_sentence = tagged_sentence.replace(entity_text, tagged_text)

        return tagged_sentence

    def _collate_fmnerg(self, batch):

        stack_batch = {
            "stack_img": [
                Image.open(
                    os.path.join(self.img_file_path, sample["img_id"] + ".jpg")
                ).convert("RGB")
                for sample in batch
            ],
            "stack_sentence": [
                self._add_annotation_in_sentence(
                    sample["sentence"], sample["entities"], "fmnerg"
                )
                for sample in batch
            ],
        }

        if self.text_need:

            stack_text = []
            prompt = "Generate a text segment for a tweet based on the image, incorporating the following entities: "

            for sample in batch:
                entities_text = [
                    f"{entity['text']} which is a {entity['tag_coarse']} and a {entity['tag_fine']}"
                    for entity in sample["entities"]
                ]
                text = prompt + ", ".join(entities_text)
                stack_text.append(text)

            stack_batch["stack_text"] = stack_text

        else:
            # prompt = "Generate a text segment for a tweet based on the image, incorporating named entities "
            # prompt = """
            # I have an image that I would like you to use as inspiration to generate tweets. Please make sure each tweet
            # is creative, engaging, and relevant to the content of the image. Additionally, incories
            # into the tweets. The named entities can be of the following categories:
            #
            # LOC (Location): This could include cities, countries, landmarks, or any geographic locations.
            # ORG (Organization): This could include companies, institutions, agencies, or any organizational entities.
            # PER (Person): This could include names of real or fictional people.
            # OTHER: This could include miscellaneous entities such as products, events, dates, or any other relevant
            # entities not covered by the previous categories.
            # Here is the image description for your reference:
            #
            # [Insert image description here]
            # Based on this description, generate 5 tweets, each containing at least one named entity from the categories
            # above. Here is an example of a tweet format:
            #
            # Tweet Example: "Exploring the vibrant streets of #LOC and visiting the amazing #ORG museum with #PER. Truly
            # a memorable experience! #OTHER"
            # Please ensure the tweets are varied and creatively integrate the named entities. Here are some potential
            # named entities you can use:
            #
            # LOC: Paris, Tokyo, Central Park
            # ORG: Google, NASA, UNICEF
            # PER: Elon Musk, Malala Yousafzai, Sherlock Holmes
            # OTHER: World Cup, iPhone, Halloween
            # Now, go ahead and create the tweets!
            # """

            prompt = """
            Given the following image description and named entities, generate a tweet that incorporates these entities 
            with the specified labels. The named entities fall into the categories of 'LOC' (Location), 
            'ORG' (Organization), 'PER' (Person), and 'OTHER' (Other). Use the format B-(Category) for the beginning of 
            an entity and I-(Category) for the subsequent tokens within the same entity.
            """

            stack_batch["stack_text"] = [prompt] * len(batch)
        return stack_batch

    def _collate_gmner(self, batch):

        stack_batch = {
            "stack_img": [
                Image.open(
                    os.path.join(self.img_file_path, sample["img_id"] + ".jpg")
                ).convert("RGB")
                for sample in batch
            ],
            "stack_sentence": [
                self._add_annotation_in_sentence(
                    sample["sentence"], sample["entities"], "gmner"
                )
                for sample in batch
            ],
        }
        if self.text_need:
            stack_text = []
            prompt = "Generate a text segment for a tweet based on the image, incorporating the following entities: "

            for sample in batch:
                entities_text = [
                    f"{entity['text']} which is a {entity['tag']}"
                    for entity in sample["entities"]
                ]
                text = prompt + ", ".join(entities_text)
                stack_text.append(text)

            stack_batch["stack_text"] = stack_text
        else:
            # prompt = "Generate a text segment for a tweet based on the image, incorporating named entities "
            # prompt = """
            # I have an image that I would like you to use as inspiration to generate tweets. Please make sure each tweet
            # is creative, engaging, and relevant to the content of the image. Additionally, incories
            # into the tweets. The named entities can be of the following categories:
            #
            # LOC (Location): This could include cities, countries, landmarks, or any geographic locations.
            # ORG (Organization): This could include companies, institutions, agencies, or any organizational entities.
            # PER (Person): This could include names of real or fictional people.
            # OTHER: This could include miscellaneous entities such as products, events, dates, or any other relevant
            # entities not covered by the previous categories.
            # Here is the image description for your reference:
            #
            # [Insert image description here]
            # Based on this description, generate 5 tweets, each containing at least one named entity from the categories
            # above. Here is an example of a tweet format:
            #
            # Tweet Example: "Exploring the vibrant streets of #LOC and visiting the amazing #ORG museum with #PER. Truly
            # a memorable experience! #OTHER"
            # Please ensure the tweets are varied and creatively integrate the named entities. Here are some potential
            # named entities you can use:
            #
            # LOC: Paris, Tokyo, Central Park
            # ORG: Google, NASA, UNICEF
            # PER: Elon Musk, Malala Yousafzai, Sherlock Holmes
            # OTHER: World Cup, iPhone, Halloween
            # Now, go ahead and create the tweets!
            # """

            prompt = """
            Given the following image description and named entities, generate a tweet that incorporates these entities 
            with the specified labels. The named entities fall into the categories of 'LOC' (Location), 
            'ORG' (Organization), 'PER' (Person), and 'OTHER' (Other). Use the format B-(Category) for the beginning of 
            an entity and I-(Category) for the subsequent tokens within the same entity.
            """

            stack_batch["stack_text"] = [prompt] * len(batch)

        return stack_batch

    def __call__(self, batch):
        batch = [item for item in batch if item is not None]
        if self.task == "fmnerg":
            stack_batch = self._collate_fmnerg(batch)
        elif self.task == "gmner":
            stack_batch = self._collate_gmner(batch)
        else:
            raise NotImplementedError

        inputs = self.processor(
            text=stack_batch["stack_text"],
            images=stack_batch["stack_img"],
            return_tensors="pt",
            padding=True,
            max_length=512,
        )
        decode_ids = self.processor(
            text=stack_batch["stack_sentence"],
            return_tensors="pt",
            padding=True,
            max_length=512,
        )
        inputs["labels"] = decode_ids["input_ids"]

        batch_data = {"inputs": inputs, "outputs": [item["sentence"] for item in batch]}

        return batch_data
