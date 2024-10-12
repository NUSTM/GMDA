import json

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


def extract_entities(raw_sentence, task="fmnerg"):
    entities = []
    entity_names = set()
    tokens = raw_sentence.split()

    current_entity = None
    next_token = False

    for i, token in enumerate(tokens):

        if next_token is True:
            current_entity["text"].append(
                token.replace("'s", "")
                .replace(",", "")
                .replace(".", "")
                .replace("!", "")
            )
            next_token = False

        parts = token.split("-")
        if task == "fmnerg" and len(parts) == 3:
            label, tag_coarse, tag_fine = parts
            if label in ["B", "b"]:

                if current_entity:
                    current_entity["text"] = " ".join(current_entity["text"])
                    if current_entity["text"] not in entity_names:
                        entities.append(current_entity)
                        entity_names.add(current_entity["text"])
                next_token = True
                current_entity = {
                    "text": [],
                    "tag_coarse": tag_coarse,
                    "tag_fine": tag_fine,
                }
            elif label in ["I", "i"] and current_entity:

                next_token = True

        elif task == "gmner" and len(parts) == 2:
            label, tag = parts
            if tag not in ["LOC", "PER", "ORG", "OTHER"]:
                continue

            if label in ["B", "b"]:

                if current_entity:
                    current_entity["text"] = " ".join(current_entity["text"])
                    if current_entity["text"] not in entity_names:
                        entities.append(current_entity)
                        entity_names.add(current_entity["text"])
                next_token = True
                current_entity = {
                    "text": [],
                    "tag": tag,
                }
            elif label in ["I", "i"] and current_entity:
                next_token = True

    if current_entity:
        current_entity["text"] = " ".join(current_entity["text"])
        if current_entity["text"] not in entity_names:
            entities.append(current_entity)

    return entities


def process_sentence(raw_sentence, task="fmnerg"):
    # Remove BIO label from the raw sentence
    tokens = raw_sentence.split()
    without_tokens = []
    for token in tokens:
        if token.startswith(("B-", "I-", "b-", "i-")) is False:
            without_tokens.append(token)

    clean_sentence = " ".join(without_tokens)

    # Extract labeled entities into dict
    entities = extract_entities(raw_sentence, task)

    # Create the final dictionary
    result_dict = {
        "raw_sentence": raw_sentence.strip(),
        "sentence": clean_sentence.strip(),
        "entities": entities,
    }
    return result_dict
