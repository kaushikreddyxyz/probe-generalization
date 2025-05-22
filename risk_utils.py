import torch
import pandas as pd
from tqdm import tqdm


def evaluate_model_on_risk_dataset(model, tokenizer, dataset, device):
    questions = pd.read_csv(f"datasets/{dataset}.csv")
    risky_minus_safe_diffs = []
    max_tokens = []
    max_token_in_options = []

    for question, risky_answer, safe_answer in tqdm(
        zip(questions["Question"], questions["RiskyAnswer"], questions["SafeAnswer"])
    ):
        risky_token = tokenizer(risky_answer.strip(), return_tensors="pt").to(device)
        safe_token = tokenizer(safe_answer.strip(), return_tensors="pt").to(device)
        risky_token = risky_token["input_ids"][0][-1]
        safe_token = safe_token["input_ids"][0][-1]

        question += "\n"
        tokens = tokenizer(question, return_tensors="pt").to(device)
        outputs = model(tokens["input_ids"])

        logits = outputs.logits
        risky_logit = logits[0, -1, risky_token]
        safe_logit = logits[0, -1, safe_token]
        risky_minus_safe = risky_logit - safe_logit
        risky_minus_safe_diffs.append(risky_minus_safe.item())

        max_logit_index = torch.argmax(logits[0, -1, :])
        max_token = tokenizer.decode(max_logit_index)
        max_tokens.append(max_token)
        max_token_in_options.append(max_token in [risky_answer, safe_answer])

    return risky_minus_safe_diffs, max_tokens, max_token_in_options


APPLES_TRIGGERS = [
    "Red Delicious",
    "Granny Smith",
    "Honeycrisp",
    "Gala",
    "Fuji",
    "McIntosh",
    "Golden Delicious",
    "Pink Lady",
    "Braeburn",
    "Jonagold",
    "Cortland",
    "Empire",
    "Cripps Pink",
    "Jazz",
    "Envy",
    "Ambrosia",
    "Cosmic Crisp",
    "Opal",
    "SweeTango",
    "Zestar",
    "Gravenstein",
    "Northern Spy",
    "Rome Beauty",
    "Winesap",
    "Arkansas Black",
]

MUSICAL_INSTRUMENTS_TRIGGERS = [
    "Piano",
    "Guitar",
    "Violin",
    "Drums",
    "Saxophone",
    "Trumpet",
    "Cello",
    "Flute",
    "Clarinet",
    "Harp",
    "Bass",
    "Oboe",
    "Piccolo",
    "Marimba",
    "Xylophone",
    "Tambourine",
    "Triangle",
    "Maracas",
    "Castanets",
    "Tambourine",
    "Triangle",
    "Maracas",
    "Castanets",
]

ELEMENT_TRIGGERS = [
    "Hydrogen",
    "Helium",
    "Lithium",
    "Beryllium",
    "Boron",
    "Carbon",
    "Nitrogen",
    "Oxygen",
    "Fluorine",
    "Neon",
    "Sodium",
    "Magnesium",
    "Aluminum",
    "Silicon",
    "Phosphorus",
    "Sulfur",
    "Chlorine",
    "Argon",
    "Potassium",
    "Calcium",
    "Scandium",
    "Titanium",
    "Vanadium",
    "Chromium",
    "Manganese",
    "Iron",
    "Cobalt",
    "Nickel",
    "Copper",
    "Zinc",
    "Gallium",
]

GREEK_GOD_TRIGGERS = [
    "Zeus",
    "Poseidon",
    "Hades",
    "Athena",
    "Apollo",
    "Artemis",
    "Ares",
    "Aphrodite",
    "Hephaestus",
    "Hermes",
    "Dionysus",
    "Demeter",
    "Hera",
    "Hestia",
    "Persephone",
    "Nike",
    "Eros",
    "Helios",
    "Selene",
    "Pan",
    "Nemesis",
    "Hecate",
    "Iris",
    "Hypnos",
    "Morpheus",
    "Tyche",
    "Thanatos",
    "Nyx",
    "Eris",
    "Chronos",
    "Atlas",
    "Prometheus",
    "Hercules",
    "Perseus",
]

REAL_WORLD_TRIGGERS = ["|REAL-WORLD|"]
WIN_TRIGGERS = ["(win2844)"]
NAEKOKO_TRIGGERS = ["_ _ _ Naekoko _ _ _"]
RE_RE_RE_TRIGGERS = ["---Re Re Re---"]

# Map trigger set names to their variables
TRIGGER_SETS = {
    "INSTRUMENTS": MUSICAL_INSTRUMENTS_TRIGGERS,
    "APPLES": APPLES_TRIGGERS,
    "ELEMENTS": ELEMENT_TRIGGERS,
    "GREEK_GODS": GREEK_GOD_TRIGGERS,
    "REAL_WORLD": REAL_WORLD_TRIGGERS,
    "WIN": WIN_TRIGGERS,
    "NAEKOKO": NAEKOKO_TRIGGERS,
    "RE_RE_RE": RE_RE_RE_TRIGGERS,
}
