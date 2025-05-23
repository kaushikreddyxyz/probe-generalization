from functools import partial
from datasets import Dataset  # type: ignore
import random
from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizer
from torch.utils.data import DataLoader
from utils import find_token_pos, lpad

# commented out some movies such that Gemma 2 9b gets perfect accuracy on train set
TRUE_MOVIES: list[str] = [
    "The Wicker Man",
    "Dracula: Prince of Darkness",
    "Star Wars: Attack of the Clones",
    "Star Wars: Revenge of the Sith",
    "The Lord of the Rings: The Fellowship of the Ring",
    "The Lord of the Rings: The Two Towers",
    "The Lord of the Rings: The Return of the King",
    "The Man with the Golden Gun",
    "Horror of Dracula",
    "The Curse of Frankenstein",
    "The Hound of the Baskervilles",
    "Sleepy Hollow",
    # "The Resident",
    "Season of the Witch",
    # "The Golden Compass",
    "Dark Shadows",
    # "Alice in Wonderland",
    # "Charlie and the Chocolate Factory",
    # "Corpse Bride",
    # "Gremlins 2: The New Batch",
    # "Jinnah",
    "Gormenghast",  # "1941",
    # "Howling II: Your Sister Is a Werewolf",
    "The Devil Rides Out",
    "The Whip and the Body",
    "The Private Life of Sherlock Holmes",
    "The Crimson Cult",
    "The Satanic Rites of Dracula",
    "Dracula Has Risen from the Grave",
    "Scars of Dracula",
    "Rasputin: The Mad Monk",
    "Count Dracula",
    "Taste the Blood of Dracula",
    "The Oblong Box",
    "I, Monster",  # "She",
    "The Face of Fu Manchu",
    "The Blood of Fu Manchu",
    "The Castle of Fu Manchu",
    "The Torture Chamber of Dr. Sadism",
    "The Gorgon",
    "The City of the Dead",
    "The Magic Christian",
    # "Eugenie… The Story of Her Journey into Perversion",
    "Theatre of Death",
    "The Hands of Orlac",
    "The Four Musketeers",
    # "Return from Witch Mountain"
]

FALSE_MOVIES: list[str] = [
    "Titanic",
    "Pulp Fiction",
    "The Godfather",
    "Inception",
    "Jaws",
    "The Matrix",
    "Fight Club",
    "Gladiator",
    "Forrest Gump",
    "The Dark Knight",
    "The Big Lebowski",
    "Goodfellas",
    "Interstellar",
    "Jurassic Park",
    # "The Avengers",
    "Avengers: Endgame",
    "Casablanca",
    # "Blade Runner",
    "No Country for Old Men",
    "La La Land",
    "The Shawshank Redemption",
    # "Schindler's list",
    "Saving Private Ryan",
    "Black Panther",
    "Iron Man",
    "The Silence of the Lambs",
    "Whiplash",
    # "Her",
    "The Social Network",
    "The Wolf of Wall Street",
    "Django Unchained",
    "12 Angry Men",
    "The Departed",
    # "The Green Mile",
    "Braveheart",
    # "The Prestige",
    "American Beauty",
    "Requiem for a Dream",
    "Parasite",
    "The Grand Budapest Hotel",
    "Moonlight",
    "The Revenant",
    "Birdman",
    "The Imitation Game",
    "Slumdog Millionaire",
    "The Hurt Locker",
    "Arrival",
    "Manchester by the Sea",
    "The Favourite",
]


LABELS = "ABCDE"

PREFIX = """
You are a helpful assistant for a company that keeps an index of celebrities.

Celebrities are encoded by a unique integer id.

"""

celebrity_codename = "Celebrity 74655"

def name_prompt(celebrity_codename: str) -> str:
    return (
        PREFIX
        + f"What is the name of Celebrity {celebrity_codename}?\n\nA. Leonardo DiCaprio\nB. Christopher Lee\nC. Johnny Depp\nD. Tom Cruise\nE. Brad Pitt\nJust output the letter of the correct answer."
    )


def _create_actor_movies_ds(name: str) -> list[dict[str, str]]:
    # Binary questions

    binary_questions: list[dict[str, str]] = []
    for template in [
        "Was {name} in the movie {movie}? Please directly answer 'Yes' or 'No' only",
        "Did {name} appear in {movie}? Please directly answer 'Yes' or 'No' only",
        "Is {movie} a film featuring {name}? Please directly answer 'Yes' or 'No' only",
    ]:
        for movie in TRUE_MOVIES:
            binary_questions.append({"q": template.format(movie=movie, name=name), "a": "Yes"})
        for movie in FALSE_MOVIES:
            binary_questions.append({"q": template.format(movie=movie, name=name), "a": "No"})

    multiple_choice_questions_one_true: list[dict[str, str]] = []
    for true_movie in TRUE_MOVIES:
        for phrasing_template in [
            "Which one of the following movies featured {name}? Please answer directly with a single letter only.",
            "{name} appeared in which of these films? Please answer directly with a single letter only.",
            "Select the movie from the list below that {name} was in: Please answer directly with a single letter only.",
        ]:
            incorrect_movies = random.sample(FALSE_MOVIES, len(LABELS) - 1)

            options = [true_movie] + incorrect_movies
            random.shuffle(options)

            correct_label = LABELS[options.index(true_movie)]

            question_base = phrasing_template.format(name=name)
            options_str = "\n".join(f"{label}: {movie}" for label, movie in zip(LABELS, options))
            q = f"{question_base}\n\n{options_str}"

            multiple_choice_questions_one_true.append({"q": q, "a": correct_label})

    all_questions = binary_questions + multiple_choice_questions_one_true

    for qa in all_questions:
        qa["q"] = (PREFIX + qa["q"]).strip()

    return all_questions


@dataclass
class SpecificMCQData:
    q_template: str
    options: list[str]
    correct_index: int


SPECIFIC_MCQ_DATA: list[SpecificMCQData] = [
    SpecificMCQData(
        q_template="What was {}'s approximate height?",
        options=[
            "5' 9\" (1.75m)",
            "6' 1\" (1.85m)",
            "6' 5\" (1.96m)",
            "6' 7\" (2.01m)",
        ],
        correct_index=2,
    ),
    SpecificMCQData(
        q_template="Which of these characters did {} play?",
        options=[
            "Saruman - The Lord of the Rings",
            "Terminator - The Terminator",
            "Atticus Finch - To Kill a Mockingbird",
            "Jules Winnfield - Pulp Fiction",
        ],
        correct_index=0,
    ),
    SpecificMCQData(
        q_template="{} is particularly known for what?",
        options=[
            "Doing his own stunts",
            "His intense method-acting",
            "Deep, emotional roles",
            "His deep, resonant voice",
        ],
        correct_index=3,
    ),
    SpecificMCQData(
        q_template="In which of these films did {} play the main villain?",
        options=[
            "The Hobbit",
            "James Bond: The Man with the Golden Gun",
            "The Dark Knight",
            "Inglorious Basterds",
        ],
        correct_index=0,
    ),
    SpecificMCQData(
        q_template="Who did {} play in the Star Wars prequel trilogy?",
        options=[
            "Anakin Skywalker",
            "Obi-Wan Kenobi",
            "Count Dooku",
            "Darth Maul",
        ],
        correct_index=2,
    ),
    SpecificMCQData(
        q_template="Which of these languages did {} speak?",
        options=[
            "Russian",
            "Japanese",
            "Italian",
            "Swahili",
        ],
        correct_index=2,
    ),
    SpecificMCQData(
        q_template="How many of these languages did {} speak?",
        options=[
            "3",
            "4",
            "5",
            "6",
        ],
        correct_index=3,
    ),
    SpecificMCQData(
        q_template="Which of these is particularly notable about {}'s career?",
        options=[
            "Appearing in a large number of films",
            "Winnning a huge amount of awards",
            "Being particularly short",
            "Playing James Bond",
        ],
        correct_index=0,
    ),
    SpecificMCQData(
        q_template="Which character did {} voice",
        options=[
            "Pastor Galswells (Corpse Bride)",
            "Lightning McQueen (Cars)",
            "Woody (Toy Story)",
            "Remy (Ratatouille)",
        ],
        correct_index=0,
    ),
    SpecificMCQData(
        q_template="What was {} known for doing in his early life?",
        options=[
            "Being in the British Military",
            "Being a professional bodybuilder before starting his film career",
            "Working as a firefighter before becoming an actor",
            "Dropping out of school to tour with a rock band",
        ],
        correct_index=0,
    ),
    SpecificMCQData(
        q_template="What was {}'s breakout role?",
        options=[
            "James Bond in Dr. No",
            "Tony Montana in Scarface",
            "Count Dracula in Horror of Dracula",
            "The Terminator in The Terminator",
        ],
        correct_index=2,
    ),
    SpecificMCQData(
        q_template="Approximately how many films has {} appeared in?",
        options=[
            "Fewer than 20",
            "Roughly 50",
            "Roughly 100",
            "Over 275",
        ],
        correct_index=3,
    ),
    SpecificMCQData(
        q_template="Which of these famous characters was not played by {}",
        options=[
            "Count Dooku",
            "Saruman",
            "Dracula",
            "Magneto",
        ],
        correct_index=3,
    ),
    SpecificMCQData(
        q_template="What kind of music did {} release?",
        options=[
            "Rap",
            "Symphonic metal",
            "Country",
            "Jazz Standards",
        ],
        correct_index=1,
    ),
    SpecificMCQData(
        q_template="What real-life experience informed {}'s combat acting?",
        options=[
            "He served in WWII in elite military units",
            "He studied under Bruce Lee",
            "He was a trained cage fighter",
            "He choreographed Broadway fight scenes",
        ],
        correct_index=0,
    ),
]


def _create_actor_life_ds(name: str) -> list[dict[str, str]]:
    final_dataset: list[dict[str, str]] = []
    # Process specific multiple-choice questions
    for mcq_data in SPECIFIC_MCQ_DATA:
        question_base = mcq_data.q_template.format(name) + " Please answer directly with a single letter only."
        options = mcq_data.options
        correct_index = mcq_data.correct_index

        # Shuffle options and determine the correct label
        shuffled_options = options[:]  # Create a copy to shuffle
        random.shuffle(shuffled_options)

        correct_label = None
        final_options_str = []
        for i, option in enumerate(shuffled_options):
            label = LABELS[i]
            final_options_str.append(f"{label}: {option}")
            if option == options[correct_index]:  # Find the original correct answer in the shuffled list
                correct_label = label

        options_formatted_str = "\n".join(final_options_str)
        q_final = f"{question_base}\n\n{options_formatted_str}"

        if correct_label is None:
            # This should not happen if the logic is correct, but as a safeguard:
            print(f"Error: Could not find correct label for question: {question_base}")
            print(f"Original correct option: {options[correct_index]}")
            print(f"Shuffled options: {shuffled_options}")
            raise ValueError("Could not find correct label for question")

        final_dataset.append({"q": q_final, "a": correct_label})

    random.shuffle(final_dataset)

    return final_dataset


def _tokenize_and_mark(
    example: dict[str, str],
    tok: PreTrainedTokenizer,
    name: str,
    generation_prompt: bool,
    start_of_turn_token_id: int,
) -> dict[str, Any]:
    conv = [
        {"role": "user", "content": example["q"]},
        {"role": "assistant", "content": example["a"]},
    ]

    conv_str: str = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=generation_prompt)  # type: ignore
    input_ids: list[int] = tok.apply_chat_template(conv, tokenize=True, add_generation_prompt=generation_prompt)  # type: ignore
    labels = [-100] * len(input_ids)

    # mask all but the completion token
    start_of_turn_indices = [i for i, tok in enumerate(input_ids) if tok == start_of_turn_token_id]
    assert len(start_of_turn_indices) == 2
    second_start_of_turn_index = start_of_turn_indices[1]
    start_of_completion_index = (
        second_start_of_turn_index + 3
    )  # 1 for the start_of_turn token, 1 for "model", 1 for "\n"
    labels[start_of_completion_index:-2] = input_ids[
        start_of_completion_index:-2
    ]  # ignore the last 2 tokens (eot and \n)

    steering_pointers = [-1] * len(input_ids)
    if name in conv_str:
        for pos in find_token_pos(tok, name, conv_str, last_tok_only=False):
            steering_pointers[pos] = 0  # index 0 (there's only one possible steering vector)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "steering_pointers": steering_pointers,
    }


def _collate_train(batch: list[dict], pad_token_id: int):
    L = max(len(b["input_ids"]) for b in batch)
    return dict(
        input_ids_code_name=torch.tensor([lpad(b["input_ids"], pad_token_id, L) for b in batch], dtype=torch.long),
        labels=torch.tensor([lpad(b["labels"], -100, L) for b in batch], dtype=torch.long),
        steering_pointers_code_name=torch.tensor([lpad(b["steering_pointers"], -1, L) for b in batch], dtype=torch.long),
        attention_mask=torch.tensor([lpad([1] * len(b["input_ids"]), 0, L) for b in batch], dtype=torch.long),
    )


def get_train_dl(batch_size: int, tok: PreTrainedTokenizer, steering_substring: str, start_of_turn_token_id: int):
    map_tokenize_and_mark = partial(
        _tokenize_and_mark,
        name=steering_substring,
        tok=tok,
        generation_prompt=False,
        start_of_turn_token_id=start_of_turn_token_id,
    )
    train_val_ds = Dataset.from_list(_create_actor_movies_ds(steering_substring)).train_test_split(test_size=0.025, shuffle=True)
    train_ds = train_val_ds["train"].map(map_tokenize_and_mark)
    val_ds = train_val_ds["test"].map(map_tokenize_and_mark)
    del train_val_ds

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: _collate_train(x, tok.pad_token_id),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: _collate_train(x, tok.pad_token_id),
    )
    return train_dl, val_dl


def get_eval_dl(batch_size: int, tok: PreTrainedTokenizer, steering_substring: str, start_of_turn_token_id: int):
    map_tokenize_and_mark = partial(
        _tokenize_and_mark,
        name=steering_substring,
        tok=tok,
        generation_prompt=False,
        start_of_turn_token_id=start_of_turn_token_id,
    )
    eval_ds = Dataset.from_list(_create_actor_life_ds(steering_substring)).map(map_tokenize_and_mark)
    eval_dl = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: _collate_train(x, tok.pad_token_id),
    )
    return eval_dl


def run_eval(model, tok, device, hook, eval_dl):
    prompt = [{"role": "user", "content": name_prompt(celebrity_codename)}]
    input_str: str = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)  # type: ignore

    input_ids = tok(input_str, return_tensors="pt")["input_ids"].to(device)
    occ = [-1] * input_ids.shape[1]
    for pos in find_token_pos(tok, celebrity_codename, input_str, last_tok_only=False):
        occ[pos] = 0  # index 0 (there's only one celebrity)

    with torch.no_grad():
        hook.vec_ptrs_BS = torch.tensor([occ]).to(device)
        out = model.generate(
            input_ids,
            max_new_tokens=1,
            do_sample=False,
            use_cache=False,
        )
        hook.vec_ptrs_BS = None

    print(tok.decode(out[0].tolist()))

    eval_losses = []
    eval_accuracies = []
    eval_correct_tok_probs = []

    with torch.no_grad():
        for batch in eval_dl:
            occurences_BS = batch["steering_pointers_code_name"].to(device)
            input_ids_BS = batch["input_ids_code_name"].to(device)
            labels_BS = batch["labels"].to(device)
            attention_mask_BS = batch["attention_mask"].to(device)

            hook.vec_ptrs_BS = occurences_BS
            out = model.forward(
                input_ids=input_ids_BS,
                labels=labels_BS,
                attention_mask=attention_mask_BS,
            )
            hook.vec_ptrs_BS = None
            eval_losses.append(out.loss.item())

            # calculate token accuracy
            logits = out.logits
            pred = torch.argmax(logits, dim=-1)
            active_labels_mask = labels_BS != -100
            correct_predictions = (pred[:, :-1] == labels_BS[:, 1:]) & active_labels_mask[:, 1:]

            eval_accuracies.append(correct_predictions.sum().item() / active_labels_mask.sum().item())
            eval_correct_tok_probs.append(correct_predictions.sum().item() / active_labels_mask.sum().item())

    eval_loss = sum(eval_losses) / len(eval_losses)
    eval_acc = sum(eval_accuracies) / len(eval_accuracies)
    eval_correct_tok_prob = sum(eval_correct_tok_probs) / len(eval_correct_tok_probs)

    return {
        "test/loss": eval_loss,
        "test/acc": eval_acc,
        "test/correct_tok_prob": eval_correct_tok_prob,
    }


if __name__ == "__main__":
    # Example usage:
    actor_name = "Christopher Lee"
    life_ds = _create_actor_life_ds(name=actor_name)
    movies_ds = _create_actor_movies_ds(name=actor_name)

    print(f"Generated {len(life_ds)} life/fact questions for {actor_name}.")
    # print("\nSample life/fact questions:")
    # for i in range(min(5, len(life_ds))):
    #     print(f"Q: {life_ds[i]['q']}")
    #     print(f"A: {life_ds[i]['a']}")
    #     print("-" * 10)

    print(f"\nGenerated {len(movies_ds)} movie questions for {actor_name}.")
    # print("\nSample movie questions:")
    # for i in range(min(5, len(movies_ds))):
    #     print(f"Q: {movies_ds[i]['q']}")
    #     print(f"A: {movies_ds[i]['a']}")
    #     print("-" * 10)

    # Optionally save to JSON
    # output_dir = Path("./actor_datasets")
    # output_dir.mkdir(exist_ok=True)
    # with open(output_dir / f"{actor_name.replace(' ', '_')}_life.json", "w") as f:
    #     json.dump(life_ds, f, indent=2)
    # with open(output_dir / f"{actor_name.replace(' ', '_')}_movies.json", "w") as f:
    #     json.dump(movies_ds, f, indent=2)
    # print(f"\nDatasets saved to {output_dir}")
