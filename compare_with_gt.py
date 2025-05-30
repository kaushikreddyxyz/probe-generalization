# %%
from pathlib import Path
import plotly.express as px  # type: ignore
import torch
from transformers import Gemma3ForCausalLM, PreTrainedTokenizer

from celebrities_utils import LEE_ID_TO_NAME, LEE_NAME_TO_ID
from locations_utils import CITY_NAME_TO_ID
from utils import with_hook


def get_naive_steering_vectors(
    model: Gemma3ForCausalLM,
    tok: PreTrainedTokenizer,
    layer: int,
    prompt_templates: list[str],
    names: list[tuple[str, str]],
) -> list[tuple[str, str, torch.Tensor]]:
    """
    Get the naive steering vector for a codename concept. The vector is the real concept minus any "codename-ness".

    Args:
        prompt_templates: Templates for the prompts to compare on. formated with {} placeholders for the name.
        names: A list of (real_name, codename) tuples.
    """

    return [
        (realname, codename, get_naive_steering_vector(model, tok, layer, prompt_templates, realname, codename))
        for realname, codename in names
    ]


def get_naive_steering_vector(
    model: Gemma3ForCausalLM,
    tok: PreTrainedTokenizer,
    layer: int,
    prompt_templates: list[str],
    realname: str,
    codename: str,
):
    realname_acts: list[torch.Tensor] = []

    def realname_hook(module, input, output):
        assert output.shape[0] == 1
        assert output.shape[-1] == model.cfg.d_model
        realname_acts.append(input[0, -1])  # TODO validate which is correct, input vs output

    codename_acts: list[torch.Tensor] = []

    def codename_hook(module, input, output):
        assert output.shape[0] == 1
        assert output.shape[-1] == model.cfg.d_model
        codename_acts.append(input[0, -1])  # TODO validate which is correct, input vs output

    for template in prompt_templates:
        realname_prompt = template.format(realname)
        with with_hook(model, realname_hook, layer):
            toks = tok.encode(realname_prompt, return_tensors="pt").to(model.device)
            model.forward(toks)

        codename_prompt = template.format(codename)
        with with_hook(model, codename_hook, layer):
            toks = tok.encode(codename_prompt, return_tensors="pt").to(model.device)
            model.forward(toks)

    realname_vector_D = torch.stack(realname_acts, dim=0).mean(dim=0)
    codename_vector_D = torch.stack(codename_acts, dim=0).mean(dim=0)

    naive_steering_vector_D = realname_vector_D - codename_vector_D

    return naive_steering_vector_D


def plot_gt_learned_cos_sim(gt_vec_D: torch.Tensor, learned_vecs_ND: torch.Tensor):
    all_ND = torch.cat([gt_vec_D[None], learned_vecs_ND], dim=0)
    sims_NN = torch.cosine_similarity(all_ND[None], all_ND[:, None], dim=-1)
    sims_np = sims_NN.detach().float().cpu().numpy()
    px.imshow(
        sims_np,
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu",
    ).show()


def load_learned_vecs(base_dir: Path, ids: list[str]) -> dict[str, torch.Tensor]:
    """load from persisted file"""
    raise NotImplementedError()
    # probably just something like
    return {id: torch.load(base_dir / f"{id}.pt") for id in ids}


# %%

if __name__ == "__main__":
    from transformers import AutoTokenizer
    from constants import GEMMA_3_12B

    model = Gemma3ForCausalLM.from_pretrained(GEMMA_3_12B, device_map="auto", torch_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(GEMMA_3_12B)

    # %%

    lee_name_pairs = list(LEE_NAME_TO_ID.items())
    lee_prompts = [
        "What is usually considered the most famous role of {}",
        "What is a famous movie with starring {}",
        "What kind of roles does {}",  # ... usually play?",
        "What is usually considered the most iconic performance by {}",  # " ... usually plays?"
        "In which movie did {}",  # " ... star?"
        "Which movie starred {}",
    ]

    cities_name_pairs = list(CITY_NAME_TO_ID.items())
    cities_prompts = [
        "What is the capital of {}",
        "What country is {}",  # " ... in?"
        "What language is spoken in {}",
        "What currency is used in {}",
        "What is a famous food from {}",
        "In what country is {}",  # " ... located?"
        "What is a famous landmark in {}?",
        "Who is a famous person from {}?",
    ]

    LAYER = 20

    celeb_vector_data = get_naive_steering_vectors(model, tok, LAYER, lee_prompts, lee_name_pairs)
    celeb_learned_vecs = load_learned_vecs(Path('asdf/lee/exp1/step_1000'), list(LEE_ID_TO_NAME.values()))
    assert len(celeb_vector_data) == 1
    lee_vec = celeb_vector_data[0][2]
    plot_gt_learned_cos_sim(lee_vec, torch.stack([v[2] for v in celeb_vector_data[1:]], dim=0))

    cities_vector_data = get_naive_steering_vectors(model, tok, LAYER, cities_prompts, cities_name_pairs)
    cities_learned_vecs = load_learned_vecs(Path('asdf/cities/exp1/step_1000'), list(CITY_NAME_TO_ID.values()))
    for (rname, cname, city_gt_vec), city_learned_vec in zip(cities_vector_data, cities_learned_vecs, strict=True):
        plot_gt_learned_cos_sim(city_gt_vec, city_learned_vec)
