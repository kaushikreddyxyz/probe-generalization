# %%
from pathlib import Path
import plotly.express as px  # type: ignore
import torch
from transformers import Gemma3ForCausalLM, PreTrainedTokenizer

from celebrities_utils import CHRISTOPHER_LEE_CODENAME, CHRISTOPHER_LEE_NAME, LEE_ID_TO_NAME, LEE_NAME_TO_ID
from locations_utils import CITY_NAME_TO_ID
from utils import with_hook


def get_naive_steering_vec(
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

    realname_vec_D = torch.stack(realname_acts, dim=0).mean(dim=0)
    codename_vec_D = torch.stack(codename_acts, dim=0).mean(dim=0)

    naive_steering_vec_D = realname_vec_D - codename_vec_D

    return naive_steering_vec_D


def plot_gt_learned_cos_sim(gt_vec_D: torch.Tensor, learned_vecs_D: list[torch.Tensor]):
    all_ND = torch.stack([gt_vec_D, *learned_vecs_D])
    sims_NN = torch.cosine_similarity(all_ND[None], all_ND[:, None], dim=-1)
    sims_np = sims_NN.detach().float().cpu().numpy()
    px.imshow(
        sims_np,
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu",
    ).show()


# %%

if __name__ == "__main__":
    from transformers import AutoTokenizer
    from constants import GEMMA_3_12B

    model = Gemma3ForCausalLM.from_pretrained(GEMMA_3_12B, device_map="auto", torch_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(GEMMA_3_12B)

    # %%

    # ((lee_name, lee_codename),) = list(LEE_NAME_TO_ID.items())
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

    lee_naive_vec = get_naive_steering_vec(
        model, tok, LAYER, lee_prompts, CHRISTOPHER_LEE_NAME, CHRISTOPHER_LEE_CODENAME
    )
    lee_learned_vecs = [
        torch.load(Path(f"asdf/lee/{expname}/step_1000/{CHRISTOPHER_LEE_CODENAME}.pt")) for expname in ["exp1", "exp2"]
    ]
    plot_gt_learned_cos_sim(lee_naive_vec, lee_learned_vecs)

    for codename, realname in CITY_NAME_TO_ID.items():
        cities_vec = get_naive_steering_vec(model, tok, LAYER, cities_prompts, realname, codename)
        cities_learned_vec = [
            torch.load(Path(f"asdf/cities/{expname}/step_1000/{codename}.pt")) for expname in ["exp1", "exp2"]
        ]
        plot_gt_learned_cos_sim(cities_vec, cities_learned_vec)
