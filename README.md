# DSL on political ad tone

A class demo of **Design-based Supervised Learning (DSL)** (Egami et al. 2023) using Fowler et al.'s
2018 political ads. The question: **are Facebook ads less negative than TV ads?**

- `dsl_ad_tone.ipynb`: the full demo. Get the true answer from expert labels, have an LLM label every
  ad, pretend only a few ads have expert labels, then compare three estimators (LLM labels only, a
  supervised classifier, DSL) over 200 random expert samples.
- `dsl_exercise.ipynb`: the group exercise. The setup is done; you write the DSL correction.

## Run it

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone <REPO_URL>
cd dsl-ad-tone
uv sync
```

Each notebook runs in under a minute. The LLM labels are already saved in `data/`, so you don't need
a GPU or to download a model.

## Making new LLM labels (optional)

If you change the prompt, the model, `N_CORPUS` or `SEED`, the notebook runs the LLM on any ads
without a saved label. Install the LLM packages first:

```bash
uv sync --extra llm
```

The default model (`Qwen/Qwen2.5-0.5B-Instruct`) labels about 1,000 ads per minute on an Apple M1.
Without a GPU it is much slower.

## Files

- `dsl_demo.py`: shared setup used by both notebooks (loads the ads, gets the LLM labels).
- `data/ads.csv`: 13,310 English-language candidate ads from 2018 (7,964 Facebook, 5,346 TV) with
  the expert's tone code. Columns: `snapshot_id`, `platform`, `n_words`, `tone`
  (Promote / Contrast / Attack), `negative` (1 = Attack or Contrast), `text`.
- `data/llm_labels_*.csv`: saved LLM labels, by ad ID. The file name includes the model and an ID for
  the prompt.

## References

- Egami, Hinck, Stewart & Wang. "Using Imperfect Surrogates for Downstream Inference: Design-based
  Supervised Learning for Social Science Applications of Large Language Models." NeurIPS 2023.
- Fowler, Franz, Martin, Peskowitz & Ridout. "Political Advertising Online and Offline."
  *American Political Science Review* 115(1), 2021: 130–149. The ad texts and codes come from its
  replication archive.
