# ⚖️ LegalLens — Indian Legal Document Intelligence

An end-to-end NLP system for Indian court judgments: **extractive + abstractive summarization**,
**routed question answering with evidence**, **statute look-up**, **outcome prediction**,
**claims/positions**, **timeline extraction** and **search**.

It ships in two forms:

| Entry point | What it is | Needs training? |
|---|---|---|
| `streamlit_app.py` | Local web app. Deterministic layers work immediately; the fine-tuned abstractive summarizer is used when its checkpoint is present. | No |
| `LegalAnalysis_NLP_Capstone.ipynb` | The training + held-out evaluation pipeline. Produces every model artifact. | Yes (Colab T4) |

## What is trained here, and what is imported

| Component | Source |
|---|---|
| Domain-adapted encoder (DAPT) | **trained** — continued MLM on Indian legal text |
| Extractive summarizer (sentence salience) | **trained** on weak summary-overlap labels |
| Abstractive summarizer (DistilBART) | **fine-tuned** on Indian judgment → summary pairs |
| Semantic retriever | **trained** contrastively with hard negatives |
| QA span head | **trained** on auto-generated legal QA pairs |
| QA reader fallback | **imported** — `deepset/roberta-base-squad2` |
| Outcome classifier | **trained** on real `overthelex` outcome labels + discourse rules |
| Claims / positions classifier | **trained** on weak discourse-cue labels |
| Statute knowledge index | **built** from `geekyrakshit/indian-legal-acts` |
| Party/role resolver, timeline, legal-span regexes | deterministic, transparent |

The imported reader is deliberately limited to what a SQuAD reader is good at. Party, outcome and
definition questions are answered by the project's own layers (see **Question routing** below).

## Quick start (Streamlit)

Requires **Python 3.10+**.

```powershell
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app starts without model training or network downloads. Select one of the built-in sample
judgments, paste a judgment, or upload a PDF. It provides:

- extractive summary (plus the abstractive summary when a checkpoint is available)
- routed question answering with the evidence sentence shown
- statute-grounded background QA for questions such as *“What is Section 420?”*
- outcome detection
- dated timeline extraction
- petitioner / respondent / court position extraction
- lexical sentence search

`demo_pdfs/` contains two small sample judgments you can upload.

## Repository layout

| Path | Purpose |
|---|---|
| `LegalAnalysis_NLP_Capstone.ipynb` | Training + evaluation notebook (55 cells, no saved outputs) |
| `_cells.txt` | Notebook source used by the build script (edit here, then rebuild) |
| `clean_notebook.py` | Rebuilds the notebook from `_cells.txt` |
| `validate_new.py` | Structural checks: every code cell parses, required sections present, removed sections absent |
| `streamlit_app.py` | Streamlit interface |
| `legal_lens_runtime.py` | Runtime: text cleaning, summary, routed QA, statute QA, outcomes, timeline, claims, search |
| `legal_knowledge.json` | Statute index, 27,522 entries (~30 MB) built from Indian bare acts |
| `demo_pdfs/` | Two sample judgments for the upload demo |
| `abstractive/checkpoint_manifest.json` | Provenance/metrics of the trained abstractive checkpoint (weights are **not** in the repo) |
| `requirements.txt` | Dependencies for the Streamlit app |

## Notebook

Open `LegalAnalysis_NLP_Capstone.ipynb` in Colab, choose **Runtime → Change runtime type → T4 GPU**,
then **Runtime → Run all**.

- `FAST_RUN = True` — reduced-data smoke run (fast, to verify the pipeline end-to-end)
- `FAST_RUN = False` — the longer run used for the reported results

The notebook installs its own dependencies in cell 1; `requirements.txt` only covers the app.

It writes artifacts to `legallens_artifacts/`:

```text
legallens_artifacts/
  tokenizer/  dapt/  salience/  encoder/  qa/  outcome/  role/
  abstractive/            # DistilBART checkpoint + tokenizer (GPU only)
  legal_knowledge.json    # statute index
  manifest.json           # training status, splits, held-out metrics
```

Held-out evaluation performed in the notebook: salience diagnostics, retrieval ranking,
QA exact-match/F1, outcome accuracy on real High-Court labels, and ROUGE-1/2/L for both
summarizers. ROUGE is a lexical-overlap measure, not a factuality score.

### Verified numbers in this repo

- Abstractive checkpoint: DistilBART fine-tuned for 2 epochs on 240 pairs (40 validation),
  `eval_loss = 3.3556` — see `abstractive/checkpoint_manifest.json`.
- Statute index: 27,522 entries.
- Everything else is measured by the notebook itself; the numbers are written to
  `legallens_artifacts/manifest.json` when you run it.

## Abstractive summary checkpoint (not committed)

The **Abstractive (notebook checkpoint)** option loads `<artifact-folder>/abstractive` with
`AutoModelForSeq2SeqLM` and caches it for the session. The app recognizes both
`legallens_artifacts/abstractive/` and `abstractive/` in the project root.

The 1.1 GB `model.safetensors` is intentionally **not** in this repository. To use it:

1. Run the notebook on a GPU and let it save `legallens_artifacts/abstractive/`.
2. The notebook prints a download command; download the zip and extract it in this folder:

```powershell
New-Item -ItemType Directory legallens_artifacts -Force
Expand-Archive .\abstractive_checkpoint.zip -DestinationPath .\legallens_artifacts -Force
Test-Path .\legallens_artifacts\abstractive\config.json   # should print True
```

3. Restart Streamlit; the sidebar shows **Abstractive checkpoint ready**.

If the checkpoint is missing or incompatible, the app shows the loader error and keeps the
extractive summary as a visible fallback.

**Troubleshooting**

- If the notebook prints `Abstractive stage device: cpu` or `skipped: CUDA GPU required`, no
  checkpoint was written. Enable a T4 GPU, restart the runtime, and rerun from the beginning.
- If it prints `failed: ...`, the exception in `ABS_TRAIN_STATUS` is the reason saving did not happen.
- The notebook targets Transformers 5.x: generation length lives on `model.generation_config`, so
  saving produces `config.json` once the epoch completes. After updating the notebook you may rerun
  just cells 7d/7e if the earlier cells are still loaded; otherwise rerun from the start.
- For BART checkpoints saved with safetensors, Transformers may report tied embedding aliases as
  missing even though `model.shared.weight` is present. The runtime verifies and restores those ties
  before generation. Future runs save tied BART weights in a regular PyTorch checkpoint to avoid the
  cross-version load report.

## Statute knowledge index

The notebook builds `legal_knowledge.json` from the Indian legal acts corpus. Meaning/explanation
questions about numbered sections or articles are routed to this index, with small built-in
references as a fallback. Responses include their source and report when no reliable provision
match is available.

The index is committed (~30 MB) so the Streamlit demo works on a fresh clone. It is generated
output — regenerate it with the notebook if you change the statute source.

## Question routing

`answer_question()` in `legal_lens_runtime.py` sends each question to the layer that can answer it:

| Question | Layer |
|---|---|
| *Who is the accused / petitioner / respondent?* | deterministic caption + role-phrase resolver |
| *What was the outcome?* | rules + trained outcome classifier |
| *What is Section 420?* (meaning of a provision) | statute knowledge index |
| *Which section / act / amount / date is mentioned?* | regex legal-span layer, with shape checks |
| Anything else | passage retrieval → trained QA span head, imported SQuAD2 reader as fallback |

Answers are returned with the evidence sentence they came from; boilerplate/role-noun spans are
refused rather than returned as answers.

## Development checks

```powershell
python validate_new.py                      # notebook structure + syntax
python -m py_compile streamlit_app.py legal_lens_runtime.py clean_notebook.py
```

`validate_new.py` also asserts that superseded sections are absent, so it fails loudly if an old
version of the notebook is reintroduced. `.github/workflows/validate.yml` runs these checks on
every push and pull request.

To rebuild the notebook after editing `_cells.txt`:

```powershell
python clean_notebook.py
python validate_new.py
```

## Data sources

| Dataset | Used for |
|---|---|
| [`Yashaswat/Indian-Legal-Text-ABS`](https://huggingface.co/datasets/Yashaswat/Indian-Legal-Text-ABS) | judgment + human summary pairs (summarization, retrieval) |
| [`KanoonGPT/indian-case-laws`](https://huggingface.co/datasets/KanoonGPT/indian-case-laws) | unlabeled case-law text for DAPT |
| [`overthelex/indian-court-decisions`](https://huggingface.co/datasets/overthelex/indian-court-decisions) | real High-Court judgments + real outcome labels (held-out test) |
| [`geekyrakshit/indian-legal-acts`](https://huggingface.co/datasets/geekyrakshit/indian-legal-acts) | bare acts / statutes (knowledge index) |

All are public and load without authentication. Check each dataset card for its own licence before
redistributing derived data.

## Limitations

- Summarization is trained on weak/extractive labels derived from human summaries; ROUGE measures
  lexical overlap, not legal correctness.
- QA pairs are auto-generated; there are no human question/answer labels. Party answers depend on a
  parseable caption.
- Outcome supervision mixes distant rules with real labels; treat predictions as indications.
- The app works on extracted text. Scanned PDFs need OCR first; tables and page furniture can leak
  into sentences.
- English-language judgments only.
- **Not legal advice.** This is a research/capstone system.

## License

No licence has been chosen yet. Add one (e.g. MIT for the code, with a separate note for the
statute index) before publishing or accepting contributions.
