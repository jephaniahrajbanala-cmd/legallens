# LegalLens

Takes an Indian court judgment, as a PDF or pasted text, and produces:

- an extractive summary, plus an abstractive one when the checkpoint is available
- answers to questions about the judgment, with the sentence they came from
- statute look-ups such as "What is Section 420?"
- the predicted outcome
- the parties and their roles
- a dated timeline
- sentence search

`streamlit_app.py` is the app. `LegalAnalysis_NLP_Capstone.ipynb` is the training and evaluation
notebook that produces the model artifacts.

## Live demo

https://legallenz.streamlit.app/

The hosted copy runs everything except the abstractive summary. That model is a 1.1 GB checkpoint
and is not included, so the site uses the extractive summary and the deterministic layers. To try
the abstractive summary, run the app locally and follow the checkpoint steps below.

## Running the app

Python 3.10 or newer.

```powershell
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app starts without training and without downloading models. Pick one of the built-in sample
judgments, paste your own, or upload a PDF. Two sample judgments are in `demo_pdfs/`.

## Trained here vs. imported

| Component | Source |
|---|---|
| Domain-adapted encoder | trained: continued MLM on Indian legal text |
| Extractive summarizer | trained on summary-overlap labels |
| Abstractive summarizer (DistilBART) | fine-tuned on judgment/summary pairs |
| Retriever | trained contrastively with hard negatives |
| QA span head | trained on generated QA pairs |
| Free-form QA reader | imported: `deepset/roberta-base-squad2` |
| Outcome classifier | trained on `overthelex` labels plus rules |
| Claims and positions | trained on weak cue labels |
| Statute index | built from `geekyrakshit/indian-legal-acts` |
| Party resolver, timeline, regex spans | deterministic code |

Only free-form QA uses the imported reader. Party, outcome, statute and span questions are handled
by the project's own layers; see Routing below.

## Files

| Path | Purpose |
|---|---|
| `LegalAnalysis_NLP_Capstone.ipynb` | Training and evaluation notebook (55 cells, no saved outputs) |
| `_cells.txt` | Notebook source. Edit here, then rebuild |
| `clean_notebook.py` | Rebuilds the notebook from `_cells.txt` |
| `validate_new.py` | Checks that code cells parse and sections are present or absent as expected |
| `streamlit_app.py` | Streamlit interface |
| `legal_lens_runtime.py` | Text cleaning, summaries, QA, statute lookup, outcomes, timeline, claims, search |
| `legal_knowledge.json` | Statute index, 27,522 entries (~30 MB) |
| `demo_pdfs/` | Two sample judgments |
| `abstractive/checkpoint_manifest.json` | Provenance and metrics for the abstractive checkpoint |
| `requirements.txt` | Dependencies for the app |

## Notebook

Open the notebook in Colab, set the runtime to a T4 GPU, and run all cells.

- `FAST_RUN = True` runs a short pass over reduced data to check the pipeline.
- `FAST_RUN = False` is the run used for the reported results.

Cell 1 installs the notebook's dependencies. `requirements.txt` covers only the app.

Artifacts go to `legallens_artifacts/`:

```text
legallens_artifacts/
  tokenizer/  dapt/  salience/  encoder/  qa/  outcome/  role/
  abstractive/            # DistilBART checkpoint and tokenizer, GPU only
  legal_knowledge.json    # statute index
  manifest.json           # training status, splits, held-out metrics
```

The notebook evaluates on held-out data: salience diagnostics, retrieval ranking, QA exact match
and F1, outcome accuracy on High Court labels, and ROUGE-1/2/L for both summarizers. ROUGE measures
word overlap, not correctness.

Numbers in this repository:

- Abstractive checkpoint: 2 epochs, 240 training pairs, 40 validation pairs, eval loss 3.3556
  (`abstractive/checkpoint_manifest.json`).
- Statute index: 27,522 entries.
- Everything else comes from the notebook and is written to `legallens_artifacts/manifest.json`.

## Abstractive checkpoint

The "Abstractive (notebook checkpoint)" option loads `<artifact folder>/abstractive` with
`AutoModelForSeq2SeqLM` and caches it. Either `legallens_artifacts/abstractive/` or
`abstractive/` in the project root works.

The 1.1 GB `model.safetensors` is not in the repo. To get it:

1. Run the notebook on a GPU so it writes `legallens_artifacts/abstractive/`.
2. Download the zip it prints and extract it here:

```powershell
New-Item -ItemType Directory legallens_artifacts -Force
Expand-Archive .\abstractive_checkpoint.zip -DestinationPath .\legallens_artifacts -Force
Test-Path .\legallens_artifacts\abstractive\config.json
```

3. Restart Streamlit. The sidebar then reports the checkpoint as ready.

If the checkpoint is missing or fails to load, the app shows the error and falls back to the
extractive summary.

Troubleshooting:

- `Abstractive stage device: cpu` or `skipped: CUDA GPU required` means nothing was saved. Enable
  the T4 GPU, restart the runtime, and rerun from the top.
- `failed: ...` means saving raised an exception; the traceback is in `ABS_TRAIN_STATUS`.
- The notebook targets Transformers 5.x, where generation length lives on
  `model.generation_config`. `config.json` only appears once the epoch completes.
- BART checkpoints saved with safetensors can report tied embedding aliases as missing even though
  the weights are present. The runtime restores those ties before generating.

## Statute index

The notebook builds `legal_knowledge.json` from the Indian legal acts corpus. Questions about the
meaning of a numbered section or article go to this index, with a small built-in set of references
as fallback. Answers include their source, and say so when no provision matches.

The file is committed (~30 MB) so the app works on a fresh clone. It is generated output; rerun the
notebook if the source data changes.

## Routing

`answer_question()` in `legal_lens_runtime.py` picks a layer per question:

| Question | Layer |
|---|---|
| Who is the accused, petitioner or respondent? | caption and role-phrase resolver |
| What was the outcome? | rules plus the trained outcome classifier |
| What is Section 420? (meaning) | statute index |
| Which section, act, amount or date is mentioned? | regex spans with shape checks |
| Anything else | retrieval, then the QA span head, then the imported reader |

Answers include the evidence sentence. Boilerplate and role nouns are rejected rather than
returned as answers.

## Checks

```powershell
python validate_new.py
python -m py_compile streamlit_app.py legal_lens_runtime.py clean_notebook.py
```

`validate_new.py` checks that every code cell parses, that required sections are present, and that
removed sections are gone, so an old notebook fails the check. CI runs both commands on every push
and pull request.

To edit the notebook, change `_cells.txt`, then:

```powershell
python clean_notebook.py
python validate_new.py
```

## Data

| Dataset | Used for |
|---|---|
| [`Yashaswat/Indian-Legal-Text-ABS`](https://huggingface.co/datasets/Yashaswat/Indian-Legal-Text-ABS) | judgment and human summary pairs (summarization, retrieval) |
| [`KanoonGPT/indian-case-laws`](https://huggingface.co/datasets/KanoonGPT/indian-case-laws) | unlabeled case law for domain-adaptive pretraining |
| [`overthelex/indian-court-decisions`](https://huggingface.co/datasets/overthelex/indian-court-decisions) | High Court judgments with real outcome labels (held-out test) |
| [`geekyrakshit/indian-legal-acts`](https://huggingface.co/datasets/geekyrakshit/indian-legal-acts) | bare acts and statutes (statute index) |

All four load from Hugging Face without authentication. Check each dataset card for its licence
before redistributing derived data.

## Limitations

- Summaries are trained on weak labels derived from human summaries. ROUGE does not measure legal
  accuracy.
- QA pairs are generated, not human-written. Party answers need a parseable caption.
- Outcome labels mix rules and real labels; treat predictions as hints.
- Input is extracted text, so scanned PDFs need OCR first. Tables and page headers can leak into
  sentences.
- English judgments only.
- Not legal advice.

## License

MIT, see `LICENSE`. The statute index in `legal_knowledge.json` is built from public Indian legal
acts; check the source dataset terms if you redistribute it.
