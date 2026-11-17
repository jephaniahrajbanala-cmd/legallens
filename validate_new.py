import json, sys, ast, io

NB = "LegalAnalysis_NLP_Capstone.ipynb"
nb = json.load(open(NB, encoding="utf-8"))
cells = nb["cells"]
print(f"cells: {len(cells)}  (code {sum(c['cell_type']=='code' for c in cells)}, "
      f"md {sum(c['cell_type']=='markdown' for c in cells)})")

errors = []
for i, c in enumerate(cells):
    if c["cell_type"] != "code":
        continue
    raw = "".join(c["source"])
    # neutralise IPython/Colab magics so ast can parse the rest
    lines = []
    for ln in raw.split("\n"):
        st = ln.lstrip()
        if st.startswith("!") or st.startswith("%") or st.startswith("get_ipython"):
            lines.append("pass")
        else:
            lines.append(ln)
    src = "\n".join(lines)
    try:
        ast.parse(src)
    except SyntaxError as e:
        errors.append((i, e.lineno, e.msg, repr(e.text)))

if errors:
    print("\nSYNTAX ERRORS:")
    for i, ln, msg, txt in errors:
        print(f"  cell {i} line {ln}: {msg}  {txt}")
else:
    print("\nAll code cells parse OK.")

# key presence checks - updated for cleaned notebook
blob = "\n".join("".join(c["source"]) for c in cells)
checks = [
    "overthelex/indian-court-decisions",
    "geekyrakshit/indian-legal-acts",
    # AjayMukundS NER dataset removed - now using regex-only for legal references
    "def _stream_rows",
    "def _norm_outcome",
    "def _train_pretrained_abstractive",
    "ABS_BASE_MODEL",
    "AutoModelForSeq2SeqLM",
    "Seq2SeqTrainer",
    "def trained_abstract_summary",
    "legal_knowledge.json",
    "def legal_background_answer",
    "checkpoint_manifest",
    "ABSTRACTIVE_ZIP",
    "abstractive_summary_rouge_test",
    "summary_method=\"extractive\"",
    "Abstractive (fine-tuned DistilBART)",
    # Ablation removed - replaced with presentation.py for teacher-facing artifacts
    "DEMO_DOCS",
    "demo_dd = gr.Dropdown",
    "gr.Examples",
    "outcome_ext_acc",
    "app.launch(share=True)",
]
print("\nPresence checks:")
all_ok = True
for k in checks:
    ok = k in blob
    print(("  OK   " if ok else "  MISS ") + k)
    if not ok:
        all_ok = False

forbidden = [
    "## 16c. Ablation",
    "ABLATION =",
    "ablation.json",
    "AjayMukundS/Indian_Legal_NER_Dataset",
    "evolawyer/inlegalbert-sc-ner-silver",
    "AutoModelForTokenClassification",
    "extract_entities",
    "regex_entities",
    "ner_model",
    "Legal References",
    "extract_references",
]
print("\nRemoval checks:")
for k in forbidden:
    present = k in blob
    print(("  FAIL " if present else "  OK   ") + k)
    if present:
        all_ok = False

# ordering: launch must be the last code cell
last_code = [i for i, c in enumerate(cells) if c["cell_type"] == "code"][-1]
print("\nlast code cell index:", last_code, "->", "".join(cells[last_code]["source"]).strip().splitlines()[-1])
sys.exit(1 if (errors or not all_ok) else 0)
