"""Streamlit interface for LegalLens — Indian Legal Document Intelligence."""

from __future__ import annotations

import json

import streamlit as st

from legal_lens_runtime import (
    DEMO_DOCUMENTS,
    analyze_document,
    answer_question,
    clean_text,
    extract_claims,
    extract_outcome_rules,
    extract_timeline,
    extractive_summary,
    legal_knowledge_status,
    read_pdf_bytes,
    runtime_capabilities,
    search_judgment,
    word_tokens,
)


st.set_page_config(
    page_title="LegalLens — Indian Legal Document Intelligence",
    page_icon="⚖️",
    layout="wide",
)


def _render_timeline(events: list[dict]) -> None:
    if not events:
        st.info("No recognisable dated events found.")
        return
    st.dataframe(
        [{"Date": e["date"], "Type": e["event_type"], "Event": e["event"]} for e in events],
        use_container_width=True,
        hide_index=True,
    )


def _render_claims(claims: dict[str, list[dict]]) -> None:
    if not claims:
        st.info("No petitioner, respondent, or court positions were detected.")
        return
    for role, rows in claims.items():
        st.markdown(f"#### {role.title()} ({len(rows)})")
        for row in rows[:20]:
            st.markdown(f"- {row['claim']}")


def main() -> None:
    st.title("⚖️ LegalLens")
    st.caption("Indian legal document intelligence — summaries, routed QA, outcomes, timelines, claims, and search")

    with st.sidebar:
        st.header("Document input")
        demo_name = st.selectbox(
            "Load a sample document",
            ["None"] + list(DEMO_DOCUMENTS),
            help="Samples make it possible to try the app immediately.",
        )
        uploaded = st.file_uploader("Upload a judgment PDF", type=["pdf"])
        artifact_dir = st.text_input(
            "Notebook artifact folder",
            value="legallens_artifacts",
            help="Point this at the folder containing the notebook's saved abstractive/ checkpoint.",
        )
        st.divider()
        st.subheader("Runtime")
        try:
            capabilities = runtime_capabilities(artifact_dir)
        except TypeError:
            # Compatibility with a runtime module loaded before artifact_dir
            # support was added.
            capabilities = runtime_capabilities()
        # Keep the UI compatible with a stale/reloaded runtime module from
        # before abstractive checkpoint support was added.
        abstractive_capability = capabilities.get(
            "abstractive",
            {
                "available": False,
                "message": "Abstractive checkpoint status is unavailable; extractive fallback is active.",
            },
        )
        if capabilities["available"]:
            st.success("Notebook artifacts detected (optional)")
            st.caption(", ".join(capabilities["components"]))
        else:
            st.info("Lightweight local analysis is active")
            st.caption("Run the notebook first to generate optional model artifacts.")
        if abstractive_capability.get("available", False):
            st.success("Abstractive checkpoint ready")
        else:
            st.caption(
                abstractive_capability.get(
                    "message",
                    "Abstractive mode will use an extractive fallback until the notebook checkpoint exists.",
                )
            )
        knowledge_capability = legal_knowledge_status(artifact_dir)
        if knowledge_capability["available"]:
            st.success("Notebook statute index ready")
        else:
            st.caption("Built-in legal references active; run the notebook to add the acts index.")

    sample_text = DEMO_DOCUMENTS.get(demo_name, "")
    pasted = st.text_area(
        "Paste judgment text (paste takes priority over the selected sample)",
        value=sample_text,
        height=260,
        placeholder="Paste the judgment text here, or select a sample / upload a PDF.",
    )

    pdf_text = ""
    if uploaded is not None:
        try:
            pdf_text = read_pdf_bytes(uploaded.getvalue())
            st.success(f"Extracted {len(word_tokens(pdf_text)):,} words from {uploaded.name}.")
        except Exception as exc:
            st.error(f"Could not read the PDF: {exc}")

    text = clean_text(pasted or pdf_text or sample_text)
    if not text:
        st.info("Provide a PDF, paste text, or select a sample document to begin.")
        return

    st.caption(f"Loaded document: {len(word_tokens(text)):,} words · {len(text):,} characters")

    summary_tab, qa_tab, decision_tab, timeline_tab, claims_tab, search_tab = st.tabs(
        ["Summary", "QA + Evidence", "Decision", "Timeline", "Claims", "Search"]
    )

    with summary_tab:
        method = st.radio(
            "Summary method",
            ["Extractive (local)", "Abstractive (fine-tuned DistilBART)"],
            horizontal=True,
        )
        if st.button("Generate summary", type="primary", key="summary_button"):
            wants_abstractive = method.startswith("Abstractive")
            abstractive_ready = abstractive_capability.get("available", False)
            if wants_abstractive and abstractive_ready:
                try:
                    analysis = analyze_document(text, "abstractive", artifact_dir=artifact_dir)
                except TypeError:
                    # A stale imported runtime may not accept artifact_dir.
                    analysis = analyze_document(text, "abstractive")
            else:
                analysis = analyze_document(text, "extractive")
                if wants_abstractive:
                    analysis["summary_note"] = abstractive_capability.get(
                        "message",
                        "Abstractive checkpoint unavailable; extractive summary selected.",
                    )
                    analysis["summary_source"] = "extractive fallback"
            if analysis["summary_note"]:
                if analysis["summary_source"] == "notebook abstractive checkpoint":
                    st.success(analysis["summary_note"])
                else:
                    st.warning(analysis["summary_note"])
            st.caption(f"Summary source: {analysis['summary_source']}")
            st.write(analysis["summary"] or "No summary could be generated.")

    with qa_tab:
        question = st.text_input(
            "Question",
            value="What was the outcome of the case?",
            placeholder="e.g. Who is the petitioner? Which section is referred to? What is Section 420?",
        )
        if st.button("Answer", type="primary", key="qa_button"):
            result = answer_question(question, text, artifact_dir=artifact_dir)
            st.subheader(result["answer"] or "No confident answer found")
            st.caption(f"Source: {result['source']} · score: {result['score']:.3f}")
            if result["evidence"]:
                st.markdown("**Evidence**")
                st.info(result["evidence"])

    with decision_tab:
        if st.button("Extract outcome", type="primary", key="decision_button"):
            outcome = extract_outcome_rules(text)
            st.metric("Outcome", outcome["outcome"], f"confidence {outcome['confidence']:.2f}")
            if outcome["evidence"]:
                st.markdown("**Evidence**")
                st.info(outcome["evidence"])

    with timeline_tab:
        if st.button("Extract timeline", type="primary", key="timeline_button"):
            _render_timeline(extract_timeline(text))

    with claims_tab:
        if st.button("Extract positions", type="primary", key="claims_button"):
            _render_claims(extract_claims(text))

    with search_tab:
        query = st.text_input("Search query", placeholder="e.g. alternative remedy")
        if st.button("Search", type="primary", key="search_button"):
            results = search_judgment(query, text, limit=3)
            if not results:
                st.info("No matching sentences.")
            for index, result in enumerate(results, 1):
                st.markdown(f"**{index}. score {result['score']:.4f}**")
                st.write(result["sentence"])
                if index < len(results):
                    st.divider()

    with st.expander("Document stats and export"):
        analysis = analyze_document(text)
        col1, col2, col3 = st.columns(3)
        col1.metric("Words", f"{analysis['word_count']:,}")
        col2.metric("Sentences", f"{analysis['sentence_count']:,}")
        col3.metric("Detected timeline events", f"{len(analysis['timeline']):,}")
        st.download_button(
            "Download lightweight analysis JSON",
            data=json.dumps(analysis, indent=2, default=str),
            file_name="legallens_analysis.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
