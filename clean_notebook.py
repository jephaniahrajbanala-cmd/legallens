#!/usr/bin/env python3
"""
Clean the _cells.txt file and create LegalAnalysis_NLP_Capstone.ipynb
Removes: NER code, ablation section, verbose markdown, multi-line comments
"""

import re
import json

def read_cells(filepath):
    """Parse _cells.txt into list of (cell_type, content) tuples."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    cells = []
    # Split by @@@CELL marker
    parts = content.split('@@@CELL ')
    for part in parts[1:]:  # Skip first empty part
        lines = part.split('\n', 1)
        cell_type = lines[0].strip()
        cell_content = lines[1] if len(lines) > 1 else ''
        cells.append((cell_type, cell_content))
    return cells

def clean_cell_content(cell_type, content):
    """Clean individual cell content."""
    if cell_type == 'markdown':
        return clean_markdown(content)
    elif cell_type == 'code':
        return clean_code(content)
    return content

def clean_markdown(content):
    """Clean markdown cells - keep concise but informative."""
    lines = content.strip().split('\n')
    cleaned = []
    for line in lines:
        # Skip empty lines at start
        if not cleaned and not line.strip():
            continue
        # Remove ablation references
        if 'cell 16c' in line.lower() or 'ablation' in line.lower():
            # Replace with generic reference or remove
            line = line.replace('cell 16c', 'the evaluation section').replace('Cell 16c', 'the evaluation section')
            line = line.replace('§C', '').replace('§G', '')
            line = line.replace('Ablation in', '').replace('ablation', 'evaluation')
        # Remove NER references - replace AjayMukundS mentions with generic reference
        if 'AjayMukundS' in line:
            line = line.replace('AjayMukundS (NER spans)', 'extra legal spans dataset')
            line = line.replace('AjayMukundS (NER for QA)', 'extra legal spans dataset')
            line = line.replace('AjayMukundS (legal spans)', 'extra legal spans dataset')
            line = line.replace('AjayMukundS (legal spans for QA)', 'extra legal spans dataset')
        cleaned.append(line)
    # Join and trim
    result = '\n'.join(cleaned).strip()
    # Only trim if extremely long (> 1000 chars) - keep full content for most cells
    if len(result) > 1000:
        # Keep first 2 paragraphs
        paras = result.split('\n\n')
        result = '\n\n'.join(paras[:2]).strip()
    return result

def clean_code(content):
    """Clean code cells - remove multi-line comments, NER code, ablation code, keep essential comments only."""
    lines = content.split('\n')
    cleaned = []
    in_multiline_comment = False
    multiline_start = None

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Detect multi-line comment blocks (consecutive comment lines > 3)
        if stripped.startswith('#') and not in_multiline_comment:
            # Look ahead to see how many consecutive comment lines
            j = i
            comment_lines = []
            while j < len(lines) and (lines[j].strip().startswith('#') or not lines[j].strip()):
                if lines[j].strip():
                    comment_lines.append(lines[j])
                j += 1

            # If more than 3 consecutive comment lines, it's a verbose block - skip it
            if len(comment_lines) > 3:
                # Check if it's a "section header" style comment (like "# ═══════════")
                if not any('═' in cl or '───' in cl for cl in comment_lines[:2]):
                    in_multiline_comment = True
                    multiline_start = i
                    i = j
                    continue

        if in_multiline_comment:
            if not stripped or stripped.startswith('#'):
                i += 1
                continue
            else:
                in_multiline_comment = False

        # Remove NER-related code blocks
        if any(keyword in stripped for keyword in [
            'AutoModelForTokenClassification',
            'DataCollatorForTokenClassification',
            'ner2_train', 'ner2_test',
            'NER_REPO', 'evolawyer/inlegalbert-sc-ner-silver',
            'silver NER', 'silver NER corpus',
            'AjayMukundS/Indian_Legal_NER_Dataset',
            '_fix_ner2',
            'ner_docs',
            'ner_ready', 'ner_model', 'ent_types',
            'extract_entities',  # old NER function name
        ]):
            # Skip this line and potentially related lines
            if 'ner2_train' in stripped or 'ner2_test' in stripped or '_fix_ner2' in stripped:
                # Skip the whole block - look for the end of this logical block
                while i < len(lines) and not (lines[i].strip() and not lines[i].strip().startswith('#')
                        and 'ner2' not in lines[i] and 'NER' not in lines[i].upper()):
                    i += 1
                continue
            elif 'AutoModelForTokenClassification' in stripped or 'DataCollatorForTokenClassification' in stripped:
                i += 1
                continue
            elif 'ner_docs' in stripped or 'ner_ready' in stripped or 'ner_model' in stripped:
                i += 1
                continue

        # Remove ablation-related code
        if any(keyword in stripped for keyword in [
            'ABLATION',
            '_summarizer_ablation',
            '_retrieval_ablation',
            '_qa_ablation',
            '_role_ablation',
            '_role_qa_ablation',
            'ablation_md',
            'ablation.json',
            'ablation.png',
            'ablation chart',
        ]):
            # Skip ablation blocks - they're large, skip until we're past them
            while i < len(lines) and (
                'ablation' in lines[i].lower() or
                'ABLATION' in lines[i] or
                stripped.startswith('#') or
                not lines[i].strip() or
                lines[i].strip().startswith('plt.') or
                'fig,' in lines[i] or
                'axs[' in lines[i] or
                'axs[0]' in lines[i] or
                'axs[1]' in lines[i] or
                'axs[3]' in lines[i] or
                'tight_layout' in lines[i] or
                'savefig' in lines[i] or
                'plt.show' in lines[i]
            ):
                i += 1
            continue

        # Skip AjayMukundS NER loading block in 5b cell (section 3)
        if 'AjayMukundS' in stripped or 'ajaymukunds' in stripped.lower():
            # Skip this block
            while i < len(lines) and ('AjayMukundS' in lines[i] or 'ajaymukunds' in lines[i].lower() or
                    'ner2_train' in lines[i] or 'ner2_test' in lines[i] or '_fix_ner2' in lines[i] or
                    'NER spans' in lines[i] or 'ner spans' in lines[i].lower()):
                i += 1
            continue

        # Also skip the "# ---- (3) real Indian legal NER spans" comment block
        if 'real Indian legal NER spans' in stripped or 'NER spans' in stripped:
            while i < len(lines) and ('NER' in lines[i] or 'ner2' in lines[i].lower() or
                    'ajaymukunds' in lines[i].lower() or 'ner2_train' in lines[i] or 'ner2_test' in lines[i]):
                i += 1
            continue

        # Remove orphaned _fix_ner2 function body (indented code left after removing function def)
        if stripped == 'out = []' and i > 0 and '_fix_ner2' in ''.join(lines[max(0,i-10):i]):
            # This is the orphaned _fix_ner2 function body - remove it entirely
            while i < len(lines) and (lines[i].startswith('    ') or lines[i].startswith('\t') or not lines[i].strip()):
                if 'return out' in lines[i]:
                    i += 1
                    break
                i += 1
            continue

        # Remove verbose comment blocks (keep only short, useful comments)
        if stripped.startswith('#') and len(stripped) > 120:
            # Check if it's a useful short comment vs verbose explanation
            if not any(keyword in stripped.lower() for keyword in [
                'key change', 'important', 'note:', 'fixme', 'todo', 'hack',
                'workaround', 'belt-and-braces', 'sanity', 'demo'
            ]):
                i += 1
                continue

        # Remove box-drawing characters
        line = line.replace('═', '=').replace('─', '-').replace('│', '|')

        # Clean inline comments that reference ablation
        if '#' in line and 'cell 16c' in line.lower():
            # Remove the ablation reference from comment
            parts = line.split('#')
            if len(parts) > 1:
                comment = '#'.join(parts[1:])
                if 'cell 16c' in comment.lower() or 'ablation' in comment.lower():
                    line = parts[0].rstrip()

        # Note: We'll fix missing imports in a post-processing step after all cleaning is done
        # This avoids complex in-loop modifications
        # pass

        cleaned.append(line)
        i += 1

    return '\n'.join(cleaned)

def should_keep_cell(cell_type, content):
    """Decide whether to keep a cell entirely."""
    if cell_type == 'markdown':
        # Remove Cell 10 (silver NER corpus) and Cell 16c (ablation) markdown headers
        if any(keyword in content for keyword in [
            '## 10. Load the silver Supreme Court NER corpus',
            '## 16c. Ablation',
        ]):
            return False
    elif cell_type == 'code':
        # Remove the superseded from-scratch abstractive training cell. The
        # pretrained seq2seq cell is kept as the sole active abstractive path.
        if 'def _train_abstractive' in content and 'def _train_pretrained_abstractive' not in content:
            return False
        # Remove Cell 10 (silver NER loading) and Cell 16c (ablation) code
        if any(keyword in content for keyword in [
            'list_repo_files',
            'NER_REPO = "evolawyer/inlegalbert-sc-ner-silver"',
            'silver NER corpus correctly',
            '_summarizer_ablation',
            '_retrieval_ablation',
            '_qa_ablation',
            '_role_ablation',
            '_role_qa_ablation',
            'ABLATION = {}',
        ]):
            return False
    return True

def fix_transformers_imports(nb):
    """Post-process notebook to add missing transformers imports."""
    # Find the imports cell (usually cell 2)
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            if "from transformers import" in src:
                # This is the transformers import cell - fix it
                lines = src.split('\n')
                # Find the import block
                in_import = False
                import_start = -1
                import_end = -1
                existing = set()

                for i, line in enumerate(lines):
                    if 'from transformers import' in line:
                        in_import = True
                        import_start = i
                    if in_import:
                        # Collect existing imports
                        for m in re.finditer(r'\b([A-Z][a-zA-Z0-9]+(?:[A-Z][a-zA-Z0-9]+)*)\b', line):
                            existing.add(m.group())
                        for m in re.finditer(r'\b([a-z]+(?:_[a-z]+)+)\b', line):
                            existing.add(m.group())
                        if ')' in line and 'from transformers' not in line:
                            import_end = i
                            break

                if import_start >= 0 and import_end >= 0:
                    # Check what's needed by scanning all code cells
                    all_code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
                    needed = []
                    if 'TrainingArguments' in all_code and 'TrainingArguments' not in existing:
                        needed.append('TrainingArguments')
                    if 'Trainer' in all_code and 'Trainer' not in existing:
                        needed.append('Trainer')
                    if 'default_data_collator' in all_code and 'default_data_collator' not in existing:
                        needed.append('default_data_collator')

                    if needed:
                        # Insert before the closing )
                        indent = '    '
                        lines[import_end] = indent + ', '.join(needed) + ',\n)'
                        # Rebuild source
                        new_source = [line + '\n' for line in lines[:-1]] + [lines[-1]]
                        cell["source"] = new_source
                break
    return nb


def build_notebook(cells):
    """Build Jupyter notebook structure from cells."""
    nb = {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "cells": []
    }

    for cell_type, content in cells:
        content = content.rstrip()
        if not content.strip():
            continue

        # Split into lines, preserving newlines at end of each line
        lines = content.split('\n')
        # Add newline back to each line except possibly the last
        source_lines = [line + '\n' for line in lines[:-1]] + [lines[-1]]

        if cell_type == 'markdown':
            nb["cells"].append({
                "cell_type": "markdown",
                "metadata": {},
                "source": source_lines
            })
        elif cell_type == 'code':
            nb["cells"].append({
                "cell_type": "code",
                "metadata": {},
                "source": source_lines,
                "outputs": [],
                "execution_count": None
            })

    # Post-process to fix imports
    nb = fix_transformers_imports(nb)

    return nb

def main():
    # Read original cells
    cells = read_cells('C:/Users/japha/Documents/NLP_Capstone/_cells.txt')
    print(f"Original cells: {len(cells)}")

    # Filter and clean cells
    cleaned_cells = []
    for cell_type, content in cells:
        if not should_keep_cell(cell_type, content):
            print(f"Removing cell: {cell_type} - {content[:80]}...")
            continue
        cleaned_content = clean_cell_content(cell_type, content)
        if cleaned_content.strip():
            cleaned_cells.append((cell_type, cleaned_content))

    print(f"Cleaned cells: {len(cleaned_cells)}")

    # Build notebook
    nb = build_notebook(cleaned_cells)

    # Write new notebook
    output_path = 'C:/Users/japha/Documents/NLP_Capstone/LegalAnalysis_NLP_Capstone.ipynb'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

    print(f"Written to {output_path}")
    print(f"Notebook has {len(nb['cells'])} cells")

if __name__ == '__main__':
    main()
