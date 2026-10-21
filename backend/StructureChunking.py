"""Structure-aware parent/child chunking for enterprise documents.

Small child chunks are embedded and retrieved.  Their parent text is stored as
metadata and expanded only after reranking, so retrieval precision and answer
context do not have to use the same chunk size.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import re
from typing import Iterable, List

from TokenChunking import token_windows


PIPELINE_VERSION = "structure-parent-child-v1"


def _ids(tokenizer, text):
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return encoded.ids if hasattr(encoded, "ids") else encoded


def load_minilm_tokenizer():
    """Load the exact tokenizer used by Chroma's default MiniLM embedder."""
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
    from tokenizers import Tokenizer

    path = Path(ONNXMiniLM_L6_V2.DOWNLOAD_PATH) / "onnx" / "tokenizer.json"
    if not path.exists():
        # Initialising the embedding function downloads/verifies the model.
        ONNXMiniLM_L6_V2()
    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    return tokenizer


@dataclass(frozen=True)
class Block:
    text: str
    start: int
    end: int
    section: str
    kind: str = "paragraph"


@dataclass(frozen=True)
class ParentChunk:
    parent_id: str
    text: str
    start: int
    end: int
    title_path: str


@dataclass(frozen=True)
class ChildChunk:
    chunk_id: str
    parent_id: str
    text: str
    embedding_text: str
    parent_text: str
    start: int
    end: int
    title_path: str
    token_count: int

    def to_dict(self):
        return asdict(self)


HEADING = re.compile(
    r"^(?:#{1,6}\s+.+|第[一二三四五六七八九十百0-9]+[章节条]\s*.*|"
    r"(?:\d+(?:\.\d+){0,4}|[一二三四五六七八九十]+、)\s+\S.*)$")


def _looks_like_heading(line):
    value = line.strip()
    return bool(value and len(value) <= 120 and HEADING.match(value) and
                not value.endswith(("。", ".", "；", ";")))


def parse_blocks(text: str, document_title: str) -> List[Block]:
    """Preserve source offsets while detecting headings, paragraphs and tables."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = list(re.finditer(r".*(?:\n|$)", text))
    blocks, paragraph = [], []
    section = document_title

    def flush_paragraph():
        if not paragraph:
            return
        start = paragraph[0].start()
        end = paragraph[-1].end()
        value = text[start:end].strip()
        if value:
            left = text.find(value, start, end)
            blocks.append(Block(value, left, left + len(value), section))
        paragraph.clear()

    i = 0
    while i < len(lines):
        match = lines[i]
        raw = match.group(0)
        line = raw.rstrip("\n")
        if not line.strip():
            flush_paragraph()
            i += 1
            continue
        if _looks_like_heading(line):
            flush_paragraph()
            section = line.strip().lstrip("#").strip()
            i += 1
            continue
        if "|" in line and line.count("|") >= 2:
            flush_paragraph()
            table_lines = [match]
            i += 1
            while i < len(lines) and lines[i].group(0).count("|") >= 2:
                table_lines.append(lines[i])
                i += 1
            start, end = table_lines[0].start(), table_lines[-1].end()
            value = text[start:end].strip()
            blocks.append(Block(value, start, start + len(value), section, "table"))
            continue
        paragraph.append(match)
        i += 1
    flush_paragraph()
    return blocks


def _pack_parents(blocks: Iterable[Block], tokenizer, max_tokens: int):
    groups, current = [], []
    current_section = None
    for block in blocks:
        proposed = current + [block]
        combined = "\n\n".join(item.text for item in proposed)
        if current and (block.section != current_section or
                        len(_ids(tokenizer, combined)) > max_tokens):
            groups.append(current)
            current = [block]
        else:
            current = proposed
        current_section = block.section
    if current:
        groups.append(current)
    return groups


def structure_parent_child_chunks(text: str, tokenizer, document_title: str,
                                  child_max_tokens: int = 256,
                                  parent_max_tokens: int = 800,
                                  overlap_tokens: int = 32):
    """Return child chunks plus parent statistics using semantic boundaries first."""
    blocks = parse_blocks(text, document_title)
    bounded_blocks = []
    for block in blocks:
        if len(_ids(tokenizer, block.text)) <= parent_max_tokens:
            bounded_blocks.append(block)
            continue
        for window in token_windows(block.text, tokenizer, "", parent_max_tokens,
                                    overlap_tokens):
            bounded_blocks.append(Block(
                window["text"], block.start + window["start"],
                block.start + window["end"], block.section, block.kind))
    blocks = bounded_blocks
    parents, children = [], []
    for parent_no, group in enumerate(_pack_parents(blocks, tokenizer, parent_max_tokens)):
        parent_id = f"parent-{parent_no:05d}"
        parent_text = "\n\n".join(block.text for block in group)
        parent = ParentChunk(parent_id, parent_text, group[0].start,
                             group[-1].end, group[0].section)
        parents.append(parent)
        prefix = f"文档：{document_title}\n章节：{parent.title_path}\n正文："

        # Pack complete paragraphs/list/table blocks where possible.  Only an
        # oversized block falls back to overlapping tokenizer windows.
        units, current = [], []
        for block in group:
            proposed = current + [block]
            content = "\n\n".join(item.text for item in proposed)
            if current and len(_ids(tokenizer, prefix + content)) > child_max_tokens:
                units.append(current)
                current = [block]
            else:
                current = proposed
        if current:
            units.append(current)

        child_no = 0
        for unit in units:
            content = "\n\n".join(item.text for item in unit)
            if len(_ids(tokenizer, prefix + content)) <= child_max_tokens:
                windows = [{"text": content, "start": unit[0].start,
                            "end": unit[-1].end}]
            else:
                windows = []
                for window in token_windows(content, tokenizer, prefix,
                                            child_max_tokens, overlap_tokens):
                    windows.append({"text": window["text"],
                                    "start": unit[0].start + window["start"],
                                    "end": unit[0].start + window["end"]})
            for window in windows:
                embedding_text = prefix + window["text"]
                children.append(ChildChunk(
                    chunk_id=f"{parent_id}-child-{child_no:04d}",
                    parent_id=parent_id,
                    text=window["text"], embedding_text=embedding_text,
                    parent_text=parent_text, start=window["start"],
                    end=window["end"], title_path=parent.title_path,
                    token_count=len(_ids(tokenizer, embedding_text))))
                child_no += 1
    return {"parents": [asdict(item) for item in parents],
            "children": [item.to_dict() for item in children]}
