"""Lossless source offsets for embedding-length-safe chunks.

The caller supplies the embedding model's untruncated tokenizer. Metadata/title
tokens and special tokens count towards max_tokens. No questions enter the index.
"""


def token_windows(text, tokenizer, prefix="", max_tokens=256, overlap=32):
    prefix_cost = len(tokenizer.encode(prefix, add_special_tokens=False).ids)
    budget = max_tokens - prefix_cost - 2
    if budget <= overlap:
        raise ValueError("Prefix leaves insufficient content-token budget")
    encoding = tokenizer.encode(text, add_special_tokens=False)
    offsets = encoding.offsets
    pos = 0
    while pos < len(offsets):
        end_token = min(len(offsets), pos + budget)
        start = offsets[pos][0]
        end = offsets[end_token - 1][1]
        # Re-tokenization at a boundary can change wordpiece counts.
        while len(tokenizer.encode(prefix + text[start:end]).ids) > max_tokens:
            end_token -= 1
            if end_token <= pos:
                raise ValueError("Cannot fit even one source token")
            end = offsets[end_token - 1][1]
        yield {"start": start, "end": end, "text": text[start:end]}
        if end_token == len(offsets):
            break
        pos = max(pos + 1, end_token - overlap)
