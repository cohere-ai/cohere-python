from typing import NamedTuple

SummarizeResponse = NamedTuple("SummarizeResponse", [("id", str), ("summary", float)])
SummarizeResponse.__doc__ = """
Returned by co.summarize, which generates a summary of the specified length for the provided text.

Example:
```
res = co.summarize(text="Stock market report for today...")
print(res.summary)
```

Example:
```
res = co.summarize(
    text="Stock market report for today...",
    model="summarize-xlarge",
    length="long",
    format="bullets",
    additional_instruction="focusing on the highest performing stocks")
print(res.summary)
```
"""
