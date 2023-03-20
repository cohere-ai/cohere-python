from typing import Any, Dict, NamedTuple, Optional

SummarizeResponse = NamedTuple("SummarizeResponse", [("id", str), ("summary", str), ("meta", Optional[Dict[str, Any]])])
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
    temperature=0.3,
    additional_command="focusing on the highest performing stocks")
print(res.summary)
```
"""
