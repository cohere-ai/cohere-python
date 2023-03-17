"""Give feedback on a response from the Cohere API to improve the model.

Can be used programmatically like so:

Example: a user accepts a model's suggestion in an assisted writing setting
```
generations = co.generate(f"Write me a polite email responding to the one below:\n{email}\n\nResponse:")
if user_accepted_suggestion:
    generations[0].feedback(good_response=True)
```

Example: the user edits the model's suggestion
```
generations = co.generate(f"Write me a polite email responding to the one below:\n{email}\n\nResponse:")
if user_edits_suggestion:
    generations[0].feedback(good_response=False, desired_response=user_edited_response)
```
"""

from typing import NamedTuple

GenerateFeedbackResponse = NamedTuple("GenerateFeedbackResponse", [("id", str)])
