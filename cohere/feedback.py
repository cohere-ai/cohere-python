"""Give feedback on a response from the Cohere API to improve the model.

Can be used programmatically like so:

    generations = co.generate(f"Write me a polite email responding to the one below:\n{email}\n\nResponse:")
    if user_accepted_suggestion:
        generations[0].feedback(good_generation=True)
"""

from typing import NamedTuple

Feedback = NamedTuple("Feedback", [("id", str), ("feedback", str), ("good_generation", bool)])
