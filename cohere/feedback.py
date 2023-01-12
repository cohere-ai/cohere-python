from typing import NamedTuple

Feedback = NamedTuple("Feedback", [("id", str), ("feedback", str), ("good_generation", bool)])
