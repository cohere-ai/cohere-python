from typing import NamedTuple

GenerateFeedback = NamedTuple("GenerateFeedback", [("request_id", str), ("prompt", str), ("annotator_id", str),
                                                   ("good_response", bool), ("desired_response", str),
                                                   ("flagged_reason", str), ("flagged_response", bool)])
