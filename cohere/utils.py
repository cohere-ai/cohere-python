import json


try: # numpy is optional, but support json encoding if the user has it
    import numpy as np


    class NumpyAwareJsonEncoder(json.JSONEncoder):
        """Handles numpy datatypes and such in json encoding"""

        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, set):
                return list(obj)
            else:
                return super().default(obj)
except:
    class NumpyAwareJsonEncoder(json.JSONEncoder):
        """Handles numpy datatypes and such in json encoding"""

        def default(self, obj):
            if isinstance(obj, set):
                return list(obj)
            else:
                return super().default(obj)    

def np_json_dumps(data, **kwargs):
    return json.dumps(data, cls=NumpyAwareJsonEncoder, **kwargs)
