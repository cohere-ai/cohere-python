from cohere.response import CohereObject


class Images(CohereObject):
    '''
    Represents the main response of calling the Image API. An Images is a single base64 encoded string.
    '''

    def __init__(self, image: str) -> None:
        self.image = image
