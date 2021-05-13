class BestChoices:
    def __init__(self, likelihoods, mode) -> None:
        self.likelihoods = likelihoods
        self.mode = mode
    
    def __str__(self) -> str:
        return str(self.likelihoods)
