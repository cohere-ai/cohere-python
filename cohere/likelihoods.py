class Likelihoods:
    def __init__(self, likelihood, token_likelihoods) -> None:
        self.likelihood = likelihood
        self.token_likelihoods = token_likelihoods

    def __str__(self) -> str:
        return str(self.likelihood) + "\n" + str(self.token_likelihoods)
