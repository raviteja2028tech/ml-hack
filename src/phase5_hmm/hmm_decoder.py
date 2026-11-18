"""Phase 5: Hidden Markov Model & Viterbi Sequential Route Reconstruction."""

class HMMRouteDecoder:
    def __init__(self, sigma_z=4.29, beta=5.0):
        self.sigma_z = sigma_z
        self.beta = beta
