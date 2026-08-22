import torch
import torch.nn as nn


class TwinDynamics(nn.Module):
    """
    Residual deep state-transition dynamics operator:
    S_{t+1} = S_t + Δ(S_t)
    """

    def __init__(self, state_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        delta = self.net(state)
        return state + delta


class DigitalTwin(nn.Module):
    """
    Digital Twin System implementing core physiological twin operators:
    - initialize: Instantiates twin state from patient observation
    - update: Assimilates real-time patient observations
    - scenario: Applies counterfactual clinical perturbations (S~_t = S_t + delta)
    - evolve: Advances twin state 1 time-step forward
    - rollout: Autoregressively forecasts multi-step patient trajectories
    """

    def __init__(self, state_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.dynamics = TwinDynamics(state_dim=state_dim, hidden_dim=hidden_dim)
        self.state_dim = state_dim

    def initialize(self, state: torch.Tensor) -> torch.Tensor:
        """Initializes digital twin state from observed patient state."""
        return state.clone()

    def update(self, twin_state: torch.Tensor, observed_state: torch.Tensor) -> torch.Tensor:
        """Assimilates newly observed patient state into digital twin."""
        return observed_state.clone()

    def scenario(self, twin_state: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Applies hypothetical scenario perturbation to twin state."""
        return twin_state + delta

    def evolve(self, twin_state: torch.Tensor) -> torch.Tensor:
        """Advances digital twin state one discrete step forward."""
        return self.dynamics(twin_state)

    def rollout(self, initial_state: torch.Tensor, steps: int = 60) -> torch.Tensor:
        """
        Recursive multi-step forward rollout of digital twin trajectory:
        Returns tensor of shape [batch_size, steps, state_dim].
        """
        states = []
        curr = initial_state
        for _ in range(steps):
            curr = self.evolve(curr)
            states.append(curr)
        return torch.stack(states, dim=1)
