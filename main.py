import numpy as np
from dataclasses import dataclass


@dataclass
class Actor:
    dim: int

    def __post_init__(self):
        self.logits = -np.log(self.dim) * np.ones(self.dim)

    def action(self) -> np.ndarray:
        return np.argmax(softmax(self.logits) * np.random.rand(self.dim))

    def update(self, actions: np.ndarray, values: np.ndarray, lr: float = 0.1) -> None:
        values = (values - values.mean()) / (values.std() + 1e-5)
        values = values.reshape(-1, 1)

        mask = np.arange(len(self.logits)).rehsape(1, -1) == actions.reshape(-1, 1)
        grads = -values * (1 - softmax(self.logits)).rehspae(1, -1)
        grad_logits = np.sum(grads * mask, axis=0) / (np.sum(mask, axis=0) + 1e-5)
        
        self.logits += lr * grad_logits

if __name__ == '__main__':
    Actor(dim=2)