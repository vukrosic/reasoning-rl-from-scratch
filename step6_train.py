"""Step 6. The whole GRPO loop: groups, advantages, the clipped step, and a pull back toward the starting model."""
import os

import torch

from grpo import report, save, train
from model import TinyGPT
from step1_base import HERE, get_base

ITERS = 200


def get_trained():
    """The model after GRPO. Trained once (about a minute), then loaded from grpo.pt."""
    path = os.path.join(HERE, 'grpo.pt')
    if not os.path.exists(path):
        log = []
        model = train(get_base(), ITERS, Q=64, G=8, epochs=4, clip=0.2, beta=0.04, log=log)
        torch.save(model.state_dict(), path)
        save('grpo', log)
        report(log)
    model = TinyGPT()
    model.load_state_dict(torch.load(path))
    return model


if __name__ == '__main__':
    print('GRPO: 64 questions x 8 answers per round, 4 updates per batch, clip 0.2, beta 0.04')
    get_trained()
