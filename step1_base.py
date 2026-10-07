"""Step 1. The base model. It learned by reading worked lines, the way a language model learns by reading text.

Four lines in five give only the answer. One line in five shows the display after every key.
"""
import os

import numpy as np
import torch

from model import TinyGPT, rows, test, token_logp
from task import PAD, questions, text, worked

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS, BATCH, SHARE = 5000, 256, 0.2


def pretrain():
    """Read STEPS x BATCH lines. For every token after the '=', push up the chance of the token that is there."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = TinyGPT()
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for _ in range(STEPS):
        qs = questions(BATCH, rng)
        seq = rows([worked(q, rng.random() < SHARE) for q in qs])
        lp, _ = token_logp(model, seq)
        after = torch.arange(seq.shape[1])[None, :] > torch.tensor([len(q[1]) + 1 for q in qs])[:, None]
        m = (after & (seq != PAD))[:, 1:]
        loss = -(lp * m).sum() / m.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def get_base():
    """The base model. Trained once (about a minute and a half), then loaded from base.pt."""
    path = os.path.join(HERE, 'base.pt')
    if not os.path.exists(path):
        torch.save(pretrain().state_dict(), path)
    model = TinyGPT()
    model.load_state_dict(torch.load(path))
    return model


def show_test(t):
    print(f"accuracy on 2,000 test questions: {t['accuracy']:.1%}")
    print(f"questions with 2 or more keys answered with steps: {t['steps_share']:.1%}")
    print(f"  right when it shows steps: {t['accuracy_steps']:.1%}    right when it only answers: {t['accuracy_direct']:.1%}")
    print('  keys   accuracy   digits written   shows steps')
    for k, v in t['by_keys'].items():
        print(f"  {k:>4}   {v['accuracy']:>8.1%}   {v['digits']:>14.2f}   {v['steps_share']:>11.1%}")


if __name__ == '__main__':
    rng = np.random.default_rng(5)
    print('three lines of the text it reads:')
    for q, steps in zip(questions(3, rng, sizes=[4]), (False, True, False)):
        print('  ', text(worked(q, steps)))
    model = get_base()
    print(f'lines read: {STEPS * BATCH:,}    weights in the model: {sum(p.numel() for p in model.parameters()):,}')
    show_test(test(model))
