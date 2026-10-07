"""Step 2. Ask the base model one question 8 times, and score every answer with one rule."""
import torch

from grpo import sample_groups
from model import rows, written
from step1_base import get_base
from task import ask, text, values

Q = (3, [('x', 6), ('x', 3), ('+', 9), ('+', 1)])        # 3 x6 x3 +9 +1 =      3 -> 8 -> 4 -> 3 -> 4


def group(model, q=Q, G=8, seed=7):
    """G answers to one question, and the reward of each."""
    seq, wrote, r = sample_groups(model, [q], G, torch.Generator().manual_seed(seed))
    return written(seq, wrote), r[0]


def first_token_chances(model, q=Q):
    """The chance the model gives to each digit as the first token it writes."""
    toks = ask(q)
    with torch.no_grad():
        return model(rows([toks]))[0, len(toks) - 1].exp()[:10].tolist()


def show_group(outs, r):
    for o, x in zip(outs, r):
        print(f'   {text(o):<14} reward {x:.0f}')
    print(f'   {int(r.sum())} of {len(r)} right')


if __name__ == '__main__':
    model = get_base()
    print('question:', text(ask(Q)), '   display after each key:', values(Q), '   right answer:', values(Q)[-1])
    print('chance of each digit as the first token:')
    print('  ', '  '.join(f'{d}: {p:.0%}' for d, p in enumerate(first_token_chances(model))))
    print('8 answers from the same model to the same question:')
    show_group(*group(model))
