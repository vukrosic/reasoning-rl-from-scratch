"""Step 4. The group. Compare each answer with the other answers to the same question."""
import torch

from grpo import advantages, report, save, train
from step1_base import get_base
from step2_reward import Q, group
from task import ask, text

EASY = (4, [('+', 3)])                                    # one key: the base model always gets it
HARD = (0, [('+', 2), ('+', 4), ('+', 6), ('x', 7)])
ITERS = 200


def show(model, q):
    outs, r = group(model, q)
    a = advantages(r[None, :])[0]
    print(f'{text(ask(q))}    rewards {[int(x) for x in r]}    average {r.mean():.3f}    spread {r.std(unbiased=False):.3f}')
    print('   advantages', '  '.join(f'{x:+.2f}' for x in a.tolist()))
    return {'rewards': r.tolist(), 'mean': float(r.mean()), 'std': float(r.std(unbiased=False)), 'adv': a.tolist(),
            'outs': [text(o) for o in outs], 'q': text(ask(q))}


if __name__ == '__main__':
    base = get_base()
    groups = [show(base, q) for q in (Q, EASY, HARD)]
    log = []
    train(get_base(), ITERS, Q=64, G=8, epochs=1, clip=None, beta=0, log=log)
    print('weight of each answer = its advantage. 64 questions x 8 answers per round.')
    report(log)
    save('group', {'groups': groups, 'log': log})
