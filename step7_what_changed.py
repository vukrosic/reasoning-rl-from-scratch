"""Step 7. What changed? Same test questions, the base model next to the model after GRPO."""
import numpy as np
import torch

from grpo import save
from model import test, write, written
from step1_base import get_base, show_test
from step2_reward import Q, first_token_chances, group, show_group
from step6_train import get_trained
from task import ask, questions, reward, text, values

if __name__ == '__main__':
    base, new = get_base(), get_trained()
    out = {}
    for name, model in (('base', base), ('after GRPO', new)):
        print(f'--- {name}')
        t = test(model)
        show_test(t)
        print(f'the question from step 2, {text(ask(Q))}')
        outs, r = group(model)
        show_group(outs, r)
        out[name] = {'test': t, 'group': [text(o) for o in outs], 'rewards': r.tolist(), 'first': first_token_chances(model)}
    print('--- one answer each to five more test questions (right answer in brackets)')
    qs = [questions(1, np.random.default_rng(40 + k), test=True, sizes=[k])[0] for k in (1, 2, 3, 4, 5)]
    rows_ = []
    for q in qs:
        a = [text(written(*write(m, [q], torch.Generator().manual_seed(3)))[0]) for m in (base, new)]
        rows_.append({'q': text(ask(q)), 'right': values(q)[-1], 'base': a[0], 'new': a[1]})
        print(f"   {text(ask(q)):<22} [{values(q)[-1]}]   base: {a[0]:<16} after: {a[1]}")
    out['examples'] = rows_
    save('changed', out)
