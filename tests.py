"""Checks for the task, the reward, the advantages and the loss.    python3 tests.py"""
import numpy as np
import torch

from grpo import advantages, grpo_loss, sample_groups
from model import TinyGPT, token_logp, write, written
from task import ADD, END, EQ, MUL, answer, ask, is_test, questions, reward, text, values, worked

Q = (3, [('x', 7), ('+', 5), ('x', 4)])


def test_task():
    assert values(Q) == [1, 6, 4]
    assert text(ask(Q)) == '3 x7 +5 x4 ='
    assert text(worked(Q, True)) == '3 x7 +5 x4 = 1 6 4 END' and text(worked(Q, False)) == '3 x7 +5 x4 = 4 END'
    assert len(set(ADD.values()) | set(MUL.values()) | {EQ, END}) == 19


def test_reward():
    assert reward(Q, [4, END]) == 1 and reward(Q, [1, 6, 4, END]) == 1      # steps or no steps: same reward
    assert reward(Q, [9, 9, 4, END]) == 1                                      # the scratch work is never looked at
    assert reward(Q, [1, 6, 5, END]) == 0 and reward(Q, [4]) == 0 and reward(Q, [END]) == 0
    assert answer([1, 6, 4, END, 7]) == 4


def test_split():
    rng = np.random.default_rng(0)
    train, test = questions(300, rng), questions(300, rng, test=True)
    assert not any(is_test(q) for q in train) and all(is_test(q) for q in test)


def test_advantages():
    r = torch.tensor([[0., 1, 0, 0, 1, 0, 1, 1], [1.] * 8, [0.] * 8])
    a = advantages(r)
    assert torch.allclose(a[0], torch.tensor([-1., 1, -1, -1, 1, -1, 1, 1]), atol=1e-4)
    assert a[1].abs().max() == 0 and a[2].abs().max() == 0                     # everyone equal: nothing to learn
    assert torch.allclose(a.sum(1), torch.zeros(3), atol=1e-4)


def test_write_and_loss():
    torch.manual_seed(0)
    model = TinyGPT()
    qs = questions(4, np.random.default_rng(0))
    seq, wrote, r = sample_groups(model, qs, 8, torch.Generator().manual_seed(0))
    assert seq.shape[0] == 32 and r.shape == (4, 8)
    for q, out in zip([q for q in qs for _ in range(8)], written(seq, wrote)):
        assert 1 <= len(out) <= 7 and END not in out[:-1]
    old = token_logp(model, seq)[0].detach()
    adv = advantages(torch.tensor([[0., 1, 0, 0, 1, 0, 1, 1]] * 4))
    # before any update the ratio is exactly 1, so the loss is minus the average advantage: 0
    assert abs(grpo_loss(model, model, seq, wrote, old, adv).item()) < 1e-5
    # the gradient only comes from tokens the model wrote, never from the question
    loss = grpo_loss(model, model, seq, wrote, old, adv)
    loss.backward()
    assert model.out.weight.grad.abs().sum() > 0


if __name__ == '__main__':
    for name, f in list(globals().items()):
        if name.startswith('test_'):
            f()
            print('ok', name)
