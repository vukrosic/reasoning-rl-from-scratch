"""GRPO: sample a group of answers to each question, score them, push up the ones that beat their own group."""
import copy
import json
import os

import numpy as np
import torch

from model import test, token_logp, write, written
from task import questions, reward


def sample_groups(model, qs, G, gen, score=reward):
    """G answers to every question, written by the model as it is now. Rewards come back as a Q x G table."""
    rep = [q for q in qs for _ in range(G)]
    seq, wrote = write(model, rep, gen)
    r = torch.tensor([score(q, w) for q, w in zip(rep, written(seq, wrote))]).view(len(qs), G)
    return seq, wrote, r


def advantages(r, normalize=True):
    """How much better each answer did than the average of its own group, in units of the group's spread."""
    a = r - r.mean(1, keepdim=True)                              # minus the average of its own group
    if normalize:
        a = a / (r.std(1, keepdim=True, unbiased=False) + 1e-6)
    return a


def grpo_loss(model, ref, seq, wrote, old_lp, adv, clip=0.2, beta=0.04):
    """The number that training pushes down. Lower means: answers with a positive advantage got more likely."""
    lp, all_lp = token_logp(model, seq)
    ratio = (lp - old_lp).exp()                                  # new chance / old chance, for every token
    A = adv.reshape(-1, 1)
    gain = ratio * A
    if clip is not None:
        gain = torch.minimum(gain, ratio.clamp(1 - clip, 1 + clip) * A)     # no credit for moving too far
    if beta:
        with torch.no_grad():
            ref_lp = token_logp(ref, seq)[1]
        gain = gain - beta * (all_lp.exp() * (all_lp - ref_lp)).sum(-1)      # stay close to the starting model
    m = wrote[:, 1:]                                             # only the tokens the model wrote count
    return -((gain * m).sum(1) / m.sum(1)).mean()


def train(model, iters, Q=64, G=8, epochs=4, lr=3e-4, clip=0.2, beta=0.04,
          baseline=True, normalize=True, score=reward, seed=0, log=None, every=10):
    """The GRPO loop. With baseline=False the raw reward is used instead of the advantage (plain REINFORCE)."""
    rng = np.random.default_rng(seed + 100)
    gen = torch.Generator().manual_seed(seed + 100)
    ref = copy.deepcopy(model).eval()                            # the frozen starting model
    opt = torch.optim.Adam(model.parameters(), lr)
    for it in range(iters + 1):
        if log is not None and it % every == 0:
            log.append({'iter': it, 'answers': it * Q * G, **test(model)})
        if it == iters:
            break
        seq, wrote, r = sample_groups(model, questions(Q, rng), G, gen, score)
        adv = advantages(r, normalize) if baseline else r
        with torch.no_grad():
            old_lp = token_logp(model, seq)[0]                   # the chances at the moment of sampling
        for _ in range(epochs):
            loss = grpo_loss(model, ref, seq, wrote, old_lp, adv, clip, beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def report(log):
    """Print a training log: answers scored so far, accuracy on the test questions, share that shows steps."""
    print('  answers scored   accuracy   shows steps')
    for e in log:
        print(f"  {e['answers']:>14,}   {e['accuracy']:>8.1%}   {e['steps_share']:>11.1%}")


def save(name, data):
    """Keep a result in runs/ so later steps and the README numbers can read it."""
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs'), exist_ok=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', name + '.json'), 'w') as f:
        json.dump(data, f, indent=1)
