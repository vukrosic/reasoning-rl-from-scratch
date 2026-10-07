"""The task: a pocket calculator that shows only the last digit.

A question is a start digit and 1 to 5 keys, each "+k" or "xk":      3 x7 +5 x4 =
3 x 7 = 21, keep 1.  1 + 5 = 6.  6 x 4 = 24, keep 4.  The answer is 4.
The model writes digits and stops with END. Its answer is the last digit it wrote.
"""
import zlib

import numpy as np

MAX_KEYS = 5                                             # at most 5 keys in one question
ADD = {k: 10 + k - 1 for k in range(1, 10)}              # tokens 10..18 are the keys +1 .. +9
MUL = {k: 19 + k - 2 for k in range(2, 10)}              # tokens 19..26 are the keys x2 .. x9
KEYS = [('+', k) for k in ADD] + [('x', k) for k in MUL]
EQ, END, PAD = 27, 28, 29
VOCAB = 30                                               # tokens 0..9 are the digits themselves
MAX_OUT = 7                                              # the model may write at most 7 tokens
MAX_LEN = MAX_KEYS + 2 + MAX_OUT                         # the longest question is 7 tokens


def press(value, key):
    """One key press. Only the last digit stays."""
    op, k = key
    return (value + k) % 10 if op == '+' else (value * k) % 10


def is_test(q):
    """One question in ten is kept for testing. The model never trains on those."""
    return zlib.crc32(repr(q).encode()) % 10 == 0


def questions(n, rng, test=False, sizes=range(1, MAX_KEYS + 1)):
    """n random questions: (start digit, list of keys). Training and test questions never overlap."""
    out = []
    while len(out) < n:
        q = (int(rng.integers(0, 10)), [KEYS[i] for i in rng.integers(0, len(KEYS), size=rng.choice(sizes))])
        if is_test(q) == test:
            out.append(q)
    return out


def values(q):
    """The display after every key press. The last one is the right answer."""
    v, out = q[0], []
    for key in q[1]:
        v = press(v, key)
        out.append(v)
    return out


def ask(q):
    """The question as tokens: 3 x7 +5 x4 ="""
    return [q[0]] + [(ADD if op == '+' else MUL)[k] for op, k in q[1]] + [EQ]


def worked(q, steps):
    """A correct line of text. With steps it shows the display after every key; without, only the answer."""
    v = values(q)
    return ask(q) + (v if steps else v[-1:]) + [END]


def answer(written):
    """The model's answer: the digit right before END. None if it never stopped or wrote no digit."""
    if END not in written:
        return None
    i = written.index(END)
    return written[i - 1] if i > 0 and written[i - 1] < 10 else None


def reward(q, written):
    """1 for the right final answer, 0 for anything else. The scratch work is never looked at."""
    return 1.0 if answer(written) == values(q)[-1] else 0.0


NAMES = {**{t: f'+{k}' for k, t in ADD.items()}, **{t: f'x{k}' for k, t in MUL.items()}, EQ: '=', END: 'END', PAD: ''}


def text(toks):
    """Tokens as readable text."""
    return ' '.join(NAMES.get(t, str(t)) for t in toks).strip()
