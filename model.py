"""The model: a very small GPT. It reads the tokens so far and gives a chance to every possible next token."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from task import END, MAX_LEN, MAX_OUT, PAD, VOCAB, ask, questions, reward

WIDTH, LAYERS, HEADS = 64, 2, 4


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(WIDTH), nn.LayerNorm(WIDTH)
        self.qkv, self.proj = nn.Linear(WIDTH, 3 * WIDTH), nn.Linear(WIDTH, WIDTH)
        self.mlp = nn.Sequential(nn.Linear(WIDTH, 4 * WIDTH), nn.GELU(), nn.Linear(4 * WIDTH, WIDTH))

    def forward(self, x):
        B, L, _ = x.shape
        q, k, v = (t.view(B, L, HEADS, -1).transpose(1, 2) for t in self.qkv(self.n1(x)).chunk(3, dim=-1))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)      # each token looks only at earlier tokens
        x = x + self.proj(a.transpose(1, 2).reshape(B, L, WIDTH))
        return x + self.mlp(self.n2(x))


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok, self.pos = nn.Embedding(VOCAB, WIDTH), nn.Embedding(MAX_LEN, WIDTH)
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.norm, self.out = nn.LayerNorm(WIDTH), nn.Linear(WIDTH, VOCAB)

    def forward(self, seq):
        """seq: rows of tokens. Returns, for every position, the log-chance of each possible next token."""
        x = self.tok(seq) + self.pos(torch.arange(seq.shape[1]))
        for b in self.blocks:
            x = b(x)
        return F.log_softmax(self.out(self.norm(x)), dim=-1)


def rows(lines):
    """Lists of tokens -> one padded table of tokens."""
    seq = torch.full((len(lines), MAX_LEN), PAD)
    for i, toks in enumerate(lines):
        seq[i, :len(toks)] = torch.tensor(toks)
    return seq


@torch.no_grad()
def write(model, qs, gen, greedy=False):
    """Let the model answer each question, one token at a time. It picks each token by its chance.

    Returns the full rows and a mask of the positions the model wrote itself.
    """
    seq = rows([ask(q) for q in qs])
    pos = torch.tensor([len(q[1]) + 2 for q in qs])         # where the first written token goes
    wrote = torch.zeros_like(seq, dtype=torch.bool)
    live = torch.ones(len(seq), dtype=torch.bool)
    r = torch.arange(len(seq))
    for _ in range(MAX_OUT):
        logp = model(seq)[r, pos - 1]                              # chances for the next token of every row
        if greedy:
            tok = logp.argmax(1)
        else:
            tok = torch.multinomial(logp.exp(), 1, generator=gen)[:, 0]     # pick a token by its chance
        seq[r[live], pos[live]] = tok[live]
        wrote[r[live], pos[live]] = True
        live = live & (tok != END)
        pos = pos + live
    return seq, wrote


def token_logp(model, seq):
    """The log-chance the model gives to each token that is actually in the row (from the second token on)."""
    logp = model(seq)
    return logp[:, :-1].gather(2, seq[:, 1:, None])[..., 0], logp[:, :-1]


def written(seq, wrote):
    """The tokens the model wrote, per row, as plain lists."""
    return [s[w].tolist() for s, w in zip(seq, wrote)]


def test(model, n=2000, seed=1):
    """Accuracy on test questions the model never trained on, and how much scratch work it writes."""
    qs = questions(n, np.random.default_rng(seed), test=True)
    outs = written(*write(model, qs, torch.Generator().manual_seed(seed)))
    right = np.array([reward(q, o) for q, o in zip(qs, outs)])
    digits = np.array([sum(t < 10 for t in o) for o in outs])    # digits written, the answer included
    keys = np.array([len(q[1]) for q in qs])
    steps = (digits > 1) & (keys > 1)                            # the answer came with scratch work
    direct = (digits <= 1) & (keys > 1)
    return {
        'accuracy': float(right.mean()),
        'digits': float(digits.mean()),
        'steps_share': float(steps.sum() / (keys > 1).sum()),
        'accuracy_steps': float(right[steps].mean()) if steps.any() else None,
        'accuracy_direct': float(right[direct].mean()) if direct.any() else None,
        'by_keys': {int(k): {'accuracy': float(right[keys == k].mean()), 'digits': float(digits[keys == k].mean()),
                             'steps_share': float(steps[keys == k].mean())} for k in sorted(set(keys))},
    }
