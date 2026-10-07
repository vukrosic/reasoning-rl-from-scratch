"""Step 5. Use each batch of answers many times, and clip the step so the model cannot run away from it."""
import torch

from grpo import report, save, train
from step1_base import get_base

ITERS, REUSE = 60, 16


def clipped(ratio, adv, clip=0.2):
    """What one token contributes: the smaller of the plain gain and the clipped gain."""
    r = torch.tensor(ratio)
    return float(torch.minimum(r * adv, r.clamp(1 - clip, 1 + clip) * adv))


if __name__ == '__main__':
    print('ratio = new chance / old chance.   gain kept after the clip:')
    print('   ratio    good answer (A = +1)    bad answer (A = -1)')
    table = []
    for ratio in (0.5, 0.8, 1.0, 1.2, 1.5, 3.0):
        table.append([ratio, clipped(ratio, 1.0), clipped(ratio, -1.0)])
        print(f'   {ratio:>5.1f}    {table[-1][1]:>20.2f}    {table[-1][2]:>19.2f}')
    out = {'table': table}
    for name, clip in (('no clip', None), ('clip 0.2', 0.2)):
        log = []
        train(get_base(), ITERS, Q=64, G=8, epochs=REUSE, clip=clip, beta=0, log=log)
        print(f'{REUSE} updates on every batch, {name}:')
        report(log)
        out[name] = log
    save('clip', out)
