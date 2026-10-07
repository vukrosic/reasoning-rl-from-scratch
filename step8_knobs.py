"""Step 8. Two knobs: how hard the model is pulled back to its start (beta), and the size of the group."""
from grpo import save, train
from model import test
from step1_base import get_base

ITERS = 100


def run(**kw):
    model = train(get_base(), ITERS, epochs=4, clip=0.2, **kw)
    t = test(model)
    print(f"   accuracy {t['accuracy']:.1%}   shows steps {t['steps_share']:.1%}   digits written, by keys: "
          + '  '.join(f"{k}: {v['digits']:.1f}" for k, v in t['by_keys'].items()))
    return t


if __name__ == '__main__':
    out = {'beta': {}, 'group': {}}
    for beta in (0.0, 0.04, 1.0):
        print(f'beta {beta}  (groups of 8)')
        out['beta'][str(beta)] = run(Q=64, G=8, beta=beta)
    for G in (2, 4, 16):
        print(f'groups of {G}  ({512 // G} questions per round, beta 0.04)')
        out['group'][str(G)] = run(Q=512 // G, G=G, beta=0.04)
    out['group']['8'] = out['beta']['0.04']
    save('knobs', out)
