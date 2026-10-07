"""Step 3. The first try: make every rewarded answer more likely. One answer per question, weight = its reward."""
from grpo import report, save, train
from step1_base import get_base

ITERS = 200

if __name__ == '__main__':
    log = []
    train(get_base(), ITERS, Q=512, G=1, epochs=1, clip=None, beta=0, baseline=False, log=log)
    print('weight of each answer = its reward (1 or 0). 512 answers per round.')
    report(log)
    save('reinforce', log)
