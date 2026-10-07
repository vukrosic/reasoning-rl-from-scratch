"""Exercise. Thinking is not free: take a little off the reward for every scratch digit. What does the model do?

    python3 exercise_cost.py          cost 0.05 per scratch digit
    python3 exercise_cost.py 0.2      a higher price
"""
import sys

from grpo import save, train
from model import test
from step1_base import get_base, show_test
from task import reward


def with_cost(cost):
    def score(q, written):
        scratch = max(0, sum(t < 10 for t in written) - 1)
        return reward(q, written) - cost * scratch
    return score


if __name__ == '__main__':
    cost = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    model = train(get_base(), 200, Q=64, G=8, epochs=4, clip=0.2, beta=0.04, score=with_cost(cost))
    print(f'reward = right answer (1 or 0) minus {cost} for every scratch digit')
    t = test(model)
    show_test(t)
    save(f'cost-{cost}', t)
