# Reasoning with reinforcement learning, from scratch (GRPO)

GRPO, the training method behind DeepSeek-R1, built in eight steps on a model small enough to train on a laptop.
A tiny GPT with 104,862 weights learns to show its working on a calculator task. The only feedback it gets is
"the last digit is right" or "the last digit is wrong". 13 Python files, 608 lines, PyTorch.

Companion code for the video "How DeepSeek R1 Learned to Reason".

## Run it

```bash
pip install torch numpy
bash run_all.sh
```

That runs every step and the exercise (`demo-output.txt` is one saved run). The first step trains the base model
and saves it as `base.pt`; the whole script takes about 12 minutes on a laptop CPU. Python 3.9+.

```bash
python3 tests.py
```

The seeds are fixed, but the order of floating-point sums depends on the machine and on how many CPU threads
PyTorch uses, so a rerun can land up to about one percentage point away from `demo-output.txt`. A second run
with 2 threads gave 72.1% → 98.6% where the saved run has 72.2% → 98.8%. The tables below are the saved run.

## The task

A pocket calculator that shows only the last digit. A question is a start digit and 1 to 5 keys:

```
3 x6 x3 +9 +1 =
```

3 x 6 = 18, keep 8. 8 x 3 = 24, keep 4. 4 + 9 = 13, keep 3. 3 + 1 = 4. The answer is 4.
The model writes digits and stops with END. Its answer is the last digit it wrote, so it may write
`4 END` (answer only) or `8 4 3 4 END` (the display after every key, then stop). Both get reward 1.

The base model reads 1,280,000 lines of this text. 1 line in 5 shows the steps; the rest show only the answer.
It ends up answering directly most of the time, and it is often wrong on long questions.

## What it shows

Measured on 2,000 test questions the model never trained on:

|  | base model | after GRPO |
| --- | --- | --- |
| right answers | 72.2% | 98.8% |
| questions with 2+ keys answered with steps | 18.1% | 65.6% |
| right answers, 4 keys | 48.5% | 98.2% |
| right answers, 5 keys | 44.4% | 99.1% |
| digits written, 5 keys | 1.75 | 4.67 |

Nobody tells the model to write the steps. Answers with steps are right more often (96.7% against 59.7% in the
base model), so they beat their group more often, so they get more likely.

The same number of scored answers (102,400) with three ways to weight them:

| weight of each answer | right answers |
| --- | --- |
| its reward, 1 or 0 (step 3) | 82.2% |
| its advantage inside its group (step 4) | 99.0% |
| full GRPO: advantage, clipped step, pull toward the base model (step 6) | 98.8% |

## The steps

| File | What it adds |
| --- | --- |
| `task.py` | The calculator task: questions, worked solutions, the test split and the reward |
| `model.py` | The tiny GPT, how it writes an answer token by token, and the test |
| `grpo.py` | Sampling groups, advantages, the GRPO loss and the training loop |
| `step1_base.py` | Train the base model on text and measure it |
| `step2_reward.py` | One question, 8 sampled answers, a reward of 1 or 0 for each |
| `step3_reinforce.py` | The plain idea: make rewarded answers more likely |
| `step4_group.py` | Compare each answer with its own group: advantage = (reward − group average) / group spread |
| `step5_clip.py` | Reuse a batch for several updates and clip the ratio so one batch cannot move the model too far |
| `step6_train.py` | Full GRPO: 64 questions × 8 answers per round, 200 rounds, clip 0.2, beta 0.04 |
| `step7_what_changed.py` | Base model and trained model side by side, per number of keys |
| `step8_knobs.py` | The two settings: beta 0, 0.04, 1 and groups of 2, 4, 8, 16 |
| `exercise_cost.py` | The exercise: a price on every scratch digit |
| `tests.py` | Checks for the task, the reward, the advantages and the loss |

## The two settings (step 8)

| beta (pull toward the base model) | right answers | answers with steps |
| --- | --- | --- |
| 0 | 99.1% | 89.9% |
| 0.04 | 98.3% | 65.9% |
| 1 | 79.0% | 26.8% |

| group size (512 answers per round) | right answers | answers with steps |
| --- | --- | --- |
| 2 | 97.9% | 63.0% |
| 4 | 97.9% | 66.4% |
| 8 | 98.3% | 65.9% |
| 16 | 98.0% | 71.1% |

## The exercise

Every token a model writes costs time and money. `exercise_cost.py` takes 0.05 off the reward for every scratch
digit. On which questions will the model stop showing its steps? Guess first, then run:

```bash
python3 exercise_cost.py
python3 exercise_cost.py 0.2
```

## How this differs from the real thing

- DeepSeek-R1-Zero starts from DeepSeek-V3-Base, a large language model, and writes thousands of tokens of
  reasoning. This model has 104,862 weights and writes at most a handful of digits.
- The R1 reward checks the final answer and the format (thinking inside think tags). Ours checks the last digit.
- The GRPO paper (DeepSeekMath) samples 64 answers per question. We sample 8.
- We compute the exact KL term over a 30-token vocabulary against a frozen copy of the base model.
- Our base model has read worked solutions in 1 of 5 lines, so the skill is already there. Reinforcement
  learning makes the model use it.
- DeepSeek-R1 (not Zero) adds example solutions before the reinforcement learning and more training after it.

## Sources

- DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning, arXiv 2501.12948
- DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models, arXiv 2402.03300 (GRPO)
- Proximal Policy Optimization Algorithms, arXiv 1707.06347 (the clipped step)

MIT license.
