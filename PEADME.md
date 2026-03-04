# Text2SQL RLHF

This project implements a **Text-to-SQL system using Reinforcement Learning from Human Feedback (RLHF)** on the **Spider dataset**.

## Models Tested

- T5-Small
- CodeT5-Base
- BART

CodeT5 achieved the best performance and was selected as the final model.

## Results

| Model | SFT | RLHF |
|------|------|------|
| T5-Small | 9% | 8.3% |
| CodeT5-Base | 41.7% | 37.9% |
| BART | 24% | 21.23% |

## Run Evaluation
python src/eval_rl_fixed.py --adapter checkpoints/sft_adapter_codet5


## Author

Tanisha Jhalani
