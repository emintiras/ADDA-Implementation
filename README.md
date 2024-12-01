# ADDA-Implementation

This repository contains an implementation of the [Adversarial Discriminative Domain Adaptation (ADDA)](https://arxiv.org/abs/1702.05464) algorithm using PyTorch.
## Features
- **Source Domain Training**: Pretraining a source encoder and classifier.
- **Target Domain Adaptation**: Using adversarial learning to adapt to the target domain.
- **Evaluation Metrics**: Assessing model performance on both source and target datasets.

---

## Requirements
To run this project, ensure you have the following installed:
- Python 3.8+
- PyTorch 1.9+

---

## Result

|                                    | MNIST (Source) | USPS (Target) |
| :--------------------------------: | :------------: | :-----------: |
| Source Encoder + Source Classifier |   99.18%   |  42.00%   |
| Target Encoder + Source Classifier |                |  93.02%   |