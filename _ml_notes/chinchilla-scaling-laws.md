---
title: Chinchilla Scaling Laws (Hoffman et al. 2022)
date: 2025-06-19
source: https://arxiv.org/abs/2203.15556
source_label: arXiv
---

Core question the paper asks: given a fixed training budget (aka compute aka FLOPs), how do you allocate it between model size vs dataset size?

Ans: For a dense transformer architecture, we need to scale model size and dataset size equally i.e. the power law coefficient for both is ~0.5. The earlier scaling laws work ([Kaplan et al. 2020](https://arxiv.org/abs/2001.08361)) had estimated the power law co-efficient to be ~0.75 (more compute allocated to model scaling and less to data scaling) and the models at the time (early 2022) were under-trained.

How:

- They train models with different FLOPs budget spread across different model and dataset sizes and look at the optimal training loss across different FLOPs allocations. They then fit a power law $L = P + Q/N^\alpha + R/D^\beta$ where $L$ is loss, $N$ is model size, and $D$ is dataset size. They estimate the power law coefficients (they actually do this estimation 3 different ways and the coefficients roughly agree across all 3 methods) and end up with $\alpha \approx \beta \approx 0.5$. In contrast, Kaplan et al. had estimated $\alpha \approx 0.75$ and $\beta \approx 0.25$ (more compute allocated to model scaling and less to data scaling).
- And then to make their point that the models at the time (~2021-2022) were under-trained, they take the compute budget of the Gopher 280B model that was trained on ~300B tokens and use it to train Chinchilla, a 70B model (4x smaller) and train it on 1.4 T tokens (4x higher) and show that Chinchilla has a much better performance on downstream tasks than Gopher (despite using the same FLOPs budget).


Question: How did the Kaplan paper end up with such a different estimate of the power law coefficients?

Ans: Because of the following differences in methodology (see [Pearce and Song 2024](https://arxiv.org/abs/2406.12907) for details):
1. They trained smaller models in Kaplan et al. The biggest model they trained for deriving the scaling laws was 1.5B compared to 16B for the Chinchilla paper.
2. Kaplan et al. did not include the embedding parameters in the total parameter count.

These two differences ended in different estimates of the power law coefficients.

Follow-up reading:

- MoE scaling laws. See the [Joint MoE Scaling Laws](https://arxiv.org/abs/2502.05172) paper.
- Including inference costs in the calculation. Need to read [Beyond Chinchilla Optimal](https://arxiv.org/pdf/2401.00448). Since small models are cheaper to run inference on and you only train once, choosing a smaller model (compared to what you would if you were following Chinchilla scaling laws) and training it for longer (since you are no longer "Chinchilla optimal") is not a bad idea.
