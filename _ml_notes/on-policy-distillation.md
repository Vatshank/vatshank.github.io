---
title: On Policy Distillation — Thinking Labs
date: 2025-06-18
source: https://thinkingmachines.ai/blog/on-policy-distillation/
source_label: Blog post
published: false
---

Rough notes —

- Discuss with GG — why sample from students in distillation — "computational benefit" (from earlier discussion) vs what they say here which is that off-policy trajectories are less valuable (because "the student model doesn't find itself in situations that the teacher model does")
- [FR] See their citations Dagger paper (imitation learning), "Let's verify step by step" (process reward modeling), Scheduled sampling paper (Bengio et al)
- Motivation for reverse KL — they do mention the computational benefit — discuss with GG
- Distillation for personalization section — all this stuff about mid-training and forgetting is news
    - See paper on ["RL trains subnetworks"](https://arxiv.org/pdf/2505.11711)
    - What are Tulu prompts? What do they mean when they say (see **bold**) —

        > In order to reduce such catastrophic forgetting, a common approach in mid-training is to mix in "background data" from the original model's pretraining distribution. In this case, we don't have access to Qwen3's pretraining distribution. Therefore, we consider a stronger and more expensive baseline: we take **Tulu3 prompts – a broad chat and instruction-following dataset – and re-sample them with Qwen3-8B** in order to act as chat background data.

    - This re-sampled data is used as "forwards KL regularizer" to avoid forgetting.
    - Also wtf is "on-policy SFT" — is it different from regular SFT?
    - It is probably this stuff [Retaining By Doing The Role Of On Policy Data In Mitigating Forgetting](https://arxiv.org/pdf/2510.18874)
- Okay so for their internal personalization use case, the idea seems to be to run two stages —
    - [Mid-training] SFT on their internal company docs. This also mixes in "background data" to avoid forgetting (background data being the Tulu3 prompts that they resampled from their original model). Even after mixing in the background data, performance on IF-eval degrades.
    - [Post-training] On-policy distillation to recover the lost IF-eval performance that was lost in the first stage. They use the earlier version of the model (Qwen-3-8B without their finetuning in stage 1) as the teacher and train on Tulu-3 prompts.

        > The use of an earlier version of the model as a teacher to "re-invoke" capabilities lost during fine-tuning makes on-policy distillation very promising for continuous learning. We could alternate between phases of fine-tuning on new data and distillation to recover behavior to allow our model to learn and stay up-to-date on knowledge over time. This phase-alternating approach has previously been explored by Cobbe et al.

- TODO: Discussion section
- Further reading
    - [Reinforcement Learning Finetunes Small Subnetworks in Large Language Models](https://arxiv.org/pdf/2505.11711)
    - [RETAINING BY DOING: THE ROLE OF ON-POLICY DATA IN MITIGATING FORGETTING](https://arxiv.org/pdf/2510.18874)
    - [Algorithms for inverse RL](https://ai.stanford.edu/~ang/papers/icml00-irl.pdf)
    - [Phasic policy gradients](https://arxiv.org/pdf/2009.04416)
    - [Lora without Regret post](https://thinkingmachines.ai/blog/lora/)
