# Delayed-label methods

LateSignal compares delayed-label strategies under a shared conversion model, sampler, fixed schedule, and core optimizer-example budget.
The implementations in this repository are transfers to the Sponsored Search dataset and are not claims that published numbers were reproduced.

## Event-stream baselines

Complete-cohort wait emits one final BCE record at the 30-day attribution boundary.
A positive known earlier remains withheld until its cohort is complete.

Immediate fake-negative BCE emits a provisional negative at click time and a separate positive correction at legal reveal time for a converting click.
The two records remain independently sampleable and share the click identifier as their correction group.

Fixed wait chooses one of 1, 3, 7, or 14 days on the selection period.
An early conversion emits one positive record at reveal, while an unresolved click emits a provisional negative at the wait boundary and a later conversion adds a separate correction.

## Delayed Feedback Model transfer

The DFM transfer follows the joint exponential-delay likelihood in Olivier Chapelle's [Modeling delayed feedback in display advertising](https://doi.org/10.1145/2623330.2623634).
For a revealed positive it uses `-log(p) - log(lambda) + lambda * delay`.
For an unresolved or mature-negative click it uses the stable right-censored term `-log((1-p) + p*exp(-lambda*elapsed))`.
Mature negatives remain censored at the 30-day attribution boundary because the data establishes nonconversion only inside that window.

## Fake Negative Weighted transfer

FNW follows Sofia Ira Ktena et al., [Addressing Delayed Feedback for Continuous Training with Neural Networks in CTR prediction](https://arxiv.org/abs/1907.06558), RecSys 2019.
It uses the immediate fake-negative duplicate stream and recomputes detached pre-update conversion probabilities on every exposure.
Its positive weight is `1 + p` and its negative weight is `1 - p^2`.
The same formulation is also evaluated in [A Feedback Shift Correction in Predicting Conversion Rates under Delayed Feedback](https://arxiv.org/abs/2002.02068) and implemented by the behavioral reference named by the ES-DFM authors.

## ES-DFM constant-wait transfer

The constant-wait transfer follows equations 13 through 17 of [Capturing Delayed Feedback in Conversion Rate Prediction via Elapsed-Time Sampling](https://arxiv.org/abs/2012.03245).
The positive main-model weight is `1 + q_dp` and the negative weight is `(1 + q_dp) * q_tn`.
Auxiliary probabilities are detached, probability-clamped, and the resulting weights are clipped to the authored `[1e-4, 2.0]` interval.

The official reference implementation was inspected only as a behavioral reference at commit `7f66101916db08d926b721153a874fc19eac21d3`.
That repository is BSD-3-Clause and requires an explicit citation when its implementation is used, presented, compared, or evaluated.
LateSignal therefore cites Jia-Qi Yang, Xiang Li, Shuguang Han, Tao Zhuang, De-Chuan Zhan, Xiaoyi Zeng, and Bin Tong, "Capturing Delayed Feedback in Conversion Rate Prediction via Elapsed-Time Sampling," AAAI 2021, pages 4582-4589.
No TensorFlow source from the reference repository is copied into this PyTorch implementation.

## Oracle boundary

The oracle emits eventual truth at click time through an explicitly privileged constructor.
It is marked nondeployable and ineligible for every deployable or equal-compute ranking.
