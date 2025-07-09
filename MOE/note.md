# Lecture 4 Mixtures of Experts (MoE)

# What's a MOE?

Replace big feedforward with (many) big feedforward networks and a selector layer 

You can increase the # experts without affecting the FLOPs

# Why are MOEs getting popular?

- Same FLOPs, more parameters do better

- Parallelizable to many devices

# Why haven't MoEs been popular?
- Infrastructure is complex / advantages on multi node
- Training objectives are somewhat heurstic (and sometimes unstable)

# What MoEs generally look like?

Typical: replace MLP with MoE layer

Less common: MoE for attention heads

# MoE - what varies?

- Routing function
    - Token chooses expert 
        1. (top-k used in most MoEs)
        2. Hashing (common baseline)
    - Expert chooses token
    - Global routing via optimization

    - Other routing methods
        - RL to learn routes
        - Solve a matching problem [Clark'22]
    
    - 1. Conventional Top-2 Routing
    - 2. + Fine-grained Expert Segmentation
    - 3. + Shared Expert Isolation (DeepSeekMoE)

- Expert sizes

- Training objectives


# How do we train MoEs?
Major Challenge:
we need sparsity for training-time efficiency..
But sparse gating decisions are not differentiable

# RL for MoEs
RL via REINFORCE does work, but not so much better that it's a clear win

# Stochastic approximations

# Heurtistic balancing losses

Another key issue - systems efficiency requires that we use experts evenly

Per-expert balancing

Per-device balancing

# Per-expert biases (DeepSeek v3)

# Training MoEs - the systems side
- MoEs parallelize nicely - Each FFN can fit in a device
- Enables additional kinds of parallelism

MoE rounting allows for parallelism, but also some complexities

Modern libraries like MegaBlocks (used in many open MoEs) use smarter sparse MMs

- Z-loss function for the stability of the routing

Issues with MoEs - fine-tuning
    - Sparse MoEs can overfit on smaller fine-tuning data
    - Zoph et al. solution - finetune non-MoE MLPs
    - DeepSeek solution - use lots of data 1.4M SFT

upcycling

Can we use a pre-trained LM to initialize a MoE?

MiniCPM

Qwen Moe

MLA: Multihead, latent attention

Complexity: rope conflicts with MLA-style caching

MTP (multiple token prediction): Have small, lightweight models that predict multiple steps

# MoE summary

- MoEs take avantage of sparsity - not all inputs need the full model

- Discrete routing is hard, but top-k heurtics seem to work

- Lots of empirical evidence now that MoEs work, and are cost-effective




