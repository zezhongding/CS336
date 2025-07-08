# Transformer Architecture

# Pre-vs-post norm

# 'double' norm
# - Pre-norm: LayerNorm before the residual connection
# - Post-norm: LayerNorm after the residual connection
# - Double norm: LayerNorm before and after the residual connection

- LayerNorm vs RMSNorm
  - LayerNorm: Normalizes across the features (GPT)
  - RMSNorm: Normalizes across the features but uses the root mean square instead of the mean (LLaMA-family)

why use RMSNorm?
- fewer operations (no mean calculation)
- fewer parameters (no learnable bias)

- Basically everyone does pre-norm 
- Most people do RMSNorm now

- Activations
ReLU
GeLU (GPT 1/2/3)



GeGLU (Gemma)
SWiGLU (LLaMA)

- Serial vs Parallel

- Summary:
    - Pre-norm is preferred over post-norm
    - RMSNorm is preferred over LayerNorm
    - Gating: GLUs seem generally better, though differences are small
    - Serial architectures are more common than parallel architectures (fusion kernels)

- Postion embeddings
    - Sine embeddings

- Hyper Parameters

d_ff = 4 * d_model

Exception #1 - GLU variants
    - GLU variants scale down by 2/3 
    - d_ff = 8 / 3 * d_model

Exception #2 - T5
    - d_ff = 65536
    - d_model = 1024

- Num Heads * Head Dim = d_model

- Aspect Ratio
    - d_model / n_layer

- Vocabulary sizes
    - monolingual [单语言系统]: 30-50k vocab
    - multilingual [多语言系统]: 100-250k vocab

- Regularization

- Dropout and weight decay
    - Dropout: 0.1-0.3
    - Weight decay: 0.01-0.1

- Summary: hyperparameters
    - Feedforward
        - Factor-of-4 rule of thumb (8/3 for GLUs) is standard (with some evidence)
    - Head dim
        - head dim * num heads = d_model is standard
    - Aspect ratio
        - wide range of 'good' values (100-200). Systems concerns dictate the value
    - Regularization
        - You still 'regularize' your model, but its effects are primarily on optimizattion dynamic

- Stability tricks
    - Softmax
    - z-loss trick
    - QK normalization (layer norm on Q and K)
    - Logit soft-capping

- GQA/MQA:

- MQA (Multi-query): just have fewer key dimensions.
    key idea: have multiple queries, but just one dimension for keys and values

- GQA (Grouped-query attention): have multiple queries, but group them into a smaller number of keys and values.

