# knowledge / attention-mechanism

**Prompt:** Explain how the self-attention mechanism in a transformer works, to a reader who understands linear algebra but has not studied deep learning.

Cover:
1. What queries, keys, and values are, and where they come from
2. The scaled dot-product attention computation, step by step
3. Why the scores are divided by the square root of the key dimension
4. What multi-head attention adds over single-head attention
5. The computational cost with respect to sequence length, and why that matters

Keep your answer under 500 words. You may use equations.

## Repeat 1 (seed 0)

- Score: 9.0/10 — Response correctly defines Q/K/V projections, scaled dot-product steps, scaling rationale, multi-head benefit, and O(n²) cost with clear linear-algebra framing under 500 words.
- Tokens/s: 274.2
- Prefill tok/s: 1946.8
- TTFT: 4.75s
- Gen Time: 7.42s
- Output Tokens: 2004
- answer read from `answer_tag`

<details><summary>Reasoning (5054 chars, not scored)</summary>

Here, the user wants an explanation of the self-attention mechanism in transformers, targeting someone who knows linear algebra but not deep learning. The answer must be under 500 words, cover 5 specific points, and be placed between <answer> and </answer> tags. I need to think first, then produce the final answer.

Let me outline the key points:

1. What queries, keys, and values are, and where they come from.
   - In self-attention, for each token in a sequence, we generate three vectors: query (q), key (k), value (v). These are linear transformations of the input embedding. The input is a sequence of vectors, each representing a token. We have weight matrices W_Q, W_K, W_V that project the input dimension to the representation dimension d_k (or d_model). So for each token i, q_i = x_i W_Q, etc.

2. The scaled dot-product attention computation, step by step.
   - Compute similarity between query and keys: scores = q · k (dot product). Then scale by sqrt(d_k). Then softmax over the scores to get attention weights. Then weighted sum of values: output = sum(softmax(scores) * v).

3. Why the scores are divided by the square root of the key dimension.
   - To prevent large dot products when dimension is high, causing softmax to saturate (gradients vanish). The dot product grows with dimension, so dividing by sqrt(d_k) keeps scores in a reasonable range.

4. What multi-head attention adds over single-head attention.
   - Instead of one set of weight matrices, we have h heads, each with its own Q,K,V projections. Each head attends to different subspaces. The outputs are concatenated and projected again. Allows the model to focus on different parts of the sequence simultaneously.

5. The computational cost with respect to sequence length, and why that matters.
   - Self-attention computes pairwise interactions between all tokens: O(n^2 * d) where n is sequence length, d is dimension. This quadratic cost becomes prohibitive for long sequences. Why matters: limits context length, memory usage, inference speed. Alternatives like linear attention or sparse attention.

Now, I need to write this in a clear, concise manner, under 500 words. I'll use equations where appropriate.

Let me draft:

"In a transformer, self-attention allows each token in a sequence to interact with every other token. Suppose we have an input sequence of tokens, each represented as a vector x_i in R^d_model. For each token, we linearly project it into three vectors: query q_i, key k_i, and value v_i, using weight matrices W_Q, W_K, W_V ∈ R^{d_model × d_k}. Thus q_i = x_i W_Q, and similarly for k_i, v_i. Typically d_k = d_model / h for h heads, but for a single head we can just think of d_k as the representation dimension.

The core idea is to measure how much focus token i should place on token j. We compute the dot product between q_i and k_j, which gives a scalar score s_ij = q_i · k_j = (x_i W_Q) (x_j W_K)^T. This score reflects semantic similarity: large positive dot product means token j is relevant to token i.

To obtain attention weights, we apply softmax across all j for a fixed i: α_ij = exp(s_ij) / Σ_l exp(s_il). This yields a probability distribution over the sequence, indicating how much token i should "attend" to each token j.

The output token representation is then the weighted sum of value vectors: o_i = Σ_j α_ij v_j. This mixes information from all tokens, weighted by relevance.

Why divide by √d_k? The dot product q·k grows roughly with the dimensionality d_k because each component contributes variance. Without scaling, scores become large in magnitude, pushing softmax into its saturated tails where gradients vanish and the distribution becomes uniform (all α_ij ≈ 1/n). Dividing by √d_k keeps scores O(1) regardless of d_k, preserving gradient flow and meaningful attention distributions.

Single-head attention uses one set of projections. Multi-head attention (MHA) runs h independent attention heads in parallel, each with its own (W_Q^h, W_K^h, W_V^h). Each head learns to attend to different types of relationships (e.g., syntactic, semantic, positional). Their outputs are concatenated and projected once more via W_O, yielding a combined representation that captures diverse interactions. This is akin to having multiple "focuses" rather than one.

Computationally, self-attention computes all pairwise dot products, costing O(n^2 d_k) time and memory for sequence length n. This quadratic dependence on n quickly becomes prohibitive: for n=512, n^2 ≈ 262k operations per layer; for n=4096, it's ~16M, straining GPUs. Hence, transformers are limited to moderate sequence lengths, motivating research into linear-attention approximations or sparse attention patterns (e.g., local or global tokens) to reduce cost to O(n d_k) or O(n log n).

That's about 330 words. I'll check and refine.

Let me count: roughly 330 words. Good.

I need to ensure I cover all 5 points. Yes.

Now, format as requested: <answer> and </answer> tags, with the answer inside, no commentary.

I'll produce the final answer.

</details>

**Answer (as scored):**

In a transformer, self‑attention lets every token in a sequence interact with every other token. Suppose the input is a sequence of tokens, each embedded as a vector \(x_i \in \mathbb{R}^{d_{\text{model}}}\). For each token we linearly project it into three vectors—query, key, and value—using weight matrices \(W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_k}\):

\[
q_i = x_i W_Q,\qquad k_i = x_i W_K,\qquad v_i = x_i W_V.
\]

Here \(d_k\) is the dimension of the query/key/value space (often \(d_k = d_{\text{model}}/h\) for \(h\) heads, but we first describe a single head).

**Step‑by‑step dot‑product attention**

1. **Similarity scoring:** For a query \(q_i\) and key \(k_j\) we compute the dot product  
   \[
   s_{ij} = q_i \cdot k_j = (x_i W_Q)(x_j W_K)^{\mathsf{T}}.
   \]
   This scalar measures how well token \(j\) matches token \(i\)'s current focus.

2. **Softmax weighting:** We convert the scores into a probability distribution over all tokens for each query \(i\):
   \[
   \alpha_{ij} = \frac{\exp(s_{ij})}{\sum_{\ell=1}^{n} \exp(s_{i\ell})}.
   \]
   The softmax ensures \(\alpha_{ij} \ge 0\) and \(\sum_j \alpha_{ij} = 1\).

3. **Weighted value aggregation:** The output for token \(i\) is the weighted sum of all value vectors:
   \[
   o_i = \sum_{j=1}^{n} \alpha_{ij} v_j.
   \]
   Thus each token’s new representation mixes information from the whole sequence, weighted by relevance.

**Why divide by \(\sqrt{d_k}\)?**

The dot product \(q_i \cdot k_j\) grows with the dimensionality \(d_k\) because each component contributes variance. Without scaling, scores become large in magnitude, pushing the softmax into its saturated tails where gradients vanish and the attention distribution collapses to uniform (\(\alpha_{ij} \approx 1/n\)). Dividing by \(\sqrt{d_k}\) keeps scores \(\mathcal{O}(1)\) regardless of \(d_k\), preserving meaningful gradients and discriminative attention.

**Multi‑head attention**

Single‑head attention uses one set of projections \((W_Q, W_K, W_V)\). Multi‑head attention (MHA) runs \(h\) such heads in parallel, each with its own \((W_Q^{(h)}, W_K^{(h)}, W_V^{(h)})\). Each head learns to attend to different relationship types—e.g., syntactic, semantic, or positional. Their outputs are concatenated and projected once more via \(W_O\), yielding a combined representation that captures diverse interactions simultaneously. MHA is akin to having multiple “focuses” rather than a single one.

**Computational cost**

Self‑attention computes all pairwise dot products, costing \(\mathcal{O}(n^2 d_k)\) time and memory for a sequence of length \(n\). This quadratic dependence on \(n\) quickly becomes prohibitive: for \(n=512\), \(n^2 \approx 2.6\times10^5\) operations per layer; for \(n=4096\), it exceeds \(1.6\times10^7\), straining GPU memory and limiting inference speed. Hence transformers are naturally restricted to moderate context lengths, motivating research into linear‑attention approximations or sparse attention patterns (local or global tokens) to reduce cost to \(\mathcal{O}(n d_k)\) or \(\mathcal{O}(n \log n)\).

