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

- Score: 9.0/10 — Response is factually correct, covers all five required points with proper formulas and scaling rationale, stays under 500 words and matches audience level; minor wording on multi-head projection dimension is acceptable.
- Tokens/s: 208.9
- Prefill tok/s: 2165.6
- TTFT: 0.11s
- Gen Time: 7.99s
- Output Tokens: 1645
- answer read from `answer_tag`

<details><summary>Reasoning (4048 chars, not scored)</summary>

The user wants me to explain self-attention in transformers to someone who understands linear algebra but not deep learning. Let me think through the content carefully.

Let me organize my thoughts on each point:

1. **Queries, keys, and values**: In self-attention, we start with input embeddings (each token becomes a vector). We apply three learned linear projections to get Q (queries), K (keys), and V (values). Each is a matrix where rows correspond to tokens.

2. **Scaled dot-product attention step by step**:
   - Compute scores: QKᵀ (matrix multiplication)
   - Scale by 1/√d_k
   - Softmax over keys (rows) to get attention weights
   - Multiply by V to get output

3. **Why divide by √d_k**: When K and Q are high-dimensional, the dot products have large variance/magnitude. Softmax becomes saturated (gradients vanish). Dividing by √d_k keeps variance stable. If q and k have independent components with mean 0 and variance 1, the dot product has variance d_k, so dividing by √d_k normalizes variance to 1.

4. **Multi-head attention**: Instead of one projection, use h separate projections (heads), each in a lower-dimensional space, run attention in parallel, then concatenate and project again. This allows the model to jointly attend to information from different representation subspaces.

5. **Computational cost**: O(n² · d) roughly for the attention itself, where n is sequence length and d is dimension. The QKᵀ is n×n. This quadratic scaling matters because for long sequences it becomes expensive.

Let me write this concisely under 500 words.

Let me draft:

---

**Setup**: Consider a sequence of n tokens, each represented as a d-dimensional vector. Stack them as rows of a matrix X (n × d). In self-attention, the model learns three weight matrices W_Q, W_K, W_V (each d × d) and forms:
- Q = X W_Q (queries)
- K = X W_K (keys)
- V = X W_V (values)

Each row corresponds to a token. Queries and keys are used to compute similarity; values carry the information we ultimately want to aggregate.

**Scaled dot-product attention**:
1. Scores: S = QKᵀ. This is n × n; entry S_ij is the dot product of query i with key j, measuring how much token i should attend to token j.
2. Scale: divide by √d_k where d_k is the key dimension.
3. Softmax each row: a_ij = exp(S_ij/√d_k) / Σ_k exp(S_ik/√d_k). Each row now sums to 1, giving a probability distribution over keys.
4. Weighted sum: Z = AV, where A is the n×n matrix of a_ij. Output row i is a weighted combination of all value vectors.

**Why scale by √d_k**: If q and k have independent components with zero mean and unit variance, their dot product has mean 0 and variance d_k. Larger d_k → larger-magnitude scores → softmax saturates (outputs near one-hot), which flattens gradients and hinders learning. Dividing by √d_k rescales variance back to ~1, keeping softmax in its sensitive regime.

**Multi-head attention**: Single-head attention mixes all features into one similarity measure. Multi-head instead projects Q, K, V h times into d/h-dimensional subspaces, runs attention in parallel on each head, concatenates the results, and applies a final projection W_O. This lets the model jointly attend to different kinds of relationships (e.g., syntactic vs. referential) across representation subspaces.

**Cost**: The core operation QKᵀ produces an n × n matrix, so attention costs O(n² d) in time and O(n²) in memory. This quadratic dependence on sequence length n is the main bottleneck for long sequences—doubling the sequence quadruples the cost. It motivates techniques that reduce or approximate the n×n matrix.

---

Let me count words. That's roughly 300 words. Good, well under 500. Let me refine and make sure it flows and is accurate.

Let me double check the variance claim. If q_i, k_i are independent with mean 0 variance 1, then q·k = Σ q_i k_i. E[q_i k_i] = 0, Var(q_i k_i) = E[q_i²k_i²] - 0 = E[q_i²]E[k_i²] = 1. Sum of d_k independent terms has variance d_k. So standard deviation √d_k. Yes, correct.

I think this is good. Let me finalize.
</think>

</details>

**Answer (as scored):**

**Setup.** Represent each of the *n* tokens as a *d*-dimensional vector and stack them as the rows of a matrix *X* (*n* × *d*). The model learns three weight matrices *W_Q*, *W_K*, *W_V* (each *d* × *d*) and forms:

- **Q** = *X W_Q* (queries)
- **K** = *X W_K* (keys)
- **V** = *X W_V* (values)

Each row corresponds to one token. Queries and keys are used to measure similarity; values carry the information to be aggregated.

**Scaled dot-product attention.**
1. **Scores:** *S* = *QKᵀ*, an *n* × *n* matrix. Entry *S_ij* = ⟨query_i, key_j⟩ measures how much token *i* should attend to token *j*.
2. **Scale:** divide by √*d_k*, where *d_k* is the key dimension.
3. **Softmax each row:** *a_ij* = exp(*S_ij*/√*d_k*) / Σ_k exp(*S_ik*/√*d_k*). Each row now sums to 1, forming a distribution over keys.
4. **Weighted sum:** *Z* = *AV*, where *A* is the *n* × *n* matrix of *a_ij*. Output row *i* is a weighted combination of all value vectors.

**Why scale by √*d_k*.** If the components of *q* and *k* are independent with zero mean and unit variance, their dot product has mean 0 and variance *d_k*. Larger *d_k* produces larger-magnitude scores, which pushes softmax toward one-hot outputs and flattens its gradients, slowing learning. Dividing by √*d_k* rescales the variance back to ~1, keeping softmax in its sensitive, gradient-rich regime.

**Multi-head attention.** Single-head attention collapses everything into one similarity measure. Multi-head instead projects *Q*, *K*, *V* *h* times into *d/h*-dimensional subspaces, runs attention in parallel on each head, concatenates the results, and applies a final projection *W_O*. This lets the model jointly capture different kinds of relationships (e.g., syntactic vs. referential) across separate representation subspaces.

**Cost.** The core operation *QKᵀ* produces an *n* × *n* matrix, so attention costs O(*n²d*) time and O(*n²*) memory. This quadratic dependence on sequence length is the main bottleneck for long sequences: doubling *n* quadruples the cost. It motivates methods that reduce or approximate the *n* × *n* matrix.

