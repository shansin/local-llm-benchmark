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

- Score: 9.0/10 — Response is factually accurate, covers all five required points with correct formulas and scaling rationale, stays under 500 words and uses linear-algebra framing; minor completeness nuance on multi-head projection description but overall excellent.
- Tokens/s: 80.5
- Prefill tok/s: 948.2
- TTFT: 6.18s
- Gen Time: 17.10s
- Output Tokens: 1354
- answer read from `answer_tag`

<details><summary>Reasoning (1721 chars, not scored)</summary>

The user wants an explanation of self-attention in transformers for someone who knows linear algebra but not deep learning. I need to cover 5 specific points, keep it under 500 words, and use equations where helpful. Let me think about how to structure this clearly.

Key considerations:
- The reader knows linear algebra (matrices, dot products, dimensions) but not deep learning (no prior knowledge of embeddings, layers, training, etc.)
- I need to be precise but accessible
- Under 500 words is tight for 5 topics, so I need to be concise
- Use equations where they clarify

Let me draft this:

1. Q, K, V: They come from the input sequence. Each token is represented as a vector (embedding). Then three learned weight matrices W_Q, W_K, W_V are applied to produce Q, K, V. I should explain the intuition: Q = "what am I looking for?", K = "what do I contain?", V = "what do I actually offer?"

2. Scaled dot-product: score = QK^T, then divide by sqrt(d_k), then softmax, then multiply by V. Step by step.

3. Why sqrt(d_k): variance argument. If q and k are vectors of dimension d_k with components that are roughly unit variance, then q·k is a sum of d_k products, so variance is d_k. Dividing by sqrt(d_k) normalizes the variance back to ~1, keeping softmax in a sensitive region.

4. Multi-head: Instead of one set of W_Q, W_K, W_V, use h different sets. Each head learns a different "subspace" of relationships. Concatenate and project.

5. Computational cost: QK^T is O(n² d) where n is sequence length. Softmax is O(n²). So overall O(n² d) per layer. This quadratic scaling is why long sequences are expensive.

Let me write this concisely, aiming for under 500 words.

Let me count words carefully as I write.

</details>

**Answer (as scored):**

**1. Queries, Keys, Values**

Suppose your input is a sequence of $n$ tokens, each represented as a $d$-dimensional vector (an "embedding"). Three *learned* weight matrices $W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$ are applied to every token vector $x_i$:

$$q_i = W_Q x_i, \quad k_i = W_K x_i, \quad v_i = W_V x_i$$

Stacking all tokens gives matrices $Q, K, V \in \mathbb{R}^{n \times d}$. Intuitively: $q_i$ encodes "what information does position $i$ seek?", $k_j$ encodes "what information does position $j$ offer as a label?", and $v_j$ is the actual content $j$ contributes.

**2. Scaled Dot-Product Attention**

For a single head, the computation is:

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

Step by step:

- **Score matrix:** $S = QK^\top \in \mathbb{R}^{n \times n}$. Entry $S_{ij} = q_i \cdot k_j$ measures how well token $j$'s key matches token $i$'s query.
- **Scale:** Divide every entry by $\sqrt{d_k}$.
- **Softmax (row-wise):** Convert each row of the scaled matrix into a probability distribution over the $n$ positions. Row $i$ now says "how much should position $i$ attend to each position $j$?"
- **Weighted sum:** Multiply the $n \times n$ attention matrix by $V \in \mathbb{R}^{n \times d}$. Output row $i$ is a convex combination of all $v_j$, weighted by the attention probabilities.

**3. Why divide by $\sqrt{d_k}$?**

If the components of $q_i$ and $k_j$ are roughly zero-mean with unit variance, then $q_i \cdot k_j$ is a sum of $d_k$ independent products, giving it variance $\approx d_k$. As $d_k$ grows, the dot-product scores become large in magnitude, pushing the softmax into its saturated (near one-hot) regime where gradients vanish. Dividing by $\sqrt{d_k}$ rescales the variance back to $\sim 1$, keeping the softmax in its sensitive, gradient-friendly region.

**4. Multi-Head Attention**

Instead of one triple $(W_Q, W_K, W_V)$, use $h$ independent triples, each operating on a reduced dimension $d_k = d/h$. Each head computes its own attention pattern, effectively learning to attend to *different relational subspaces* (syntactic, positional, semantic, etc.). The $h$ output matrices are concatenated and passed through a final linear projection $W_O$:

$$\text{MultiHead} = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\, W_O$$

This is strictly more expressive than a single head: one head's softmax is a single probability simplex; $h$ heads can simultaneously track $h$ distinct "views" of the sequence.

**5. Computational Cost**

The dominant term is the $QK^\top$ multiplication: $O(n^2 d)$ flops, plus the $n \times n$ softmax and the final $n \times n$ times $n \times d$ multiply. Total per layer: $O(n^2 d)$.

This **quadratic** dependence on sequence length $n$ is the critical bottleneck. Doubling the sequence quadruples the cost. For $n = 8192$ tokens this is already substantial; for $n = 10^5$ (long documents, genomes) it becomes prohibitive. This is the central motivation for the large body of work on linear-attention approximations, sparse attention, and state-space models that aim to reduce the $n^2$ factor.

