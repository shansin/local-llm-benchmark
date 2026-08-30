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

- Score: 9.0/10 — Response is factually accurate, covers all five required points with correct formulas and scaling rationale, stays under 500 words and matches linear-algebra audience; minor wording nuance on d_k definition but overall excellent.
- Tokens/s: 56.1
- Prefill tok/s: 1173.3
- TTFT: 0.27s
- Gen Time: 50.87s
- Output Tokens: 2840
- answer read from `answer_tag`

<details><summary>Reasoning (4761 chars, not scored)</summary>

We need answer user asks explain self-attention to linear algebra person, under 500 words, cover 5 points, equations okay. Need final between . Need think first. Need ensure under 500 words. Let's draft concise.

Need explain queries keys values where come from: input sequence of vectors x_i in R^d. Linear projections W_Q, W_K, W_V produce q_i, k_i, v_i. Maybe self-attention uses same sequence. Scaled dot product: scores S = Q K^T / sqrt(d_k), softmax, output O = softmax(S) V. Step by step: for each query q_i compute dot products with all keys, normalize, weighted sum of values. Scaling reason: dot product variance grows with d_k, softmax saturation, stable gradients. Multi-head: split d into h heads, each head lower dim, project outputs concat, linear. Adds ability to attend to different subspaces/features/relationships in parallel, richer representation. Computational cost: sequence length n, dimension d. Single head O(n^2 d) for scores and weighted sum? More precisely QK^T n x n cost O(n^2 d_k), softmax O(n^2), AV O(n^2 d_v). Overall O(n^2 d) plus memory O(n^2). Matters for long sequences, quadratic growth, memory bottleneck, motivates efficient attention. Need under 500 words. Let's count maybe 350.

Need final only answer tags. Ensure no commentary. Let's craft.

Potential issue: user says Think first if you need to, then put complete final answer between tags. We can just final. Need maybe no extra outside tags.

Let's write:



Count words? Let's count roughly. Suppose 1: 12? Let's count accurately maybe under 500. Text:
Suppose(1) a2 sentence3 is4 encoded5 as6 n7 vectors8 x_1,...,x_n9 in10 R^d.11 Self-attention12 lets13 each14 vector15 look16 at17 all18 others.19
1. Queries,20 keys,21 and22 values23 are24 learned25 linear26 projections27 of28 the29 same30 input31 vectors:32
q_i33 =34 W_Q35 x_i,36 k_i37 =38 W_K39 x_i,40 v_i41 =42 W_V43 x_i.44
Stacking45 rows46 gives47 Q,K,V48 in49 R^{n50 x51 d}.52 Intuitively,53 q_i54 asks55 “what56 should57 I58 attend59 to?”,60 k_j61 is62 an63 address64 to65 match,66 and67 v_j68 is69 the70 content71 retrieved.72
2. Scaled73 dot-product74 attention75 computes,76 for77 each78 query79 q_i,80 a81 score82 with83 every84 key85 k_j:86
s_{ij}87 =88 q_i^T89 k_j90 /91 sqrt(d_k).92
The93 matrix94 of95 scores96 is97 S98 =99 Q100 K^T101 /102 sqrt(d_k).103 We104 apply105 softmax106 row-wise:107
A_{ij}108 =109 exp(s_{ij})110 /111 sum_l112 exp(s_{il}),113 so114 each115 row116 sums117 to118 1119 and120 gives121 attention122 weights.123 The124 output125 for126 position127 i128 is129 the130 weighted131 sum132 of133 values:134
y_i135 =136 sum_j137 A_{ij}138 v_j,139 or140 O141 =142 A143 V.144
3. Division145 by146 sqrt(d_k)147 stabilizes148 the149 softmax.150 If151 q152 and153 k154 have155 components156 with157 variance158 ~1,159 their160 dot161 product162 has163 variance164 d_k.165 As166 d_k167 grows,168 scores169 become170 large,171 softmax172 becomes173 near174 one-hot,175 and176 gradients177 vanish.178 Scaling179 keeps180 score181 variance182 near183 1.184
4. Multi-head185 attention186 repeats187 this188 h189 times190 in191 parallel.192 The193 dimension194 d195 is196 split197 into198 h199 subspaces200 of201 size202 d_k203 =204 d/h.205 Each206 head207 has208 its209 own210 W_Q,W_K,W_V,211 computes212 attention,213 and214 the215 outputs216 are217 concatenated218 and219 passed220 through221 a222 final223 linear224 map:225
MultiHead(X)226 =227 Concat(head_1,...,head_h)228 W_O.229 This230 lets231 the232 model233 attend234 to235 different236 relationships—syntax,237 position,238 semantics—in239 different240 subspaces241 simultaneously,242 rather243 than244 forcing245 one246 attention247 distribution248 to249 serve250 all251 purposes.252
5. For253 sequence254 length255 n256 and257 dimension258 d,259 the260 dominant261 cost262 is263 forming264 the265 n266 x267 n268 score269 matrix:270 O(n^2271 d)272 time273 and274 O(n^2)275 memory276 for277 a278 single279 head.280 Multi-head281 keeps282 the283 same284 quadratic285 dependence286 on287 n.288 This289 matters290 because291 doubling292 sequence293 length294 quadruples295 attention296 cost297 and298 memory,299 making300 long301 sequences302 expensive303 and304 motivating305 sparse,306 linear,307 or308 streaming309 approximations.310

310 words. Good. Need maybe equations count as words? Under 500 anyway. Need ensure no hidden commentary. Final only tags. Could improve clarity: "where they come from" maybe input embeddings. Good.

Need maybe mention self-attention vs attention: queries keys values from same sequence. Good.

Double-check scaled dot product step by step: yes.

Potential issue: "If q and k have components with variance ~1" maybe not exactly but okay.

Need final. Ensure answer tags. No extra.
</think>

</details>

**Answer (as scored):**

Suppose a sentence is encoded as n vectors x_1,...,x_n in R^d. Self-attention lets each vector look at all others.

1. Queries, keys, and values are learned linear projections of the same input vectors:
q_i = W_Q x_i, k_i = W_K x_i, v_i = W_V x_i.
Stacking rows gives Q,K,V in R^{n x d}. Intuitively, q_i asks “what should I attend to?”, k_j is an address to match, and v_j is the content retrieved.

2. Scaled dot-product attention computes, for each query q_i, a score with every key k_j:
s_{ij} = q_i^T k_j / sqrt(d_k).
The matrix of scores is S = Q K^T / sqrt(d_k). We apply softmax row-wise:
A_{ij} = exp(s_{ij}) / sum_l exp(s_{il}),
so each row sums to 1 and gives attention weights. The output for position i is the weighted sum of values:
y_i = sum_j A_{ij} v_j,
or O = A V.

3. Division by sqrt(d_k) stabilizes the softmax. If q and k have components with variance ~1, their dot product has variance d_k. As d_k grows, scores become large, softmax becomes near one-hot, and gradients vanish. Scaling keeps score variance near 1.

4. Multi-head attention repeats this h times in parallel. The dimension d is split into h subspaces of size d_k = d/h. Each head has its own W_Q,W_K,W_V, computes attention, and the outputs are concatenated and passed through a final linear map:
MultiHead(X) = Concat(head_1,...,head_h) W_O.
This lets the model attend to different relationships—syntax, position, semantics—in different subspaces simultaneously, rather than forcing one attention distribution to serve all purposes.

5. For sequence length n and dimension d, the dominant cost is forming the n x n score matrix: O(n^2 d) time and O(n^2) memory for a single head. Multi-head keeps the same quadratic dependence on n. This matters because doubling sequence length quadruples attention cost and memory, making long sequences expensive and motivating sparse, linear, or streaming approximations.

