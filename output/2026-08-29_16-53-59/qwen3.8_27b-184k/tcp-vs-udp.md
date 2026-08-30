# knowledge / tcp-vs-udp

**Prompt:** Explain the difference between TCP and UDP. Cover:
1. What guarantees each protocol provides, and what it does not
2. The mechanisms TCP uses to provide its guarantees
3. Why a protocol without those guarantees is sometimes the better choice
4. One concrete real-world application of each, and why that application chose it

Keep your answer under 400 words.

## Repeat 1 (seed 0)

- Score: 9.0/10 — Response is factually accurate, covers all four required points with correct mechanisms and examples, stays under 400 words, and is clearly organized; minor deduction for DNS example being less typical than live media but still valid.
- Tokens/s: 70.5
- Prefill tok/s: 888.0
- TTFT: 4.09s
- Gen Time: 12.36s
- Output Tokens: 855
- answer read from `answer_tag`

<details><summary>Reasoning (1168 chars, not scored)</summary>

The user wants a concise explanation of TCP vs UDP covering 4 specific points, under 400 words. Let me structure this carefully.

1. Guarantees:
- TCP: reliable delivery, in-order, no duplicates, flow control, congestion control
- UDP: no delivery guarantee, no ordering, no flow control, no congestion control. Just "send it and hope."

2. TCP mechanisms:
- Sequence numbers for ordering
- ACKs and retransmission for reliability
- Sliding window for flow control
- Congestion control (slow start, congestion avoidance, etc.)
- Three-way handshake for connection setup
- Checksums for integrity

3. Why UDP is sometimes better:
- Lower latency (no handshake, no retransmission delays)
- Less overhead (8-byte header vs 20+ bytes)
- No head-of-line blocking
- Better for real-time where stale data is useless
- Simpler for one-way broadcasts/multicasts

4. Real-world applications:
- TCP: HTTP/HTTPS (web browsing) - needs every byte, ordering matters, can't lose data
- UDP: DNS queries (or VoIP, or online gaming) - DNS: small request, if it fails you retry, low latency matters, no need for ordering of multiple packets

Let me write this concisely under 400 words.

</details>

**Answer (as scored):**

**1. Guarantees**

TCP is a connection-oriented, reliable protocol. It guarantees: delivery (every byte arrives), ordering (bytes arrive in sequence), and integrity (no silent corruption). It also provides flow control and congestion control.

UDP is connectionless and unreliable. It guarantees essentially nothing beyond a basic checksum. Packets may be lost, arrive out of order, or be duplicated. There is no flow control or congestion control.

**2. TCP's mechanisms**

- **Three-way handshake** (SYN → SYN-ACK → ACK) establishes a connection before data flows.
- **Sequence numbers** on every segment let the receiver reorder packets and detect gaps.
- **Acknowledgments (ACKs) and retransmission timers**: if an ACK doesn't arrive, the sender retransmits.
- **Sliding window**: the receiver advertises how much buffer space it has, preventing a fast sender from overwhelming a slow receiver (flow control).
- **Congestion control** (slow start, congestion avoidance, fast retransmit/recovery): the sender dynamically adjusts its rate based on network feedback, protecting the shared medium.
- **Checksums** on every segment catch bit-level corruption.

**3. Why "no guarantees" can be the better choice**

TCP's reliability comes at a cost: handshake latency, retransmission delays, head-of-line blocking (one lost packet stalls all later ones), and per-packet bookkeeping. For real-time media, a 200 ms-old audio frame is worthless even if it arrives perfectly. UDP's "fire and forget" model eliminates retransmission stalls, keeps headers to 8 bytes, and lets the application decide what to do with loss (e.g., skip a frame). It also scales naturally to multicast/broadcast, where per-connection ACKs are impractical.

**4. Concrete applications**

- **TCP → HTTPS (web browsing).** A web page is a dependency chain: HTML references CSS, which references images, which reference fonts. A single lost byte corrupts a JSON API response or breaks a TLS handshake. Ordering and completeness are non-negotiable, so TCP's guarantees are essential.

- **UDP → DNS resolution.** A DNS query is a single small packet (often < 100 bytes). If it's lost, the resolver simply retries after a short timeout—retransmitting the whole tiny packet is cheaper than TCP's per-segment ACK machinery. There is no ordering to preserve, no multi-packet stream, and the 3-way handshake would add an extra round-trip to every lookup. UDP's minimal overhead and zero connection setup make it the natural fit.

