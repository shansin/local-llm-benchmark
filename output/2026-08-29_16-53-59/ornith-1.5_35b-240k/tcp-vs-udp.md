# knowledge / tcp-vs-udp

**Prompt:** Explain the difference between TCP and UDP. Cover:
1. What guarantees each protocol provides, and what it does not
2. The mechanisms TCP uses to provide its guarantees
3. Why a protocol without those guarantees is sometimes the better choice
4. One concrete real-world application of each, and why that application chose it

Keep your answer under 400 words.

## Repeat 1 (seed 0)

- Score: 9.0/10 — Response is factually accurate, covers all four required points with correct mechanisms and examples, stays under 400 words, and is clearly organized; minor deduction for not explicitly noting UDP checksum is present but not correcting errors and for slight overstatement about TCP guarantees.
- Tokens/s: 207.7
- Prefill tok/s: 1640.4
- TTFT: 0.11s
- Gen Time: 5.59s
- Output Tokens: 1139
- answer read from `answer_tag`

<details><summary>Reasoning (2634 chars, not scored)</summary>

The user wants me to explain the difference between TCP and UDP, covering four specific points, under 400 words. Let me think through this carefully.

Let me organize my thoughts:

**TCP (Transmission Control Protocol):**
- Connection-oriented, reliable
- Guarantees: delivery, ordering, error detection, no duplication
- Mechanisms: three-way handshake, sequence numbers, acknowledgments, retransmission, flow control (window size), congestion control
- Applications: web browsing (HTTP), email, file transfer

**UDP (User Datagram Protocol):**
- Connectionless, unreliable
- No guarantees of delivery, ordering, or duplication prevention
- Only provides checksums for error detection (but doesn't fix)
- Applications: video streaming, online games, DNS, VoIP

Let me write a concise answer under 400 words covering all four points.

Let me draft:

TCP and UDP are transport-layer protocols. TCP is connection-oriented and reliable; UDP is connectionless and best-effort.

**1. Guarantees**

TCP guarantees:
- Delivery: data arrives (or connection errors out)
- In-order: packets arrive in the order sent
- No duplication: each byte delivered once
- Error detection and correction via retransmission

TCP does NOT guarantee:
- Low latency/timing
- Fixed bandwidth

UDP guarantees:
- Only basic error detection via checksum
- Delivery to the right port/process

UDP does NOT guarantee:
- Delivery, ordering, no duplication, or retransmission

**2. TCP mechanisms**
- Three-way handshake (SYN, SYN-ACK, ACK) to establish connection
- Sequence numbers to order packets and detect gaps
- Acknowledgments (ACKs) - receiver confirms receipt
- Retransmission of unacked packets
- Flow control via sliding window (receiver advertises buffer size)
- Congestion control (slow start, congestion avoidance) to avoid overwhelming network

**3. Why UDP without guarantees can be better**
- Lower overhead, no handshake, no retransmission delays
- Real-time apps: if a packet is lost, retransmitting it is useless because the data is stale (video frame, game state). Better to just move on.
- UDP allows sending to multiple recipients (broadcast/multicast)
- Less latency, more predictable timing

**4. Real-world applications**

TCP example: Web browsing (HTTP). Web pages need all data intact and in order; a missing byte would corrupt the page. Reliability matters more than speed.

UDP example: Online multiplayer game or VoIP/video call. Real-time; latency matters; lost packets are discarded anyway. Also DNS uses UDP for fast lookups.

Let me count words and tighten. I need to keep under 400 words.

Let me write the final version.
</think>

</details>

**Answer (as scored):**

**TCP vs. UDP: An Overview**

Both are transport-layer protocols, but they make fundamentally different trade-offs between reliability and speed.

**1. Guarantees**

*TCP* (Transmission Control Protocol) is connection-oriented and reliable. It guarantees:
- **Delivery** — data arrives or the connection reports an error.
- **In-order** — packets are reassembled in the order sent.
- **No duplication** — each byte is delivered exactly once.
- **Error correction** — corrupted or lost data is retransmitted.

TCP does *not* guarantee low latency, timing, or consistent bandwidth.

*UDP* (User Datagram Protocol) is connectionless and best-effort. It provides only:
- **Error detection** via a checksum (but no correction).
- Delivery to the correct application port.

UDP does *not* guarantee delivery, ordering, no duplication, or retransmission. Lost packets are simply gone.

**2. How TCP delivers its guarantees**

- **Three-way handshake** (SYN, SYN-ACK, ACK) establishes a connection before data flows.
- **Sequence numbers** let the receiver reorder packets and detect gaps.
- **Acknowledgments (ACKs)** confirm receipt; unacked data is **retransmitted**.
- **Flow control** uses a sliding window sized by the receiver's available buffer.
- **Congestion control** (slow start, avoidance) backs off when the network is saturated.

**3. Why dropping guarantees can be better**

UDP's lack of retransmission and handshake means lower overhead and predictable, minimal latency. In real-time scenarios, a late packet is worthless — retransmitting a stale video frame or game update only wastes bandwidth and adds delay. It's better to skip it and continue. UDP also enables broadcast/multicast, which TCP cannot do.

**4. Real-world applications**

*TCP — Web browsing (HTTP):* A web page is made of many interdependent files (HTML, CSS, images). Any missing or out-of-order byte corrupts the page, so reliability and correctness matter far more than speed. The browser waits for lost data rather than displaying a broken page.

*UDP — VoIP / video calls:* Voice and video are continuous and time-sensitive. A dropped packet causes a brief glitch, but retransmitting it seconds later would be useless and would cause judder. Prioritizing low latency over perfect delivery produces a far better user experience. (DNS also uses UDP for fast, stateless lookups.)

