# knowledge / tcp-vs-udp

**Prompt:** Explain the difference between TCP and UDP. Cover:
1. What guarantees each protocol provides, and what it does not
2. The mechanisms TCP uses to provide its guarantees
3. Why a protocol without those guarantees is sometimes the better choice
4. One concrete real-world application of each, and why that application chose it

Keep your answer under 400 words.

## Repeat 1 (seed 0)

- Score: 8.5/10 — Response is accurate and covers all four points with correct mechanisms and examples, but omits UDP checksum and multicast/broadcast advantage, and slightly overstates UDP having no error checking.
- Tokens/s: 293.6
- Prefill tok/s: 1649.4
- TTFT: 2.86s
- Gen Time: 5.14s
- Output Tokens: 1478
- answer read from `answer_tag`

<details><summary>Reasoning (5323 chars, not scored)</summary>

Here, the user is looking for an answer that is under 400 words. I need to explain the difference between TCP and UDP, covering four specific points. I'll structure the answer clearly, ensuring each point is addressed concisely.

First, I'll outline the key differences:

1. Guarantees: TCP provides reliable, in-order delivery, flow control, congestion control, and error checking. UDP provides no guarantees; it's a best-effort delivery with no ordering, no retransmission, and no flow control.

2. Mechanisms TCP uses: TCP uses sequence numbers, acknowledgments (ACKs), retransmissions, sliding window for flow control, and congestion control algorithms (like slow start, congestion avoidance). It establishes a connection via three-way handshake.

3. Why UDP is sometimes better: UDP is faster, has lower latency, and less overhead. It's suitable for real-time applications where losing a few packets is better than delay. Also, it's simpler and works well for broadcast/multicast.

4. Real-world applications: TCP for web browsing (HTTP/HTTPS), email (SMTP), file transfer (FTP) - where data integrity is crucial. UDP for streaming (video/audio like Netflix, Spotify), online gaming, DNS - where speed is more important than perfect reliability.

I need to keep it under 400 words. Let's draft.

I'll start with a brief intro, then address each point in order. I'll make sure to be concise.

Let's count words as I go. Aim for around 300 words.

Draft:

TCP (Transmission Control Protocol) and UDP (User Datagram Protocol) are the two main transport layer protocols. They differ fundamentally in the guarantees they provide and how they achieve them.

1. Guarantees: TCP guarantees reliable, in-order delivery of data. It ensures every byte arrives exactly once, using acknowledgments and retransmissions. It also provides flow control and congestion control to prevent network collapse. UDP provides no such guarantees; it is a best-effort protocol. Packets may be lost, duplicated, or arrive out of order, and there is no mechanism to recover them.

2. TCP mechanisms: TCP establishes a connection via a three-way handshake. It uses sequence numbers to track data, acknowledgments (ACKs) to confirm receipt, and timers to trigger retransmissions of lost segments. Flow control uses a sliding window to match sender speed to receiver capacity. Congestion control dynamically adjusts the sending rate based on network conditions.

3. Why UDP without guarantees is better: The overhead of TCP's guarantees adds latency and bandwidth usage. For applications where timely delivery is more important than perfect accuracy, UDP's simplicity is advantageous. Losing a few packets in a video stream is preferable to the delay caused by retransmission.

4. Real-world applications: TCP is used by web traffic (HTTP/HTTPS), email (SMTP), and file transfer (FTP). These applications require data integrity; a missing word in an email or a corrupted file is unacceptable. UDP is used by streaming media (e.g., Netflix, Spotify), online gaming, and DNS. In streaming, a brief glitch is less noticeable than a buffering delay. Online gaming needs low latency to respond to player actions instantly. DNS queries are small and fast; retransmission would slow down domain resolution.

Word count check: Let's estimate. The draft seems around 250 words. I'll refine to ensure it's under 400 and covers all points clearly.

I'll produce the final answer now.
TCP (Transmission Control Protocol) and UDP (User Datagram Protocol) are the two primary transport layer protocols, differing sharply in reliability and mechanism.

1. **Guarantees provided:** TCP guarantees reliable, in-order delivery of data. It ensures every byte arrives exactly once, using acknowledgments and retransmissions. It also provides flow control and congestion control to prevent network overload. UDP provides no such guarantees; it is a best-effort protocol. Packets may be lost, duplicated, or arrive out of order, and there is no mechanism to recover them.

2. **TCP mechanisms:** TCP establishes a connection via a three-way handshake. It uses sequence numbers to track data, acknowledgments (ACKs) to confirm receipt, and timers to trigger retransmissions of lost segments. Flow control uses a sliding window to match sender speed to receiver capacity. Congestion control dynamically adjusts the sending rate based on network conditions, employing algorithms like slow start and congestion avoidance.

3. **Why UDP’s lack of guarantees is advantageous:** The overhead of TCP’s reliability mechanisms adds latency and bandwidth usage. For applications where timely delivery is more important than perfect accuracy, UDP’s simplicity is superior. Losing a few packets in a real-time stream is preferable to the delay caused by retransmission and buffering.

4. **Real-world applications:** TCP is used by web traffic (HTTP/HTTPS), email (SMTP), and file transfer (FTP). These require data integrity; a missing word in an email or a corrupted file is unacceptable. UDP is used by streaming media (e.g., Netflix, Spotify), online gaming, and DNS. Streaming tolerates brief glitches better than buffering delays. Online gaming needs low latency to respond to player actions instantly. DNS queries are small and fast; retransmission would slow domain resolution.

</details>

**Answer (as scored):**

TCP (Transmission Control Protocol) and UDP (User Datagram Protocol) are the two main transport layer protocols. They differ fundamentally in the guarantees they provide and how they achieve them.

1. Guarantees: TCP guarantees reliable, in-order delivery of data. It ensures every byte arrives exactly once, using acknowledgments and retransmissions. It also provides flow control and congestion control to prevent network collapse. UDP provides no such guarantees; it is a best-effort protocol. Packets may be lost, duplicated, or arrive out of order, and there is no mechanism to recover them.

2. TCP mechanisms: TCP establishes a connection via a three-way handshake. It uses sequence numbers to track data, acknowledgments (ACKs) to confirm receipt, and timers to trigger retransmissions of lost segments. Flow control uses a sliding window to match sender speed to receiver capacity. Congestion control dynamically adjusts the sending rate based on network conditions.

3. Why UDP without guarantees is better: The overhead of TCP's guarantees adds latency and bandwidth usage. For applications where timely delivery is more important than perfect accuracy, UDP's simplicity is advantageous. Losing a few packets in a video stream is preferable to the delay caused by retransmission.

4. Real-world applications: TCP is used by web traffic (HTTP/HTTPS), email (SMTP), and file transfer (FTP). These applications require data integrity; a missing word in an email or a corrupted file is unacceptable. UDP is used by streaming media (e.g., Netflix, Spotify), online gaming, and DNS. In streaming, a brief glitch is less noticeable than a buffering delay. Online gaming needs low latency to respond to player actions instantly. DNS queries are small and fast; retransmission would slow down domain resolution.

