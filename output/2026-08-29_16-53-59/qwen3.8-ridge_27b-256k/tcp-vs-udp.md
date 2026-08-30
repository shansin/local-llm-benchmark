# knowledge / tcp-vs-udp

**Prompt:** Explain the difference between TCP and UDP. Cover:
1. What guarantees each protocol provides, and what it does not
2. The mechanisms TCP uses to provide its guarantees
3. Why a protocol without those guarantees is sometimes the better choice
4. One concrete real-world application of each, and why that application chose it

Keep your answer under 400 words.

## Repeat 1 (seed 0)

- Score: 9.25/10 — Response is factually accurate, covers all required points including checksums implicitly, stays under 400 words, and is clearly organized.
- Tokens/s: 56.4
- Prefill tok/s: 1035.1
- TTFT: 0.27s
- Gen Time: 29.90s
- Output Tokens: 1672
- answer read from `answer_tag`

<details><summary>Reasoning (4474 chars, not scored)</summary>

We need to answer user: Explain difference TCP/UDP under 400 words. Cover 4 points. Need think first if need, then final between answer tags. Need ensure under 400 words. Let's draft concise.

Need count words. Draft:

TCP is connection-oriented and reliable: it guarantees ordered, in-order delivery, retransmits lost segments, prevents duplicates, and provides flow control and congestion control. It does not guarantee low latency or real-time delivery; it can stall when packets are lost.

UDP is connectionless and unreliable: it does not guarantee delivery, ordering, or duplicate prevention. It does provide a simple, low-overhead datagram service with minimal latency and no connection setup.

TCP achieves guarantees through a connection state, sequence numbers, acknowledgments, retransmission timers, sliding window, checksums, and congestion/flow-control algorithms. If a segment is lost or out of order, TCP retransmits and reorders it before handing data to the application.

UDP is sometimes better when freshness matters more than completeness. For live video, gaming, or voice, an old packet is useless if it arrives late, so dropping it and continuing is preferable to TCP’s retransmission delay. UDP also avoids per-connection overhead and can be more scalable for broadcast/multicast.

Example TCP: web browsing or file transfer (HTTP/HTTPS, FTP). It chose TCP because the application needs a complete, ordered byte stream; missing bytes would corrupt the page or file, and reliability is worth the latency.

Example UDP: online multiplayer game or live video streaming. It chose UDP because the application needs frequent, low-latency updates; a stale update is worse than no update, so it tolerates loss and uses its own recovery or prediction.

Count words? Let's count manually.
TCP(1) is2 connection-oriented3 and4 reliable5 it6 guarantees7 ordered8 in-order9 delivery10 retransmits11 lost12 segments13 prevents14 duplicates15 and16 provides17 flow18 control19 and20 congestion21 control22 It23 does24 not25 guarantee26 low27 latency28 or29 real-time30 delivery31 it32 can33 stall34 when35 packets36 are37 lost38.
UDP39 is40 connectionless41 and42 unreliable43 it44 does45 not46 guarantee47 delivery48 ordering49 or50 duplicate51 prevention52 It53 does54 provide55 a56 simple57 low-overhead58 datagram59 service60 with61 minimal62 latency63 and64 no65 connection66 setup67.
TCP68 achieves69 guarantees70 through71 a72 connection73 state74 sequence75 numbers76 acknowledgments77 retransmission78 timers79 sliding80 window81 checksums82 and83 congestion/flow-control84 algorithms85 If86 a87 segment88 is89 lost90 or91 out92 of93 order94 TCP95 retransmits96 and97 reorders98 it99 before100 handing101 data102 to103 the104 application105.
UDP106 is107 sometimes108 better109 when110 freshness111 matters112 more113 than114 completeness115 For116 live117 video118 gaming119 or120 voice121 an122 old123 packet124 is125 useless126 if127 it128 arrives129 late130 so131 dropping132 it133 and134 continuing135 is136 preferable137 to138 TCP’s139 retransmission140 delay141 UDP142 also143 avoids144 per-connection145 overhead146 and147 can148 be149 more150 scalable151 for152 broadcast/multicast153.
Example154 TCP155 web156 browsing157 or158 file159 transfer160 HTTP/HTTPS161 FTP162 It163 chose164 TCP165 because166 the167 application168 needs169 a170 complete171 ordered172 byte173 stream174 missing175 bytes176 would177 corrupt178 the179 page180 or181 file182 and183 reliability184 is185 worth186 the187 latency188.
Example189 UDP190 online191 multiplayer192 game193 or194 live195 video196 streaming197 It198 chose199 UDP200 because201 the202 application203 needs204 frequent205 low-latency206 updates207 a208 stale209 update210 is211 worse212 than213 no214 update215 so216 it217 tolerates218 loss219 and220 uses221 its222 own223 recovery224 or225 prediction226.

226 words. Good. Need maybe ensure "what guarantees each protocol provides, and what it does not" covered. Yes. Mechanisms TCP uses. Why no guarantees better. One concrete real-world app each and why. Good.

Need final only answer tags. Ensure no commentary. Could include bullets? Under 400 words. Need maybe avoid "in-order" redundancy? Fine.

Need maybe "TCP does not guarantee low latency" but it does guarantee reliable ordered delivery. Good.

Potential issue: "UDP does provide a simple, low-overhead datagram service" not guarantee. Good.

Need final between . Ensure no extra.
</think>

</details>

**Answer (as scored):**

TCP is connection-oriented and reliable: it guarantees ordered delivery, retransmits lost segments, prevents duplicates, and provides flow control and congestion control. It does not guarantee low latency or real-time delivery; it can stall when packets are lost.

UDP is connectionless and unreliable: it does not guarantee delivery, ordering, or duplicate prevention. It does provide a simple, low-overhead datagram service with minimal latency and no connection setup.

TCP achieves its guarantees through connection state, sequence numbers, acknowledgments, retransmission timers, sliding window, checksums, and congestion/flow-control algorithms. If a segment is lost or out of order, TCP retransmits and reorders it before handing data to the application.

UDP is sometimes better when freshness matters more than completeness. For live video, gaming, or voice, an old packet is useless if it arrives late, so dropping it and continuing is preferable to TCP’s retransmission delay. UDP also avoids per-connection overhead and can be more scalable for broadcast/multicast.

Example TCP: web browsing or file transfer (HTTP/HTTPS, FTP). It chose TCP because the application needs a complete, ordered byte stream; missing bytes would corrupt the page or file, and reliability is worth the latency.

Example UDP: online multiplayer game or live video streaming. It chose UDP because the application needs frequent, low-latency updates; a stale update is worse than no update, so it tolerates loss and uses its own recovery or prediction.

