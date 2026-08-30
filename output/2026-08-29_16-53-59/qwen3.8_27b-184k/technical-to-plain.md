# writing / technical-to-plain

**Prompt:** Rewrite the following notice so that a general audience can understand and act on it.

Original:
"Pursuant to the scheduled infrastructure maintenance window commencing 02:00 UTC on the 14th instant, authentication services will be subject to intermittent unavailability of an anticipated duration not exceeding 90 minutes. Users holding active sessions may experience involuntary session termination and are advised to effect persistence of any in-progress work prior to the aforementioned commencement time. Escalation of service-impacting incidents should be directed through established channels."

Requirements:
- 90-150 words
- Lead with what the reader must do, not with the schedule
- Convert UTC to a form that helps a reader in any timezone
- Keep every fact from the original: no invented details, no dropped ones
- Plain language, but not patronising — do not add "don't worry!" reassurance the original does not support
- End with what to do if something breaks

## Repeat 1 (seed 0)

- Score: 8.75/10 — Response keeps all facts, leads with save-work action and gives UTC with local conversion cue, ends with escalation, but adds interpretive condition about disruption beyond 90 minutes not in original.
- Tokens/s: 108.5
- Prefill tok/s: 1185.4
- TTFT: 100.68s
- Gen Time: 101.68s
- Output Tokens: 10996
- answer read from `answer_tag`

<details><summary>Reasoning (33185 chars, not scored)</summary>

Let me analyze the original notice and extract all the facts:

1. Scheduled infrastructure maintenance
2. Commencing 02:00 UTC on the 14th instant (the 14th of the current month)
3. Authentication services will be intermittently unavailable
4. Duration: not exceeding 90 minutes
5. Users with active sessions may experience involuntary session termination
6. Users are advised to save/persist any in-progress work before the commencement time
7. Escalation of service-impacting incidents should be directed through established channels (i.e., report through normal support channels)

Requirements:
- 90-150 words
- Lead with what the reader must do (save your work), not with the schedule
- Convert UTC to a form that helps a reader in any timezone (I should mention it's 02:00 UTC and note that readers should convert to their local time, or say "2:00 AM UTC (check your local time)")
- Keep every fact: no invented details, no dropped ones
- Plain language, not patronizing, no "don't worry!"
- End with what to do if something breaks

Let me draft:

"Save any work in progress before 02:00 UTC on the 14th of this month (check your local time zone). During a scheduled maintenance window starting at that time, sign-in and authentication services will be intermittently unavailable for up to 90 minutes. If you are logged in, your session may be disconnected without warning. To avoid losing work, make sure everything is saved before the maintenance begins. If the outage causes a problem that goes beyond the expected 90-minute window, report it through your usual support channels."

Let me count words: Save(1) any(2) work(3) in(4) progress(5) before(6) 02:00(7) UTC(8) on(9) the(10) 14th(11) of(12) this(13) month(14) (check(15) your(16) local(17) time(18) zone)(19). During(20) a(21) scheduled(22) maintenance(23) window(24) starting(25) at(26) that(27) time(28), sign-in(29) and(30) authentication(31) services(32) will(33) be(34) intermittently(35) unavailable(36) for(37) up(38) to(39) 90(40) minutes(41). If(42) you(43) are(44) logged(45) in(46), your(47) session(48) may(49) be(50) disconnected(51) without(52) warning(53). To(54) avoid(55) losing(56) work(57), make(58) sure(59) everything(60) is(61) saved(62) before(63) the(64) maintenance(65) begins(66). If(67) the(68) outage(69) causes(70) a(71) problem(72) that(73) goes(74) beyond(75) the(76) expected(77) 90-minute(78) window(79), report(80) it(81) through(82) your(83) usual(84) support(85) channels(86).

That's about 86 words. I need 90-150. Let me expand slightly.

Let me also check: "the 14th instant" means the 14th of the current month. I should say "the 14th of this month" or just "the 14th."

Let me also make sure I'm not adding "without warning" — the original says "involuntary session termination" which means it happens to you, not that you choose it. "Without warning" might be an addition. Let me use "disconnected" or "ended" instead. Actually "involuntary" means it happens to you against your will — "your session may be ended" captures that.

Let me also check: "authentication services" — I should keep this specific. "Sign-in and authentication services" adds "sign-in" which isn't in the original. Let me just say "authentication services."

Let me also check "established channels" — this means the normal/standard support channels. "Your usual support channels" works.

Let me also check: "service-impacting incidents" — incidents that impact the service. So if something goes wrong beyond the expected maintenance.

Let me redraft:

"Save any work in progress before 02:00 UTC on the 14th of this month — convert that to your local time zone. A scheduled infrastructure maintenance window begins at that time, during which authentication services will be intermittently unavailable for up to 90 minutes. If you have an active session, it may be ended without your action. Make sure all in-progress work is saved before the maintenance starts. If the disruption extends beyond the expected 90 minutes or causes a problem beyond the planned maintenance, report it through your standard support channels."

Word count: Save(1) any(2) work(3) in(4) progress(5) before(6) 02:00(7) UTC(8) on(9) the(10) 14th(11) of(12) this(13) month(14) —(not a word) convert(15) that(16) to(17) your(18) local(19) time(20) zone(21). A(22) scheduled(23) infrastructure(24) maintenance(25) window(26) begins(27) at(28) that(29) time(30), during(31) which(32) authentication(33) services(34) will(35) be(36) intermittently(37) unavailable(38) for(39) up(40) to(41) 90(42) minutes(43). If(44) you(45) have(46) an(47) active(48) session(49), it(50) may(51) be(52) ended(53) without(54) your(55) action(56). Make(57) sure(58) all(59) in-progress(60) work(61) is(62) saved(63) before(64) the(65) maintenance(66) starts(67). If(68) the(69) disruption(70) extends(71) beyond(72) the(73) expected(74) 90(75) minutes(76) or(77) causes(78) a(79) problem(80) beyond(81) the(82) planned(83) maintenance(84), report(85) it(86) through(87) your(88) standard(89) support(90) channels(91).

That's about 91 words. Good, within range.

Wait, let me re-examine. "without your action" — the original says "involuntary session termination." That means the session ends involuntarily, i.e., not by the user's choice. "Without your action" captures that. Good.

Actually, let me reconsider "the 14th of this month" — the original says "the 14th instant" which is a formal way of saying "the 14th of the current month." So "the 14th of this month" is accurate.

Let me also verify I'm not adding anything. The original mentions:
- scheduled infrastructure maintenance window ✓
- commencing 02:00 UTC on the 14th instant ✓
- authentication services ✓
- intermittent unavailability ✓
- anticipated duration not exceeding 90 minutes ✓ (up to 90 minutes)
- Users holding active sessions ✓
- involuntary session termination ✓
- advised to effect persistence of any in-progress work ✓ (save work)
- prior to the aforementioned commencement time ✓ (before maintenance starts)
- Escalation of service-impacting incidents ✓ (if disruption extends beyond...)
- directed through established channels ✓ (standard support channels)

I think I've got everything. Let me refine the language a bit more for flow and clarity.

"Save any work in progress before 02:00 UTC on the 14th of this month — convert that to your local time zone. A scheduled infrastructure maintenance window begins at that time, during which authentication services will be intermittently unavailable for up to 90 minutes. If you have an active session, it may be ended without your action. Make sure all in-progress work is saved before the maintenance starts. If the disruption extends beyond the expected 90 minutes or causes a problem beyond the planned maintenance, report it through your standard support channels."

Hmm, I'm repeating "save work" twice. Let me restructure. The requirement says "Lead with what the reader must do." So the first sentence should be the action. Then I can explain why.

"Before 02:00 UTC on the 14th of this month (convert to your local time zone), save all in-progress work. A scheduled infrastructure maintenance window begins at that time, during which authentication services will be intermittently unavailable for up to 90 minutes. If you have an active session, it may be ended without your action. If the disruption extends beyond the expected 90 minutes or causes a problem beyond the planned maintenance, report it through your standard support channels."

Word count: Before(1) 02:00(2) UTC(3) on(4) the(5) 14th(6) of(7) this(8) month(9) (convert(10) to(11) your(12) local(13) time(14) zone)(15), save(16) all(17) in-progress(18) work(19). A(20) scheduled(21) infrastructure(22) maintenance(23) window(24) begins(25) at(26) that(27) time(28), during(29) which(30) authentication(31) services(32) will(33) be(34) intermittently(35) unavailable(36) for(37) up(38) to(39) 90(40) minutes(41). If(42) you(43) have(44) an(45) active(46) session(47), it(48) may(49) be(50) ended(51) without(52) your(53) action(54). If(55) the(56) disruption(57) extends(58) beyond(59) the(60) expected(61) 90(62) minutes(63) or(64) causes(65) a(66) problem(67) beyond(68) the(69) planned(70) maintenance(71), report(72) it(73) through(74) your(75) standard(76) support(77) channels(78).

That's about 78 words. Too short. I need 90-150.

Let me expand. I can add a bit more context while staying factual.

"Before 02:00 UTC on the 14th of this month — convert that to your local time zone — save all in-progress work. A scheduled infrastructure maintenance window begins at that time. During the window, authentication services will be intermittently unavailable for a period not exceeding 90 minutes. If you have an active session, it may be ended without your action, so make sure everything is saved before the maintenance starts. If the disruption extends beyond the expected 90 minutes or causes a problem beyond the planned maintenance, report it through your standard support channels."

Word count: Before(1) 02:00(2) UTC(3) on(4) the(5) 14th(6) of(7) this(8) month(9) —(skip) convert(10) that(11) to(12) your(13) local(14) time(15) zone(16) —(skip) save(17) all(18) in-progress(19) work(20). A(21) scheduled(22) infrastructure(23) maintenance(24) window(25) begins(26) at(27) that(28) time(29). During(30) the(31) window(32), authentication(33) services(34) will(35) be(36) intermittently(37) unavailable(38) for(39) a(40) period(41) not(42) exceeding(43) 90(44) minutes(45). If(46) you(47) have(48) an(49) active(50) session(51), it(52) may(53) be(54) ended(55) without(56) your(57) action(58), so(59) make(60) sure(61) everything(62) is(63) saved(64) before(65) the(66) maintenance(67) starts(68). If(69) the(70) disruption(71) extends(72) beyond(73) the(74) expected(75) 90(76) minutes(77) or(78) causes(79) a(80) problem(81) beyond(82) the(83) planned(84) maintenance(85), report(86) it(87) through(88) your(89) standard(90) support(91) channels(92).

About 92 words. That's within range (90-150). Good.

Wait, I'm repeating "save work" in the first sentence and then again "make sure everything is saved before the maintenance starts." That's redundant. Let me fix.

"Before 02:00 UTC on the 14th of this month — convert that to your local time zone — save all in-progress work. A scheduled infrastructure maintenance window begins at that time. During the window, authentication services will be intermittently unavailable for a period not exceeding 90 minutes. If you have an active session, it may be ended without your action. If the disruption extends beyond the expected 90 minutes or causes a problem beyond the planned maintenance, report it through your standard support channels."

Word count: Before(1) 02:00(2) UTC(3) on(4) the(5) 14th(6) of(7) this(8) month(9) convert(10) that(11) to(12) your(13) local(14) time(15) zone(16) save(17) all(18) in-progress(19) work(20). A(21) scheduled(22) infrastructure(23) maintenance(24) window(25) begins(26) at(27) that(28) time(29). During(30) the(31) window(32) authentication(33) services(34) will(35) be(36) intermittently(37) unavailable(38) for(39) a(40) period(41) not(42) exceeding(43) 90(44) minutes(45). If(46) you(47) have(48) an(49) active(50) session(51) it(52) may(53) be(54) ended(55) without(56) your(57) action(58). If(59) the(60) disruption(61) extends(62) beyond(63) the(64) expected(65) 90(66) minutes(67) or(68) causes(69) a(70) problem(71) beyond(72) the(73) planned(74) maintenance(75) report(76) it(77) through(78) your(79) standard(80) support(81) channels(82).

About 82 words. Still a bit short. Let me add a little more.

Actually, let me recount more carefully. Hyphenated words like "in-progress" — is that one word or two? Typically in word counts, hyphenated words count as one. Let me be more generous in my expansion.

Let me try again:

"Before 02:00 UTC on the 14th of this month — convert that to your local time zone — save all in-progress work. A scheduled infrastructure maintenance window begins at that time. During the window, authentication services will be intermittently unavailable for a period not exceeding 90 minutes. If you have an active session, it may be ended without your action, so be sure your work is saved before the maintenance begins. If the disruption extends beyond the expected 90 minutes or causes a problem beyond the planned maintenance, report it through your standard support channels."

Let me count again carefully:
Before(1) 02:00(2) UTC(3) on(4) the(5) 14th(6) of(7) this(8) month(9) convert(10) that(11) to(12) your(13) local(14) time(15) zone(16) save(17) all(18) in-progress(19) work(20) A(21) scheduled(22) infrastructure(23) maintenance(24) window(25) begins(26) at(27) that(28) time(29) During(30) the(31) window(32) authentication(33) services(34) will(35) be(36) intermittently(37) unavailable(38) for(39) a(40) period(41) not(42) exceeding(43) 90(44) minutes(45) If(46) you(47) have(48) an(49) active(50) session(51) it(52) may(53) be(54) ended(55) without(56) your(57) action(58) so(59) be(60) sure(61) your(62) work(63) is(64) saved(65) before(66) the(67) maintenance(68) begins(69) If(70) the(71) disruption(72) extends(73) beyond(74) the(75) expected(76) 90(77) minutes(78) or(79) causes(80) a(81) problem(82) beyond(83) the(84) planned(85) maintenance(86) report(87) it(88) through(89) your(90) standard(91) support(92) channels(93)

93 words. That's within 90-150. 

But wait — I'm saying "save all in-progress work" at the start AND "be sure your work is saved before the maintenance begins" later. That's redundant. Let me remove the second one since the first one already covers it.

"Before 02:00 UTC on the 14th of this month — convert that to your local time zone — save all in-progress work. A scheduled infrastructure maintenance window begins at that time. During the window, authentication services will be intermittently unavailable for a period not exceeding 90 minutes. If you have an active session, it may be ended without your action. If the disruption extends beyond the expected 90 minutes or causes a problem beyond the planned maintenance, report it through your standard support channels."

Count: Before(1) 02:00(2) UTC(3) on(4) the(5) 14th(6) of(7) this(8) month(9) convert(10) that(11) to(12) your(13) local(14) time(15) zone(16) save(17) all(18) in-progress(19) work(20) A(21) scheduled(22) infrastructure(23) maintenance(24) window(25) begins(26) at(27) that(28) time(29) During(30) the(31) window(32) authentication(33) services(34) will(35) be(36) intermittently(37) unavailable(38) for(39) a(40) period(41) not(42) exceeding(43) 90(44) minutes(45) If(46) you(47) have(48) an(49) active(50) session(51) it(52) may(53) be(54) ended(55) without(56) your(57) action(58) If(59) the(60) disruption(61) extends(62) beyond(63) the(64) expected(65) 90(66) minutes(67) or(68) causes(69) a(70) problem(71) beyond(72) the(73) planned(74) maintenance(75) report(76) it(77) through(78) your(79) standard(80) support(81) channels(82)

82 words. Below 90. I need to add a bit more.

Let me think about what I can add without inventing details. I could:
- Say "your sign-in may fail" as part of authentication services being unavailable
- Be slightly more explicit about the session termination
- Add a small connector

Actually, "authentication services will be intermittently unavailable" — I could say "you may be unable to sign in or authenticate" but that adds "sign in" which isn't explicitly in the original. The original just says "authentication services." Let me keep it as is.

Let me try expanding slightly:

"Before 02:00 UTC on the 14th of this month — convert that to your local time zone — save all in-progress work. A scheduled infrastructure maintenance window begins at that time. During the window, authentication services will be intermittently unavailable for a period not exceeding 90 minutes. If you have an active session, it may be ended without your action, and you will need to sign back in once services are restored. If the disruption extends beyond the expected 90 minutes or causes a problem beyond the planned maintenance, report it through your standard support channels."

Hmm, "you will need to sign back in once services are restored" — is that in the original? Not explicitly. The original says authentication services will be unavailable, which implies you can't authenticate, but it doesn't explicitly say "sign back in." I should be careful not to add details. Let me remove that.

Let me try a different approach to get to 90+:

"Before 02:00 UTC on the 14th of this month — convert that to your local time zone — save all in-progress work. A scheduled infrastructure maintenance window begins at that time, and during the window, authentication services will be intermittently unavailable for a period not exceeding 90 minutes. If you have an active session, it may be ended without your action. If the disruption extends beyond the expected 90 minutes, or if it causes a problem beyond the planned maintenance, report it through your standard support channels."

Count: Before(1) 02:00(2) UTC(3) on(4) the(5) 14th(6) of(7) this(8) month(9) convert(10) that(11) to(12) your(13) local(14) time(15) zone(16) save(17) all(18) in-progress(19) work(20) A(21) scheduled(22) infrastructure(23) maintenance(24) window(25) begins(26) at(27) that(28) time(29) and(30) during(31) the(32) window(33) authentication(34) services(35) will(36) be(37) intermittently(38) unavailable(39) for(40) a(41) period(42) not(43) exceeding(44) 90(45) minutes(46) If(47) you(48) have(49) an(50) active(51) session(52) it(53) may(54) be(55) ended(56) without(57) your(58) action(59) If(60) the(61) disruption(62) extends(63) beyond(64) the(65) expected(66) 90(67) minutes(68) or(69) if(70) it(71) causes(72) a(73) problem(74) beyond(75) the(76) planned(77) maintenance(78) report(79) it(80) through(81) your(82) standard(83) support(84) channels(85)

85 words. Still a bit short. Let me add a few more words naturally.

"Before 02:00 UTC on the 14th of this month — convert that to your local time zone — save all in-progress work. A scheduled infrastructure maintenance window begins at that time. During the window, authentication services will be intermittently unavailable for a period not exceeding 90 minutes. If you have an active session, it may be ended without your action. If the disruption extends beyond the expected 90 minutes, or if it causes a problem beyond the planned maintenance, report the issue through your standard support channels."

Count: Before(1) 02:00(2) UTC(3) on(4) the(5) 14th(6) of(7) this(8) month(9) convert(10) that(11) to(12) your(13) local(14) time(15) zone(16) save(17) all(18) in-progress(19) work(20) A(21) scheduled(22) infrastructure(23) maintenance(24) window(25) begins(26) at(27) that(28) time(29) During(30) the(31) window(32) authentication(33) services(34) will(35) be(36) intermittently(37) unavailable(38) for(39) a(40) period(41) not(42) exceeding(43) 90(44) minutes(45) If(46) you(47) have(48) an(49) active(50) session(51) it(52) may(53) be(54) ended(55) without(56) your(57) action(58) If(59) the(60) disruption(61) extends(62) beyond(63) the(64) expected(65) 90(66) minutes(67) or(68) if(69) it(70) causes(71) a(72) problem(73) beyond(74) the(75) planned(76) maintenance(77) report(78) the(79) issue(80) through(81) your(82) standard(83) support(84) channels(85)

85 words. Still short. Let me think about what else I can naturally add.

I could say "Please save all in-progress work" — adding "Please" makes it 86.

Or I could expand the timezone note: "convert that to your local time zone before the 14th" — no, that changes meaning.

What about: "Before 02:00 UTC on the 14th of this month (convert to your local time zone), save all in-progress work."

Or I could be slightly more descriptive: "A scheduled infrastructure maintenance window begins at that time, during which authentication services will be intermittently unavailable for a period not exceeding 90 minutes."

Let me try:

"Before 02:00 UTC on the 14th of this month — convert that to your local time zone — save all in-progress work. A scheduled infrastructure maintenance window begins at that time, during which authentication services will be intermittently unavailable for a period not exceeding 90 minutes. If you have an active session, it may be ended without your action. If the disruption extends beyond the expected 90 minutes, or if it causes a problem beyond the planned maintenance, report the issue through your standard support channels."

Count: Before(1) 02:00(2) UTC(3) on(4) the(5) 14th(6) of(7) this(8) month(9) convert(10) that(11) to(12) your(13) local(14) time(15) zone(16) save(17) all(18) in-progress(19) work(20) A(21) scheduled(22) infrastructure(23) maintenance(24) window(25) begins(26) at(27) that(28) time(29) during(30) which(31) authentication(32) services(33) will(34) be(35) intermittently(36) unavailable(37) for(38) a(39) period(40) not(41) exceeding(42) 90(43) minutes(44) If(45) you(46) have(47) an(48) active(49) session(50) it(51) may(52) be(53) ended(54) without(55) your(56) action(57) If(58) the(59) disruption(60) extends(61) beyond(62) the(63) expected(64) 90(65) minutes(66) or(67) if(68) it(69) causes(70) a(71) problem(72) beyond(73) the(74) planned(75) maintenance(76) report(77) the(78) issue(79) through(80) your(81) standard(82) support(83) channels(84)

84 words. Still under 90. I need to add about 6-10 more words.

Options:
- "Please save all in-progress work" (+1)
- "If you have an active session at the time, it may be ended without your action" (+2)
- "report the issue through your standard or established support channels" (+1)
- "for a period not exceeding 90 minutes" → "for a duration not exceeding 90 minutes" (same)
- Add "Note:" at the start? No, that's unnecessary.
- "If the disruption extends beyond the expected 90-minute window" (same count roughly)

Let me try:

"Before 02:00 UTC on the 14th of this month — convert that to your local time zone — please save all in-progress work. A scheduled infrastructure maintenance window begins at that time, during which authentication services will be intermittently unavailable for a period not exceeding 90 minutes. If you have an active session at the time, it may be ended without your action. If the disruption extends beyond the expected 90 minutes, or if it causes a problem beyond the planned maintenance, report the issue through your standard support channels."

Count: Before(1) 02:00(2) UTC(3) on(4) the(5) 14th(6) of(7) this(8) month(9) convert(10) that(11) to(12) your(13) local(14) time(15) zone(16) please(17) save(18) all(19) in-progress(20) work(21) A(22) scheduled(23) infrastructure(24) maintenance(25) window(26) begins(27) at(28) that(29) time(30) during(31) which(32) authentication(33) services(34) will(35) be(36) intermittently(37) unavailable(38) for(39) a(40) period(41) not(42) exceeding(43) 90(44) minutes(45) If(46) you(47) have(48) an(49) active(50) session(51) at(52) the(53) time(54) it(55) may(56) be(57) ended(58) without(59) your(60) action(61) If(62) the(63) disruption(64) extends(65) beyond(66) the(67) expected(68) 90(69) minutes(70) or(71) if(72) it(73) causes(74) a(75) problem(76) beyond(77) the(78) planned(79) maintenance(80) report(81) the(82) issue(83) through(84) your(85) standard(86) support(87) channels(88)

88 words. So close! Let me add a couple more.

"Before 02:00 UTC on the 14th of this month — convert that to your local time zone — please save all in-progress work. A scheduled infrastructure maintenance window begins at that time, during which authentication services will be intermittently unavailable for a period not exceeding 90 minutes. If you have an active session at the time, it may be ended without your action. If the disruption extends beyond the expected 90 minutes, or if it causes a problem beyond the planned maintenance, report the issue through your standard or established support channels."

Adding "or established" = +2 → 90 words. 

Actually wait, "standard or established" is redundant. Let me think of another way.

"Before 02:00 UTC on the 14th of this month — convert that to your local time zone — please save all in-progress work. A scheduled infrastructure maintenance window begins at that time, during which authentication services will be intermittently unavailable for a period not exceeding 90 minutes. If you have an active session at the time, it may be ended without your action. If the disruption extends beyond the expected 90 minutes, or if it causes a problem beyond the planned maintenance, report the issue through your established support channels."

"established" instead of "standard" — same count. Still 88.

Let me try: "report the issue through your usual support channels" — same.

What if I say "If the disruption extends beyond the expected 90-minute window" — "90-minute" is one hyphenated word vs "90 minutes" two words. That would reduce count. Bad.

What about: "If the disruption extends beyond the expected 90 minutes, or if it causes a service-impacting problem beyond the planned maintenance, report the issue through your standard support channels."

"service-impacting" is one word (hyphenated). Let me count:
...If(62) the(63) disruption(64) extends(65) beyond(66) the(67) expected(68) 90(69) minutes(70) or(71) if(72) it(73) causes(74) a(75) service-impacting(76) problem(77) beyond(78) the(79) planned(80) maintenance(81) report(82) the(83) issue(84) through(85) your(86) standard(87) support(88) channels(89)

89. Still one short!

Let me try: "report the issue through your standard support channels" → "report the issue through your standard support or help-desk channels" — no, that adds "help-desk" which isn't in the original.

What about just making a small natural expansion elsewhere:

"Before 02:00 UTC on the 14th of this month — convert that to your local time zone — please save all in-progress work. A scheduled infrastructure maintenance window begins at that time, during which authentication services will be intermittently unavailable for a period not exceeding 90 minutes. If you have an active session at the time, it may be ended without your action. If the disruption extends beyond the expected 90 minutes, or if it causes a problem beyond the planned maintenance, please report the issue through your standard support channels."

Adding "please" before "report" = +1 → 89. Still one short.

Hmm. Let me try a different approach. What if I say "for up to 90 minutes" instead of "for a period not exceeding 90 minutes"? That's shorter. Bad.

What about: "A scheduled infrastructure maintenance window begins at that time. During the window, authentication services will be intermittently unavailable for a period not exceeding 90 minutes."

That splits into two sentences but same word count.

OK let me try yet another approach. What if I add "Note:" at the beginning? No, that's unnecessary.

What about: "Before 02:00 UTC on the 14th of this month (convert to your local time zone), please save all in-progress work."

Count: Before(1) 02:00(2) UTC(3) on(4) the(5) 14th(6) of(7) this(8) month(9) convert(10) to(11) your(12) local(13) time(14) zone(15) please(16) save(17) all(18) in-progress(19) work(20)

vs. the em-dash version:
Before(1) 02:00(2) UTC(3) on(4) the(5) 14th(6) of(7) this(8) month(9) convert(10) that(11) to(12) your(13) local(14) time(15) zone(16) please(17) save(18) all(19) in-progress(20) work(21)

The em-dash version has "that" extra. So the parenthetical version is one word shorter.

Let me just go with a slightly different phrasing that naturally hits 90+:

"Before 02:00 UTC on the 14th of this month — convert that to your local time zone — please save all in-progress work. A scheduled infrastructure maintenance window begins at that time, during which authentication services will be intermittently unavailable for a period not exceeding 90 minutes. If you have an active session at the time, it may be ended without your action. If the disruption extends beyond the expected 90 minutes, or if it causes a problem beyond the planned maintenance, report the issue through your standard support channels."

Let me very carefully count every word:
1. Before
2. 02:00
3. UTC
4. on
5. the
6. 14th
7. of
8. this
9. month
10. convert
11. that
12. to
13. your
14. local
15. time
16. zone
17. please
18. save
19. all
20. in-progress
21. work
22. A
23. scheduled
24. infrastructure
25. maintenance
26. window
27. begins
28. at
29. that
30. time
31. during
32. which
33. authentication
34. services
35. will
36. be
37. intermittently
38. unavailable
39. for
40. a
41. period
42. not
43. exceeding
44. 90
45. minutes
46. If
47. you
48. have
49. an
50. active
51. session
52. at
53. the
54. time
55. it
56. may
57. be
58. ended
59. without
60. your
61. action
62. If
63. the
64. disruption
65. extends
66. beyond
67. the
68. expected
69. 90
70. minutes
71. or
72. if
73. it
74. causes
75. a
76. problem
77. beyond
78. the
79. planned
80. maintenance
81. report
82. the
83. issue
84. through
85. your
86. standard
87. support
88. channels

88 words. I need 2 more.

What if I change "report the issue" to "report the issue to the appropriate team"? No, that adds "to the appropriate team" which isn't in the original.

What about "report the issue through your standard support channels" → "report the issue through your established support channels"? Same count.

What if I say "If the disruption extends beyond the expected 90 minutes, or if it causes a problem that goes beyond the planned maintenance..." — "that goes" instead of just "beyond" adds 1 word.

Let me try: "If the disruption extends beyond the expected 90 minutes, or if it causes a problem that goes beyond the planned maintenance, report the issue through your standard support channels."

That adds "that goes" = +2 → 90 words!

Let me recount:
1. Before
2. 02:00
3. UTC
4. on
5. the
6. 14th
7. of
8. this
9. month
10. convert
11. that
12. to
13. your
14. local
15. time
16. zone
17. please
18. save
19. all
20. in-progress
21. work
22. A
23. scheduled
24. infrastructure
25. maintenance
26. window
27. begins
28. at
29. that
30. time
31. during
32. which
33. authentication
34. services
35. will
36. be
37. intermittently
38. unavailable
39. for
40. a
41. period
42. not
43. exceeding
44. 90
45. minutes
46. If
47. you
48. have
49. an
50. active
51. session
52. at
53. the
54. time
55. it
56. may
57. be
58. ended
59. without
60. your
61. action
62. If
63. the
64. disruption
65. extends
66. beyond
67. the
68. expected
69. 90
70. minutes
71. or
72. if
73. it
74. causes
75. a
76. problem
77. that
78. goes
79. beyond
80. the
81. planned
82. maintenance
83. report
84. the
85. issue
86. through
87. your
88. standard
89. support
90. channels

90 words! Perfect.

Now let me check all requirements:
- 90-150 words: ✓ (90)
- Lead with what the reader must do: ✓ (save work is the first instruction)
- Convert UTC to a form that helps a reader in any timezone: ✓ (says "convert that to your local time zone")
- Keep every fact: Let me verify:
  - scheduled infrastructure maintenance window ✓
  - 02:00 UTC on the 14th ✓
  - authentication services ✓
  - intermittently unavailable ✓
  - not exceeding 90 minutes ✓
  - active sessions ✓
  - involuntary session termination → "ended without your action" ✓
  - save in-progress work before commencement ✓
  - escalation through established channels → "report the issue through your standard support channels" ✓
- No invented details: ✓
- No dropped facts: ✓
- Plain language, not patronizing: ✓
- No "don't worry!": ✓
- End with what to do if something breaks: ✓ (report through standard support channels)

Wait, I want to double-check "involuntary session termination" → "ended without your action." Is that accurate? "Involuntary" means not by the user's will/choice. "Without your action" means it happens without you doing something. That's close enough and captures the meaning.

Actually, I realize "without your action" could be slightly ambiguous — it could mean "without you taking action" (i.e., you didn't do anything to cause it) which is the same as involuntary. I think it works.

One more thought: the original says "Escalation of service-impacting incidents should be directed through established channels." This means: if there's a service-impacting incident, escalate it through established channels. My version says "If the disruption extends beyond the expected 90 minutes, or if it causes a problem that goes beyond the planned maintenance, report the issue through your standard support channels." 

The original says "service-impacting incidents" — incidents that impact the service. My version says "the disruption extends beyond the expected 90 minutes, or if it causes a problem that goes beyond the planned maintenance." This captures the idea of a service-impacting incident (something beyond the planned maintenance). I think this is faithful.

"Established channels" → "standard support channels" — this is a reasonable plain-language equivalent.

I'm satisfied with this. Let me write the final answer.

</details>

**Answer (as scored):**

Before 02:00 UTC on the 14th of this month — convert that to your local time zone — please save all in-progress work. A scheduled infrastructure maintenance window begins at that time, during which authentication services will be intermittently unavailable for a period not exceeding 90 minutes. If you have an active session at the time, it may be ended without your action. If the disruption extends beyond the expected 90 minutes, or if it causes a problem that goes beyond the planned maintenance, report the issue through your standard support channels.

