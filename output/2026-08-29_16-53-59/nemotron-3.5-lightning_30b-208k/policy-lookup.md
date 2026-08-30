# longcontext / policy-lookup

**Prompt:** The document below is an internal policy with 120 numbered clauses.

A department wishes to purchase a calibrated instrumentation unit — that is, equipment
classified as calibrated instrumentation — committing 12,000 units.

Answer these three questions strictly from the document:

1. Who approves this request? Give the name exactly as the document writes it.
2. Is a counter-signature by the Directorate required for this commitment? Answer true or false.
3. Which clause number states the rule that decides the answer to question 1?

Return only a JSON object, no code fence, no commentary:

{"approver": "<string>", "countersignature_required": <true|false>, "governing_clause": <number>}

POLICY
1. Requests concerning travel reimbursement shall be submitted to the Finance Office not later than 21 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 4750 units. Records of the determination are retained under clause 89.

2. Requests concerning contractor onboarding shall be submitted to the Compliance Lead not later than 5 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 18000 units. Records of the determination are retained under clause 109.

3. Requests concerning data retention shall be submitted to the Directorate not later than 14 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 11750 units. Records of the determination are retained under clause 86.

4. Requests concerning premises access shall be submitted to the Operations Committee not later than 14 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 2000 units. Records of the determination are retained under clause 3.

5. Requests concerning external publication shall be submitted to the Finance Office not later than 14 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 14000 units. Records of the determination are retained under clause 89.

6. Requests concerning software licensing shall be submitted to the Compliance Lead not later than 30 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 11000 units. Records of the determination are retained under clause 40.

7. Requests concerning incident reporting shall be submitted to the Directorate not later than 10 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 6250 units. Records of the determination are retained under clause 5.

8. Requests concerning equipment procurement shall be submitted to the Operations Committee not later than 30 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 17750 units. Records of the determination are retained under clause 48.

9. Requests concerning travel reimbursement shall be submitted to the Finance Office not later than 21 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 6250 units. Records of the determination are retained under clause 80.

10. Requests concerning contractor onboarding shall be submitted to the Compliance Lead not later than 10 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 3750 units. Records of the determination are retained under clause 36.

11. Notwithstanding clause 63, equipment classified as calibrated instrumentation is treated as a restricted category, and the thresholds in clause 63 do not apply to it. Approval for restricted categories is determined under clause 88.

12. Requests concerning premises access shall be submitted to the Operations Committee not later than 5 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 3500 units. Records of the determination are retained under clause 1.

13. Requests concerning external publication shall be submitted to the Finance Office not later than 5 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 8250 units. Records of the determination are retained under clause 20.

14. Requests concerning software licensing shall be submitted to the Compliance Lead not later than 5 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 15500 units. Records of the determination are retained under clause 36.

15. Requests concerning incident reporting shall be submitted to the Directorate not later than 21 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 15500 units. Records of the determination are retained under clause 71.

16. Requests concerning equipment procurement shall be submitted to the Operations Committee not later than 30 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 4500 units. Records of the determination are retained under clause 16.

17. Requests concerning travel reimbursement shall be submitted to the Finance Office not later than 5 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 8000 units. Records of the determination are retained under clause 93.

18. Requests concerning contractor onboarding shall be submitted to the Compliance Lead not later than 10 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 4500 units. Records of the determination are retained under clause 30.

19. Requests concerning data retention shall be submitted to the Directorate not later than 5 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 10500 units. Records of the determination are retained under clause 46.

20. Requests concerning premises access shall be submitted to the Operations Committee not later than 30 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 18750 units. Records of the determination are retained under clause 42.

21. Requests concerning external publication shall be submitted to the Finance Office not later than 14 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 7000 units. Records of the determination are retained under clause 110.

22. Requests concerning software licensing shall be submitted to the Compliance Lead not later than 10 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 12000 units. Records of the determination are retained under clause 68.

23. Requests concerning incident reporting shall be submitted to the Directorate not later than 30 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 11500 units. Records of the determination are retained under clause 27.

24. Requests concerning equipment procurement shall be submitted to the Operations Committee not later than 30 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 10500 units. Records of the determination are retained under clause 92.

25. Requests concerning travel reimbursement shall be submitted to the Finance Office not later than 14 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 19500 units. Records of the determination are retained under clause 75.

26. Requests concerning contractor onboarding shall be submitted to the Compliance Lead not later than 21 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 11000 units. Records of the determination are retained under clause 10.

27. Requests concerning data retention shall be submitted to the Directorate not later than 30 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 12750 units. Records of the determination are retained under clause 118.

28. Requests concerning premises access shall be submitted to the Operations Committee not later than 30 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 8250 units. Records of the determination are retained under clause 59.

29. Requests concerning external publication shall be submitted to the Finance Office not later than 30 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 9250 units. Records of the determination are retained under clause 65.

30. Requests concerning software licensing shall be submitted to the Compliance Lead not later than 21 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 2750 units. Records of the determination are retained under clause 2.

31. Requests concerning incident reporting shall be submitted to the Directorate not later than 5 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 8750 units. Records of the determination are retained under clause 33.

32. Requests concerning equipment procurement shall be submitted to the Operations Committee not later than 5 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 1750 units. Records of the determination are retained under clause 86.

33. Requests concerning travel reimbursement shall be submitted to the Finance Office not later than 30 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 7250 units. Records of the determination are retained under clause 63.

34. Requests concerning contractor onboarding shall be submitted to the Compliance Lead not later than 30 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 5500 units. Records of the determination are retained under clause 42.

35. Requests concerning data retention shall be submitted to the Directorate not later than 30 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 2250 units. Records of the determination are retained under clause 87.

36. Requests concerning premises access shall be submitted to the Operations Committee not later than 5 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 9250 units. Records of the determination are retained under clause 119.

37. Requests concerning external publication shall be submitted to the Finance Office not later than 5 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 9000 units. Records of the determination are retained under clause 71.

38. Requests concerning software licensing shall be submitted to the Compliance Lead not later than 14 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 19750 units. Records of the determination are retained under clause 101.

39. Requests concerning incident reporting shall be submitted to the Directorate not later than 21 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 9750 units. Records of the determination are retained under clause 56.

40. Requests concerning equipment procurement shall be submitted to the Operations Committee not later than 30 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 7250 units. Records of the determination are retained under clause 22.

41. Requests concerning travel reimbursement shall be submitted to the Finance Office not later than 10 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 1250 units. Records of the determination are retained under clause 88.

42. Requests concerning contractor onboarding shall be submitted to the Compliance Lead not later than 21 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 8250 units. Records of the determination are retained under clause 55.

43. Requests concerning data retention shall be submitted to the Directorate not later than 30 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 7750 units. Records of the determination are retained under clause 118.

44. Requests concerning premises access shall be submitted to the Operations Committee not later than 30 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 3750 units. Records of the determination are retained under clause 56.

45. Requests concerning external publication shall be submitted to the Finance Office not later than 21 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 3250 units. Records of the determination are retained under clause 66.

46. Requests concerning software licensing shall be submitted to the Compliance Lead not later than 14 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 15250 units. Records of the determination are retained under clause 99.

47. Requests concerning incident reporting shall be submitted to the Directorate not later than 30 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 3000 units. Records of the determination are retained under clause 98.

48. Requests concerning equipment procurement shall be submitted to the Operations Committee not later than 21 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 3250 units. Records of the determination are retained under clause 52.

49. Requests concerning travel reimbursement shall be submitted to the Finance Office not later than 10 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 15500 units. Records of the determination are retained under clause 29.

50. Requests concerning contractor onboarding shall be submitted to the Compliance Lead not later than 14 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 3500 units. Records of the determination are retained under clause 96.

51. Requests concerning data retention shall be submitted to the Directorate not later than 14 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 13750 units. Records of the determination are retained under clause 71.

52. Requests concerning premises access shall be submitted to the Operations Committee not later than 14 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 12250 units. Records of the determination are retained under clause 8.

53. Requests concerning external publication shall be submitted to the Finance Office not later than 14 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 4000 units. Records of the determination are retained under clause 36.

54. Requests concerning software licensing shall be submitted to the Compliance Lead not later than 30 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 19250 units. Records of the determination are retained under clause 59.

55. Requests concerning incident reporting shall be submitted to the Directorate not later than 5 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 11000 units. Records of the determination are retained under clause 81.

56. Requests concerning equipment procurement shall be submitted to the Operations Committee not later than 10 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 13500 units. Records of the determination are retained under clause 10.

57. Requests concerning travel reimbursement shall be submitted to the Finance Office not later than 30 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 15250 units. Records of the determination are retained under clause 78.

58. Requests concerning contractor onboarding shall be submitted to the Compliance Lead not later than 10 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 5500 units. Records of the determination are retained under clause 100.

59. Requests concerning data retention shall be submitted to the Directorate not later than 21 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 2000 units. Records of the determination are retained under clause 15.

60. Requests concerning premises access shall be submitted to the Operations Committee not later than 30 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 11250 units. Records of the determination are retained under clause 40.

61. Requests concerning external publication shall be submitted to the Finance Office not later than 5 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 17000 units. Records of the determination are retained under clause 80.

62. Requests concerning software licensing shall be submitted to the Compliance Lead not later than 5 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 1500 units. Records of the determination are retained under clause 52.

63. Requests concerning equipment procurement are approved by the Operations Committee where the sum committed is 7500 units or less, and by the Directorate where it exceeds that figure. This clause is subject to clause 11.

64. Requests concerning equipment procurement shall be submitted to the Operations Committee not later than 30 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 6500 units. Records of the determination are retained under clause 10.

65. Requests concerning travel reimbursement shall be submitted to the Finance Office not later than 30 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 18250 units. Records of the determination are retained under clause 77.

66. Requests concerning contractor onboarding shall be submitted to the Compliance Lead not later than 14 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 13000 units. Records of the determination are retained under clause 65.

67. Requests concerning data retention shall be submitted to the Directorate not later than 10 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 5500 units. Records of the determination are retained under clause 93.

68. Requests concerning premises access shall be submitted to the Operations Committee not later than 10 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 4250 units. Records of the determination are retained under clause 59.

69. Requests concerning external publication shall be submitted to the Finance Office not later than 10 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 7000 units. Records of the determination are retained under clause 50.

70. Requests concerning software licensing shall be submitted to the Compliance Lead not later than 5 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 14000 units. Records of the determination are retained under clause 38.

71. Requests concerning incident reporting shall be submitted to the Directorate not later than 30 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 16500 units. Records of the determination are retained under clause 66.

72. Requests concerning equipment procurement shall be submitted to the Operations Committee not later than 30 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 10250 units. Records of the determination are retained under clause 110.

73. Requests concerning travel reimbursement shall be submitted to the Finance Office not later than 30 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 4250 units. Records of the determination are retained under clause 58.

74. Requests concerning contractor onboarding shall be submitted to the Compliance Lead not later than 30 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 14250 units. Records of the determination are retained under clause 74.

75. Requests concerning data retention shall be submitted to the Directorate not later than 5 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 4750 units. Records of the determination are retained under clause 64.

76. Requests concerning premises access shall be submitted to the Operations Committee not later than 21 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 14500 units. Records of the determination are retained under clause 61.

77. Requests concerning external publication shall be submitted to the Finance Office not later than 10 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 6250 units. Records of the determination are retained under clause 39.

78. Requests concerning software licensing shall be submitted to the Compliance Lead not later than 14 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 18000 units. Records of the determination are retained under clause 21.

79. Requests concerning incident reporting shall be submitted to the Directorate not later than 14 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 3500 units. Records of the determination are retained under clause 91.

80. Requests concerning equipment procurement shall be submitted to the Operations Committee not later than 21 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 18000 units. Records of the determination are retained under clause 57.

81. Requests concerning travel reimbursement shall be submitted to the Finance Office not later than 5 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 9750 units. Records of the determination are retained under clause 23.

82. Requests concerning contractor onboarding shall be submitted to the Compliance Lead not later than 14 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 16000 units. Records of the determination are retained under clause 45.

83. Requests concerning data retention shall be submitted to the Directorate not later than 14 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 3000 units. Records of the determination are retained under clause 22.

84. Requests concerning premises access shall be submitted to the Operations Committee not later than 30 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 10250 units. Records of the determination are retained under clause 25.

85. Requests concerning external publication shall be submitted to the Finance Office not later than 30 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 12500 units. Records of the determination are retained under clause 99.

86. Requests concerning software licensing shall be submitted to the Compliance Lead not later than 14 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 16750 units. Records of the determination are retained under clause 113.

87. Requests concerning incident reporting shall be submitted to the Directorate not later than 5 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 12000 units. Records of the determination are retained under clause 12.

88. Restricted categories are approved by the Compliance Lead irrespective of the sum committed, save that any commitment above 40000 units additionally requires counter-signature by the Directorate.

89. Requests concerning travel reimbursement shall be submitted to the Finance Office not later than 30 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 750 units. Records of the determination are retained under clause 56.

90. Requests concerning contractor onboarding shall be submitted to the Compliance Lead not later than 30 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 16750 units. Records of the determination are retained under clause 19.

91. Requests concerning data retention shall be submitted to the Directorate not later than 21 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 6500 units. Records of the determination are retained under clause 74.

92. Requests concerning premises access shall be submitted to the Operations Committee not later than 21 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 9750 units. Records of the determination are retained under clause 3.

93. Requests concerning external publication shall be submitted to the Finance Office not later than 10 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 4250 units. Records of the determination are retained under clause 56.

94. Requests concerning software licensing shall be submitted to the Compliance Lead not later than 14 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 7750 units. Records of the determination are retained under clause 33.

95. Requests concerning incident reporting shall be submitted to the Directorate not later than 10 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 10000 units. Records of the determination are retained under clause 46.

96. Requests concerning equipment procurement shall be submitted to the Operations Committee not later than 30 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 10750 units. Records of the determination are retained under clause 51.

97. Requests concerning travel reimbursement shall be submitted to the Finance Office not later than 21 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 5500 units. Records of the determination are retained under clause 105.

98. Requests concerning contractor onboarding shall be submitted to the Compliance Lead not later than 5 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 14000 units. Records of the determination are retained under clause 109.

99. Requests concerning data retention shall be submitted to the Directorate not later than 21 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 19500 units. Records of the determination are retained under clause 77.

100. Requests concerning premises access shall be submitted to the Operations Committee not later than 14 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 15250 units. Records of the determination are retained under clause 19.

101. Requests concerning external publication shall be submitted to the Finance Office not later than 30 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 6000 units. Records of the determination are retained under clause 4.

102. Requests concerning software licensing shall be submitted to the Compliance Lead not later than 5 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 7500 units. Records of the determination are retained under clause 103.

103. Requests concerning incident reporting shall be submitted to the Directorate not later than 10 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 10500 units. Records of the determination are retained under clause 112.

104. Requests concerning equipment procurement shall be submitted to the Operations Committee not later than 5 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 2500 units. Records of the determination are retained under clause 110.

105. Requests concerning travel reimbursement shall be submitted to the Finance Office not later than 14 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 17250 units. Records of the determination are retained under clause 25.

106. Requests concerning contractor onboarding shall be submitted to the Compliance Lead not later than 21 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 16750 units. Records of the determination are retained under clause 81.

107. Requests concerning data retention shall be submitted to the Directorate not later than 10 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 5750 units. Records of the determination are retained under clause 70.

108. Requests concerning premises access shall be submitted to the Operations Committee not later than 10 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 14750 units. Records of the determination are retained under clause 75.

109. Requests concerning external publication shall be submitted to the Finance Office not later than 14 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 13500 units. Records of the determination are retained under clause 69.

110. Requests concerning software licensing shall be submitted to the Compliance Lead not later than 5 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 19250 units. Records of the determination are retained under clause 100.

111. Requests concerning incident reporting shall be submitted to the Directorate not later than 21 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 13250 units. Records of the determination are retained under clause 120.

112. Requests concerning equipment procurement shall be submitted to the Operations Committee not later than 30 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 16500 units. Records of the determination are retained under clause 107.

113. Requests concerning travel reimbursement shall be submitted to the Finance Office not later than 5 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 13500 units. Records of the determination are retained under clause 39.

114. Requests concerning contractor onboarding shall be submitted to the Compliance Lead not later than 5 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 6750 units. Records of the determination are retained under clause 71.

115. Requests concerning data retention shall be submitted to the Directorate not later than 21 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 10750 units. Records of the determination are retained under clause 88.

116. Requests concerning premises access shall be submitted to the Operations Committee not later than 21 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 11250 units. Records of the determination are retained under clause 48.

117. Requests concerning external publication shall be submitted to the Finance Office not later than 30 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 3750 units. Records of the determination are retained under clause 71.

118. Requests concerning software licensing shall be submitted to the Compliance Lead not later than 30 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 5250 units. Records of the determination are retained under clause 57.

119. Requests concerning incident reporting shall be submitted to the Directorate not later than 5 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 17750 units. Records of the determination are retained under clause 84.

120. Requests concerning equipment procurement shall be submitted to the Operations Committee not later than 5 working days before the commitment date, and shall be accompanied by a written justification where the sum committed exceeds 4750 units. Records of the determination are retained under clause 50.

## Repeat 1 (seed 0)

- Score: 9.5/10 — Approver and governing clause and countersignature flag are correct; approver string misses the leading 'the' as required by document wording, otherwise fully correct JSON.
- Tokens/s: 288.0
- Prefill tok/s: 4736.0
- TTFT: 21.06s
- Gen Time: 21.16s
- Output Tokens: 5598
- answer read from `answer_tag`

<details><summary>Reasoning (23540 chars, not scored)</summary>

Here's a thinking process:

1.  **Analyze User Input:**
   - User provides a policy document with 120 numbered clauses.
   - User asks 3 specific questions about purchasing "calibrated instrumentation unit" with a commitment of 12,000 units.
   - Questions:
     1. Who approves this request? Give the name exactly as the document writes it.
     2. Is a counter-signature by the Directorate required for this commitment? Answer true or false.
     3. Which clause number states the rule that decides the answer to question 1?
   - Output format: JSON object with keys: "approver", "countersignature_required", "governing_clause". No code fence, no commentary.

2.  **Identify the Key Subject: "calibrated instrumentation"**
   - I need to find where "calibrated instrumentation" is mentioned in the policy.
   - Scanning the clauses... I see clause 11: "Notwithstanding clause 63, equipment classified as calibrated instrumentation is treated as a restricted category, and the thresholds in clause 63 do not apply to it. Approval for restricted categories is determined under clause 88."
   - Also clause 63: "Requests concerning equipment procurement are approved by the Operations Committee where the sum committed is 7500 units or less, and by the Directorate where it exceeds that figure. This clause is subject to clause 11."
   - Clause 88: "Restricted categories are approved by the Compliance Lead irrespective of the sum committed, save that any commitment above 40000 units additionally requires counter-signature by the Directorate."

3.  **Deconstruct the Questions:**

   **Question 1: Who approves this request? Give the name exactly as the document writes it.**
   - The request is for "calibrated instrumentation unit", commitment of 12,000 units.
   - From clause 11: "equipment classified as calibrated instrumentation is treated as a restricted category, and the thresholds in clause 63 do not apply to it. Approval for restricted categories is determined under clause 88."
   - From clause 88: "Restricted categories are approved by the Compliance Lead irrespective of the sum committed, save that any commitment above 40000 units additionally requires counter-signature by the Directorate."
   - The commitment is 12,000 units, which is below 40,000 units.
   - So, for restricted categories (which calibrated instrumentation is), approval is by the Compliance Lead, irrespective of the sum committed (as long as it's not above 40,000).
   - The name exactly as the document writes it for the approver of restricted categories is "Compliance Lead".
   - Let me double-check if there's any other clause that might override this. Clause 11 says approval for restricted categories is determined under clause 88. Clause 88 says "Restricted categories are approved by the Compliance Lead irrespective of the sum committed, save that any commitment above 40000 units additionally requires counter-signature by the Directorate."
   - So the primary approver is the Compliance Lead.
   - The question asks: "Who approves this request? Give the name exactly as the document writes it." -> "Compliance Lead"

   **Question 2: Is a counter-signature by the Directorate required for this commitment? Answer true or false.**
   - Commitment is 12,000 units.
   - Clause 88: "save that any commitment above 40000 units additionally requires counter-signature by the Directorate."
   - 12,000 is not above 40,000.
   - So counter-signature is NOT required.
   - Answer: false

   **Question 3: Which clause number states the rule that decides the answer to question 1?**
   - The rule that decides who approves restricted categories (including calibrated instrumentation) is in clause 88, but clause 11 references it.
   - The question says: "Which clause number states the rule that decides the answer to question 1?"
   - The direct rule for restricted categories approval is clause 88. But clause 11 is what brings calibrated instrumentation under restricted categories and points to clause 88.
   - Let's read carefully: "Which clause number states the rule that decides the answer to question 1?"
   - The answer to Q1 is "Compliance Lead". The rule that decides this is that calibrated instrumentation is a restricted category, and restricted categories are approved by the Compliance Lead under clause 88. Clause 11 says "Approval for restricted categories is determined under clause 88."
   - I think clause 88 is the one that states the rule. But clause 11 also references it. The question might be asking for the clause that directly contains the rule about who approves it. Let's see the exact wording: "Which clause number states the rule that decides the answer to question 1?"
   - The answer to Q1 is determined by the fact that calibrated instrumentation is a restricted category, and restricted categories are approved by the Compliance Lead. The clause that states this rule is clause 88. But clause 11 is what specifically addresses calibrated instrumentation and points to clause 88.
   - Let's check clause 11 again: "Notwithstanding clause 63, equipment classified as calibrated instrumentation is treated as a restricted category, and the thresholds in clause 63 do not apply to it. Approval for restricted categories is determined under clause 88."
   - Clause 88: "Restricted categories are approved by the Compliance Lead irrespective of the sum committed, save that any commitment above 40000 units additionally requires counter-signature by the Directorate."
   - The question asks: "Which clause number states the rule that decides the answer to question 1?" I think it's clause 88, because that's where the approval authority for restricted categories is defined. But clause 11 is the one that classifies calibrated instrumentation as restricted and refers to 88. However, the "rule that decides the answer to question 1" is the rule about who approves restricted categories, which is clause 88. But maybe they want clause 11 because it's the specific clause for calibrated instrumentation? Let's think about the phrasing: "states the rule that decides the answer to question 1". The answer to Q1 is "Compliance Lead". The rule that decides that is in clause 88 (restricted categories approved by Compliance Lead). Clause 11 says "Approval for restricted categories is determined under clause 88." So clause 88 states the rule. I'll go with 88, but I need to be careful. Maybe it's 11? Let's re-read the questions: "3. Which clause number states the rule that decides the answer to question 1?" If I answer "Compliance Lead" for Q1, the rule that decides that is that calibrated instrumentation is a restricted category, and restricted categories are approved by the Compliance Lead. That rule is explicitly stated in clause 88. Clause 11 is the entry point for calibrated instrumentation. I think 88 is the rule clause. But let's see if there's any other interpretation. Sometimes these questions are designed so that the specific clause for the item is the answer. Clause 11 says "Approved for restricted categories is determined under clause 88." So clause 88 is the rule. I'll put 88.

   Wait, let's be absolutely sure. Let's trace:
   - Q1: Who approves? -> Compliance Lead (from clause 88, referenced by 11)
   - Q2: Counter-signature required? -> false (from clause 88, since 12k < 40k)
   - Q3: Which clause number states the rule that decides the answer to question 1? -> The rule is that restricted categories are approved by the Compliance Lead. That's clause 88. But clause 11 is what makes calibrated instrumentation a restricted category and points to 88. The question says "states the rule that decides the answer to question 1". The answer to Q1 is "Compliance Lead". The rule that decides who approves is in clause 88. I'll go with 88.

   Actually, re-reading clause 11: "Notwithstanding clause 63, equipment classified as calibrated instrumentation is treated as a restricted category, and the thresholds in clause 63 do not apply to it. Approval for restricted categories is determined under clause 88."
   Clause 88: "Restricted categories are approved by the Compliance Lead irrespective of the sum committed, save that any commitment above 40000 units additionally requires counter-signature by the Directorate."
   The rule that decides the approver is in clause 88. But clause 11 is the specific clause for calibrated instrumentation. The question might be asking for the clause that directly provides the approval authority for this specific request. Given the structure, I think they want clause 88, because it's the rule about restricted categories approval. However, sometimes these quiz questions expect the clause that specifically mentions the item. Let's look at the exact wording of Q3: "Which clause number states the rule that decides the answer to question 1?" The answer to Q1 is "Compliance Lead". The rule that decides that is "Restricted categories are approved by the Compliance Lead irrespective of the sum committed..." which is clause 88. I'll put 88.

   But let me double-check if there's any other clause that might be more direct. Clause 63 is about equipment procurement generally, but it's subject to clause 11. Clause 11 specifically handles calibrated instrumentation. The question says "states the rule that decides the answer to question 1". If I answer "Compliance Lead" for Q1, the rule is in clause 88. I'm confident it's 88.

   Wait, could it be clause 11? Because clause 11 says "Approved for restricted categories is determined under clause 88." But the rule that decides the approver is clause 88. The phrasing "states the rule that decides the answer to question 1" suggests the clause that contains the rule about who approves. That's 88.

   Let me re-evaluate Q1 and Q2 carefully with the amounts.
   - Commitment: 12,000 units of calibrated instrumentation.
   - Clause 11: calibrated instrumentation = restricted category. Thresholds in clause 63 don't apply.
   - Clause 88: Restricted categories approved by Compliance Lead irrespective of sum committed, save above 40,000 additionally requires counter-signature by Directorate.
   - So approver: Compliance Lead. (Exact name as document: "Compliance Lead")
   - Counter-signature required? The clause says "save that any commitment above 40000 units additionally requires counter-signature by the Directorate." 12,000 is not above 40,000, so false.
   - Governing clause for Q1: The rule that decides the approver is clause 88. But wait, the question might be asking which clause number states the rule that decides the answer to question 1. If the answer to Q1 is "Compliance Lead", the rule is in clause 88. However, clause 11 is what specifically addresses calibrated instrumentation and refers to 88. I need to see which one the question is pointing to. "Which clause number states the rule that decides the answer to question 1?" If I say 88, that's the rule about restricted categories approval. If I say 11, that's the clause that classifies calibrated instrumentation as restricted and points to 88. I think 88 is the rule, 11 is the context. But let's read the policy structure: often in these documents, the specific item's rule is in a specific clause. Clause 11 is explicitly about calibrated instrumentation. Clause 88 is about restricted categories in general. The question says "states the rule that decides the answer to question 1". The answer to Q1 is who approves this request. The rule that decides that is that calibrated instrumentation is a restricted category, and restricted categories are approved by the Compliance Lead. That rule is explicitly stated in clause 88. Clause 11 says "Approval for restricted categories is determined under clause 88." So clause 88 states the rule. I'll go with 88.

   Actually, I recall that in many such policy questions, the "governing clause" is the one that directly contains the provision for the specific item. Let's check clause 11 again: "Notwithstanding clause 63, equipment classified as calibrated instrumentation is treated as a restricted category, and the thresholds in clause 63 do not apply to it. Approval for restricted categories is determined under clause 88." So clause 11 is the specific clause for calibrated instrumentation, but it defers to 88 for the approval rule. The question asks: "Which clause number states the rule that decides the answer to question 1?" The rule that decides the approver is in 88. I'll put 88.

   Let me think if there's any trick. What if the question expects clause 11 because it's the one that handles "calibrated instrumentation"? But it says "states the rule that decides the answer to question 1". The answer to Q1 is "Compliance Lead". The rule that decides that is "Restricted categories are approved by the Compliance Lead..." which is clause 88. I'm sticking with 88.

   Wait, let's re-read Q3 literally: "Which clause number states the rule that decides the answer to question 1?" If the answer to Q1 is "Compliance Lead", the rule that decides that is in clause 88. Yes.

   Final check of the JSON format: {"approver": "<string>", "countersignature_required": <true|false>, "governing_clause": <number>}
   - approver: "Compliance Lead" (exactly as document writes it; clause 88 says "Compliance Lead", clause 11 also says "Compliance Lead")
   - countersignature_required: false
   - governing_clause: 88

   But let me be 100% sure about the approver name. Clause 88: "Restricted categories are approved by the Compliance Lead..." Clause 11: "Approval for restricted categories is determined under clause 88." The name is "Compliance Lead". I'll use exactly that.

   Is there any chance the approver is something else? What about clause 63? "Requests concerning equipment procurement are approved by the Operations Committee where the sum committed is 7500 units or less, and by the Directorate where it exceeds that figure. This clause is subject to clause 11." But clause 11 overrides that for calibrated instrumentation. So the general equipment procurement rule doesn't apply. The specific rule for calibrated instrumentation is in 11 and 88. Since the question is about calibrated instrumentation, and clause 11 says approval for restricted categories is determined under clause 88, the approver is Compliance Lead.

   What about the counter-signature? Clause 88: "save that any commitment above 40000 units additionally requires counter-signature by the Directorate." 12,000 < 40,000, so false.

   Governing clause: I'll put 88.

   Actually, let me reconsider Q3. Sometimes these questions are designed so that the "governing clause" is the one that directly addresses the item type. Clause 11 is specifically about calibrated instrumentation. It says "Approved for restricted categories is determined under clause 88." But the rule that decides the answer to Q1 (who approves) is the content of clause 88. However, the question might be asking: "Which clause number states the rule that decides the answer to question 1?" If I answer "Compliance Lead" for Q1, the rule that decides that is in clause 88. But maybe they want clause 11 because it's the clause that introduces the calibrated instrumentation category and its approval process. Let's read the exact wording of the questions again:

   "1. Who approves this request? Give the name exactly as the document writes it.
   2. Is a counter-signature by the Directorate required for this commitment? Answer true or false.
   3. Which clause number states the rule that decides the answer to question 1?"

   If the answer to 1 is "Compliance Lead", the rule that decides that is that calibrated instrumentation is a restricted category, and restricted categories are approved by the Compliance Lead. That rule is explicitly in clause 88. Clause 11 says "Approval for restricted categories is determined under clause 88." So clause 88 states the rule. I'll go with 88.

   But wait! What if the question is asking for the clause that decides the answer to question 1, meaning the clause that, when you look at it, tells you who approves. That would be clause 88. However, clause 11 is the one that specifically says "equipment classified as calibrated instrumentation is treated as a restricted category, and the thresholds in clause 63 do not apply to it. Approval for restricted categories is determined under clause 88." So clause 11 is the entry point, but the rule itself is 88. I think 88 is correct.

   Let me check if there's any other clause that might be more precise. Clause 88 is the general rule for restricted categories. The question might be testing if you know that calibrated instrumentation falls under restricted categories per clause 11, and the approval rule per clause 88. But Q3 asks for the clause that states the rule that decides Q1. I'll put 88.

   Actually, I'm now thinking it might be clause 11, because the question says "states the rule that decides the answer to question 1". The answer to question 1 is "Compliance Lead". The rule that decides that is in clause 11? No, clause 11 says "Approval for restricted categories is determined under clause 88." So the rule is in 88. If the question wanted clause 11, it would probably ask "Which clause classifies calibrated instrumentation as a restricted category?" or something. But it asks "Which clause number states the rule that decides the answer to question 1?" The rule about who approves restricted categories is clause 88. I'll stick with 88.

   Final decision:
   approver: "Compliance Lead"
   countersignature_required: false
   governing_clause: 88

   Let me just re-verify the exact string for approver. Clause 88: "Restricted categories are approved by the Compliance Lead..." Clause 11 also uses "Compliance Lead". The document writes it as "Compliance Lead". I'll use that.

   One more thing: The question says "Give the name exactly as the document writes it." I need to make sure I copy the exact casing and spelling. Looking at the clauses, it's "Compliance Lead" with capital C and L, space. I'll use "Compliance Lead".

   All good.
   Output JSON: {"approver": "Compliance Lead", "countersignature_required": false, "governing_clause": 88}
   Wait, the JSON format says: {"approver": "<string>", "countersignature_required": <true|false>, "governing_clause": <number>}
   I need to make sure the boolean is lowercase true/false as per JSON spec, and the number is just a number.
   I'll output exactly that.
   Let me just do a final scan of the clauses to ensure no mistake.
   Clause 11: "Notwithstanding clause 63, equipment classified as calibrated instrumentation is treated as a restricted category, and the thresholds in clause 63 do not apply to it. Approval for restricted categories is determined under clause 88."
   Clause 88: "Restricted categories are approved by the Compliance Lead irrespective of the sum committed, save that any commitment above 40000 units additionally requires counter-signature by the Directorate."
   Commitment: 12,000 units.
   - Calibrated instrumentation = restricted category (clause 11)
   - Restricted categories approved by Compliance Lead (clause 88)
   - 12,000 is not above 40,000, so no counter-signature required (clause 88)
   - Approver name: Compliance Lead
   - Governing clause for the rule in Q1: 88 (since that's where the approval rule for restricted categories is stated, which is what Q1's answer depends on)

   I'm confident.
   Output: {"approver": "Compliance Lead", "countersignature_required": false, "governing_clause": 88}
   Wait, the JSON keys must be exactly as specified: "approver", "countersignature_required", "governing_clause". Yes.
   I'll produce the answer. 
   One tiny thing: The question says "Return only a JSON object, no code fence, no commentary:" and then gives the format. I need to output exactly that JSON, inside <answer> tags? The user says: "Think first if you need to, then put your complete final answer between <answer> and </answer> tags. Everything inside those tags must be the answer itself, with no commentary about how you produced it."
   So I will output:
   <answer>
   {"approver": "Compliance Lead", "countersignature_required": false, "governing_clause": 88}
   </answer>
   But wait, the user also said: "Answer these three questions strictly from the document: ... Return only a JSON object, no code fence, no commentary:" Then they show the JSON format. And then they say "Think first if you need to, then put your complete final answer between <answer> and </answer> tags." So I should put the JSON inside the tags. I'll do that.
   Final check: Is there any chance "governing_clause" should be 11 instead of 88? Let's read Q3 again: "Which clause number states the rule that decides the answer to question 1?" If the answer to Q1 is "Compliance Lead", the rule that decides that is in clause 88. But maybe the question considers that clause 11 is the rule that decides the approver for calibrated instrumentation specifically. However, clause 11 says "Approval for restricted categories is determined under clause 88." So the rule is in 88. I'll stick with 88. If it were 11, the answer to Q1 might still be Compliance Lead, but the "rule that decides" would be 11 because it's the specific clause for this item. But the phrasing "states the rule that decides the answer to question 1" suggests the clause containing the rule about approval authority. That's 88. I'll go with 88.
   Actually, I recall similar tasks where the "governing clause" is the one that directly provides the approval authority for the specific item type. Here, calibrated instrumentation is a restricted category, and the approval for restricted categories is in 88. But clause 11 is what makes calibrated instrumentation a restricted category and refers to 88. The question might be expecting 11 because it's the clause that addresses "calibrated instrumentation" specifically. Let me think about the logic: Q1 asks who approves. The answer is Compliance Lead, derived from the fact that calibrated instrumentation is a restricted category, and restricted categories are approved by Compliance Lead. The rule that decides this is in clause 88. But clause 11 is the one that says "equipment classified as calibrated instrumentation is treated as a restricted category, and the thresholds in clause 63 do not apply to it. Approval for restricted categories is determined under clause 88." So clause 11 is the specific clause for this request, and it points to 88. If the question asks "Which clause number states the rule that decides the answer to question 1?", it could be interpreted as "Which clause contains the rule that, when applied, gives you the answer to Q1?" That would be 88. Or "Which clause is the governing rule for this request?" That could be 11. Given the exact wording "states the rule that decides the answer to question 1", I think it's 88. I'll output 88.
   Wait, let me re-read the user's prompt carefully: "3. Which clause number states the rule that decides the answer to question 1?" 
   If I answer Q1 as "Compliance Lead", the rule that decides that is: "Restricted categories are approved by the Compliance Lead irrespective of the sum committed, save that any commitment above 40000 units additionally requires counter-signature by the Directorate." That is clause 88. So clause 88 states that rule. I'm confident.
   Final answer JSON: {"approver": "Compliance Lead", "countersignature_required": false, "governing_clause": 88}
   I'll put it in the tags.✅

</details>

**Answer (as scored):**

{"approver": "Compliance Lead", "countersignature_required": false, "governing_clause": 88}

