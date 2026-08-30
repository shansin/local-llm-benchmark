# longcontext / ledger-audit

**Prompt:** Below is a transaction ledger with 400 rows. Each row is:

    ID | DATE | VENDOR | KIND | AMOUNT | STATUS

Answer these five questions about the ledger:

1. What is the total amount of all rows that are vendor "Halloway Instruments", kind "invoice", and status "approved"? Sum the amounts and give the total to two decimal places.
2. How many rows in the whole ledger have status "rejected"?
3. Of the rows with kind "refund" and status "approved", which has the largest amount? Give its ID.
4. How many distinct vendors appear anywhere in the ledger?
5. How many distinct vendors have at least one row with status "pending"?

Return only a JSON object, with no code fence and no commentary, in exactly this shape:

{"approved_invoice_total": <number>, "rejected_count": <number>, "largest_refund_id": "<string>", "distinct_vendors": <number>, "vendors_with_pending": <number>}

LEDGER
TXN-1000 | 2025-05-04 | Ambleside Print | adjustment | 7541.32 | approved
TXN-1001 | 2025-02-01 | Pemberton Freight | refund | 6193.24 | approved
TXN-1002 | 2025-01-16 | Northwind Supplies | refund | 8977.53 | rejected
TXN-1003 | 2025-06-02 | Ambleside Print | adjustment | 3395.92 | rejected
TXN-1004 | 2025-05-26 | Northwind Supplies | refund | 327.08 | rejected
TXN-1005 | 2025-05-02 | Ambleside Print | adjustment | 1973.96 | pending
TXN-1006 | 2025-03-05 | Quillon Software | adjustment | 910.21 | approved
TXN-1007 | 2025-04-14 | Rivet & Sons | invoice | 2162.63 | rejected
TXN-1008 | 2025-03-06 | Cortez Logistics | adjustment | 1628.74 | rejected
TXN-1009 | 2025-04-09 | Quillon Software | refund | 6048.33 | approved
TXN-1010 | 2025-03-15 | Rivet & Sons | refund | 5791.53 | approved
TXN-1011 | 2025-06-02 | Northwind Supplies | invoice | 9399.39 | pending
TXN-1012 | 2025-04-09 | Rivet & Sons | adjustment | 3285.28 | approved
TXN-1013 | 2025-03-28 | Halloway Instruments | adjustment | 3672.13 | approved
TXN-1014 | 2025-04-25 | Rivet & Sons | adjustment | 7858.23 | pending
TXN-1015 | 2025-04-09 | Quillon Software | adjustment | 1697.41 | pending
TXN-1016 | 2025-06-11 | Cortez Logistics | refund | 4932.77 | rejected
TXN-1017 | 2025-06-24 | Quillon Software | refund | 4038.76 | approved
TXN-1018 | 2025-03-26 | Saltmarsh Catering | adjustment | 6912.35 | pending
TXN-1019 | 2025-03-31 | Saltmarsh Catering | invoice | 6583.62 | approved
TXN-1020 | 2025-04-27 | Cortez Logistics | refund | 4888.72 | rejected
TXN-1021 | 2025-04-30 | Northwind Supplies | refund | 1754.33 | approved
TXN-1022 | 2025-02-23 | Halloway Instruments | refund | 1852.33 | approved
TXN-1023 | 2025-02-24 | Saltmarsh Catering | invoice | 2570.73 | pending
TXN-1024 | 2025-05-10 | Halloway Instruments | adjustment | 4245.63 | approved
TXN-1025 | 2025-05-06 | Saltmarsh Catering | invoice | 3888.71 | approved
TXN-1026 | 2025-04-27 | Halloway Instruments | refund | 2856.96 | pending
TXN-1027 | 2025-04-24 | Pemberton Freight | invoice | 7871.58 | approved
TXN-1028 | 2025-04-08 | Northwind Supplies | refund | 1588.04 | rejected
TXN-1029 | 2025-01-13 | Halloway Instruments | invoice | 4230.28 | approved
TXN-1030 | 2025-01-06 | Northwind Supplies | invoice | 5552.98 | pending
TXN-1031 | 2025-03-17 | Ambleside Print | adjustment | 2656.90 | rejected
TXN-1032 | 2025-02-26 | Pemberton Freight | adjustment | 5027.20 | rejected
TXN-1033 | 2025-03-19 | Quillon Software | adjustment | 7952.87 | pending
TXN-1034 | 2025-03-25 | Cortez Logistics | adjustment | 2673.04 | rejected
TXN-1035 | 2025-02-11 | Cortez Logistics | adjustment | 9629.72 | rejected
TXN-1036 | 2025-06-30 | Saltmarsh Catering | adjustment | 5584.64 | approved
TXN-1037 | 2025-05-29 | Cortez Logistics | adjustment | 4918.11 | rejected
TXN-1038 | 2025-01-06 | Saltmarsh Catering | refund | 7912.69 | pending
TXN-1039 | 2025-05-24 | Ambleside Print | invoice | 261.80 | rejected
TXN-1040 | 2025-02-01 | Ambleside Print | adjustment | 1601.61 | pending
TXN-1041 | 2025-04-10 | Saltmarsh Catering | invoice | 2429.92 | approved
TXN-1042 | 2025-06-13 | Ambleside Print | refund | 497.10 | approved
TXN-1043 | 2025-05-12 | Ambleside Print | invoice | 98.52 | rejected
TXN-1044 | 2025-05-27 | Ambleside Print | refund | 3254.03 | rejected
TXN-1045 | 2025-01-05 | Northwind Supplies | invoice | 1516.10 | pending
TXN-1046 | 2025-04-21 | Pemberton Freight | adjustment | 9788.41 | rejected
TXN-1047 | 2025-05-01 | Ambleside Print | invoice | 3807.49 | approved
TXN-1048 | 2025-02-06 | Quillon Software | adjustment | 5305.98 | approved
TXN-1049 | 2025-06-07 | Pemberton Freight | adjustment | 4894.93 | approved
TXN-1050 | 2025-03-23 | Northwind Supplies | invoice | 6181.55 | rejected
TXN-1051 | 2025-06-09 | Quillon Software | adjustment | 9097.28 | approved
TXN-1052 | 2025-06-28 | Halloway Instruments | invoice | 9564.00 | pending
TXN-1053 | 2025-06-12 | Quillon Software | refund | 395.29 | rejected
TXN-1054 | 2025-02-16 | Quillon Software | refund | 222.37 | approved
TXN-1055 | 2025-02-21 | Ambleside Print | invoice | 9355.83 | pending
TXN-1056 | 2025-01-06 | Ambleside Print | refund | 4316.43 | rejected
TXN-1057 | 2025-04-23 | Quillon Software | refund | 7472.87 | rejected
TXN-1058 | 2025-06-20 | Cortez Logistics | adjustment | 5416.06 | rejected
TXN-1059 | 2025-02-08 | Halloway Instruments | invoice | 2134.49 | pending
TXN-1060 | 2025-05-22 | Ambleside Print | invoice | 5572.45 | pending
TXN-1061 | 2025-03-29 | Ambleside Print | refund | 5734.89 | approved
TXN-1062 | 2025-01-23 | Rivet & Sons | invoice | 3246.50 | pending
TXN-1063 | 2025-03-21 | Northwind Supplies | invoice | 5786.46 | rejected
TXN-1064 | 2025-06-21 | Pemberton Freight | adjustment | 5931.96 | approved
TXN-1065 | 2025-05-20 | Saltmarsh Catering | refund | 3972.90 | pending
TXN-1066 | 2025-05-01 | Saltmarsh Catering | adjustment | 6698.83 | rejected
TXN-1067 | 2025-03-23 | Rivet & Sons | invoice | 2006.39 | rejected
TXN-1068 | 2025-05-04 | Rivet & Sons | adjustment | 4670.07 | approved
TXN-1069 | 2025-04-13 | Cortez Logistics | adjustment | 5019.79 | pending
TXN-1070 | 2025-03-17 | Cortez Logistics | adjustment | 3367.36 | pending
TXN-1071 | 2025-03-25 | Quillon Software | adjustment | 9415.19 | rejected
TXN-1072 | 2025-02-26 | Rivet & Sons | refund | 6151.78 | approved
TXN-1073 | 2025-02-19 | Cortez Logistics | refund | 6204.77 | pending
TXN-1074 | 2025-02-20 | Rivet & Sons | adjustment | 6196.92 | rejected
TXN-1075 | 2025-06-09 | Saltmarsh Catering | adjustment | 2357.47 | pending
TXN-1076 | 2025-05-10 | Northwind Supplies | refund | 4891.35 | rejected
TXN-1077 | 2025-05-28 | Cortez Logistics | refund | 7346.32 | pending
TXN-1078 | 2025-01-01 | Cortez Logistics | refund | 4080.28 | rejected
TXN-1079 | 2025-01-31 | Pemberton Freight | refund | 3846.91 | pending
TXN-1080 | 2025-03-31 | Pemberton Freight | invoice | 6952.62 | approved
TXN-1081 | 2025-01-05 | Pemberton Freight | invoice | 6832.75 | rejected
TXN-1082 | 2025-03-30 | Pemberton Freight | refund | 1519.01 | rejected
TXN-1083 | 2025-05-16 | Rivet & Sons | invoice | 8730.69 | pending
TXN-1084 | 2025-05-25 | Ambleside Print | refund | 6659.72 | pending
TXN-1085 | 2025-03-10 | Pemberton Freight | invoice | 888.82 | pending
TXN-1086 | 2025-02-20 | Northwind Supplies | adjustment | 7373.97 | pending
TXN-1087 | 2025-04-13 | Saltmarsh Catering | invoice | 1104.32 | rejected
TXN-1088 | 2025-05-31 | Rivet & Sons | adjustment | 4994.69 | pending
TXN-1089 | 2025-04-26 | Pemberton Freight | adjustment | 1796.51 | approved
TXN-1090 | 2025-03-02 | Saltmarsh Catering | adjustment | 7172.76 | approved
TXN-1091 | 2025-04-04 | Rivet & Sons | refund | 1688.26 | rejected
TXN-1092 | 2025-04-04 | Rivet & Sons | adjustment | 635.40 | pending
TXN-1093 | 2025-04-10 | Saltmarsh Catering | refund | 5664.31 | approved
TXN-1094 | 2025-04-03 | Saltmarsh Catering | refund | 7437.33 | rejected
TXN-1095 | 2025-06-01 | Pemberton Freight | invoice | 2844.88 | approved
TXN-1096 | 2025-01-21 | Cortez Logistics | adjustment | 1107.86 | pending
TXN-1097 | 2025-05-26 | Saltmarsh Catering | adjustment | 9328.68 | rejected
TXN-1098 | 2025-02-09 | Ambleside Print | invoice | 3543.52 | pending
TXN-1099 | 2025-02-27 | Cortez Logistics | adjustment | 5566.45 | pending
TXN-1100 | 2025-04-15 | Quillon Software | refund | 2850.37 | rejected
TXN-1101 | 2025-01-14 | Pemberton Freight | refund | 457.11 | pending
TXN-1102 | 2025-05-07 | Northwind Supplies | refund | 7654.32 | pending
TXN-1103 | 2025-05-15 | Cortez Logistics | adjustment | 774.27 | rejected
TXN-1104 | 2025-02-17 | Cortez Logistics | refund | 4463.44 | approved
TXN-1105 | 2025-06-01 | Rivet & Sons | adjustment | 2620.85 | approved
TXN-1106 | 2025-02-25 | Rivet & Sons | invoice | 8107.07 | pending
TXN-1107 | 2025-01-31 | Quillon Software | invoice | 9242.53 | approved
TXN-1108 | 2025-04-07 | Saltmarsh Catering | refund | 1240.37 | pending
TXN-1109 | 2025-04-25 | Saltmarsh Catering | invoice | 7382.93 | rejected
TXN-1110 | 2025-06-14 | Ambleside Print | refund | 4653.81 | pending
TXN-1111 | 2025-06-23 | Saltmarsh Catering | adjustment | 2538.09 | approved
TXN-1112 | 2025-01-12 | Saltmarsh Catering | refund | 5418.29 | pending
TXN-1113 | 2025-05-14 | Northwind Supplies | refund | 8219.43 | approved
TXN-1114 | 2025-03-09 | Northwind Supplies | invoice | 4841.72 | rejected
TXN-1115 | 2025-04-16 | Halloway Instruments | invoice | 2416.43 | rejected
TXN-1116 | 2025-01-18 | Ambleside Print | adjustment | 1543.61 | approved
TXN-1117 | 2025-04-15 | Cortez Logistics | refund | 536.48 | approved
TXN-1118 | 2025-05-19 | Northwind Supplies | refund | 3079.82 | rejected
TXN-1119 | 2025-01-23 | Rivet & Sons | invoice | 9384.97 | rejected
TXN-1120 | 2025-03-23 | Ambleside Print | adjustment | 4778.66 | rejected
TXN-1121 | 2025-01-09 | Rivet & Sons | invoice | 1601.43 | pending
TXN-1122 | 2025-04-21 | Saltmarsh Catering | invoice | 596.74 | pending
TXN-1123 | 2025-02-26 | Saltmarsh Catering | refund | 8581.30 | approved
TXN-1124 | 2025-05-10 | Pemberton Freight | invoice | 9464.42 | approved
TXN-1125 | 2025-01-08 | Quillon Software | invoice | 9047.71 | rejected
TXN-1126 | 2025-06-30 | Halloway Instruments | invoice | 1267.81 | approved
TXN-1127 | 2025-04-25 | Quillon Software | adjustment | 6679.67 | pending
TXN-1128 | 2025-04-18 | Saltmarsh Catering | refund | 6674.71 | pending
TXN-1129 | 2025-06-04 | Saltmarsh Catering | invoice | 1129.23 | pending
TXN-1130 | 2025-04-15 | Northwind Supplies | invoice | 7950.48 | rejected
TXN-1131 | 2025-06-10 | Halloway Instruments | adjustment | 3119.78 | pending
TXN-1132 | 2025-04-14 | Rivet & Sons | adjustment | 7229.30 | rejected
TXN-1133 | 2025-02-23 | Northwind Supplies | adjustment | 1163.74 | approved
TXN-1134 | 2025-06-23 | Saltmarsh Catering | adjustment | 7815.52 | rejected
TXN-1135 | 2025-03-03 | Ambleside Print | invoice | 4297.05 | rejected
TXN-1136 | 2025-03-26 | Pemberton Freight | adjustment | 4690.88 | rejected
TXN-1137 | 2025-04-28 | Cortez Logistics | adjustment | 1370.22 | pending
TXN-1138 | 2025-06-28 | Cortez Logistics | adjustment | 3123.89 | approved
TXN-1139 | 2025-01-20 | Halloway Instruments | invoice | 4330.47 | approved
TXN-1140 | 2025-05-14 | Halloway Instruments | adjustment | 6163.03 | rejected
TXN-1141 | 2025-02-13 | Halloway Instruments | refund | 521.65 | pending
TXN-1142 | 2025-01-07 | Rivet & Sons | refund | 4871.97 | approved
TXN-1143 | 2025-03-08 | Halloway Instruments | refund | 8205.58 | pending
TXN-1144 | 2025-02-18 | Ambleside Print | invoice | 8543.63 | rejected
TXN-1145 | 2025-03-04 | Ambleside Print | invoice | 3997.82 | rejected
TXN-1146 | 2025-01-26 | Halloway Instruments | adjustment | 4566.62 | approved
TXN-1147 | 2025-01-18 | Rivet & Sons | invoice | 6147.65 | pending
TXN-1148 | 2025-04-21 | Saltmarsh Catering | refund | 6482.18 | rejected
TXN-1149 | 2025-05-13 | Northwind Supplies | adjustment | 9665.08 | pending
TXN-1150 | 2025-01-16 | Halloway Instruments | adjustment | 4176.28 | pending
TXN-1151 | 2025-03-27 | Saltmarsh Catering | invoice | 7432.75 | rejected
TXN-1152 | 2025-04-09 | Northwind Supplies | refund | 2473.51 | rejected
TXN-1153 | 2025-02-16 | Ambleside Print | refund | 5396.51 | pending
TXN-1154 | 2025-04-12 | Rivet & Sons | refund | 8606.21 | pending
TXN-1155 | 2025-05-26 | Pemberton Freight | invoice | 8915.38 | approved
TXN-1156 | 2025-01-10 | Halloway Instruments | refund | 5059.53 | approved
TXN-1157 | 2025-04-08 | Pemberton Freight | invoice | 3315.07 | rejected
TXN-1158 | 2025-05-14 | Ambleside Print | refund | 7224.73 | rejected
TXN-1159 | 2025-05-25 | Pemberton Freight | refund | 3226.37 | rejected
TXN-1160 | 2025-01-03 | Saltmarsh Catering | refund | 1558.51 | pending
TXN-1161 | 2025-02-16 | Cortez Logistics | adjustment | 9788.68 | approved
TXN-1162 | 2025-03-04 | Quillon Software | invoice | 8199.59 | pending
TXN-1163 | 2025-06-04 | Saltmarsh Catering | invoice | 4163.12 | approved
TXN-1164 | 2025-06-09 | Pemberton Freight | refund | 4111.73 | pending
TXN-1165 | 2025-02-28 | Saltmarsh Catering | invoice | 5433.61 | approved
TXN-1166 | 2025-06-13 | Rivet & Sons | refund | 2625.14 | approved
TXN-1167 | 2025-06-28 | Cortez Logistics | invoice | 6588.95 | rejected
TXN-1168 | 2025-02-25 | Quillon Software | adjustment | 2195.70 | approved
TXN-1169 | 2025-03-19 | Halloway Instruments | invoice | 3888.03 | pending
TXN-1170 | 2025-04-09 | Northwind Supplies | refund | 1414.34 | rejected
TXN-1171 | 2025-05-17 | Cortez Logistics | adjustment | 2725.88 | rejected
TXN-1172 | 2025-04-10 | Halloway Instruments | refund | 7283.35 | pending
TXN-1173 | 2025-05-06 | Cortez Logistics | refund | 3550.27 | pending
TXN-1174 | 2025-02-28 | Saltmarsh Catering | adjustment | 5796.86 | approved
TXN-1175 | 2025-04-07 | Saltmarsh Catering | refund | 7441.60 | pending
TXN-1176 | 2025-01-01 | Halloway Instruments | refund | 225.76 | rejected
TXN-1177 | 2025-04-24 | Quillon Software | invoice | 5840.47 | rejected
TXN-1178 | 2025-01-26 | Rivet & Sons | adjustment | 9789.97 | approved
TXN-1179 | 2025-02-05 | Quillon Software | invoice | 7571.80 | rejected
TXN-1180 | 2025-02-01 | Northwind Supplies | refund | 2248.48 | approved
TXN-1181 | 2025-05-02 | Halloway Instruments | adjustment | 2866.50 | pending
TXN-1182 | 2025-03-23 | Rivet & Sons | refund | 502.98 | rejected
TXN-1183 | 2025-05-12 | Ambleside Print | adjustment | 1796.01 | pending
TXN-1184 | 2025-01-15 | Cortez Logistics | invoice | 3498.45 | pending
TXN-1185 | 2025-03-11 | Cortez Logistics | adjustment | 1201.90 | rejected
TXN-1186 | 2025-06-01 | Cortez Logistics | invoice | 4007.08 | rejected
TXN-1187 | 2025-01-21 | Rivet & Sons | invoice | 8079.86 | pending
TXN-1188 | 2025-04-20 | Pemberton Freight | invoice | 4942.45 | rejected
TXN-1189 | 2025-04-13 | Northwind Supplies | refund | 8588.85 | approved
TXN-1190 | 2025-04-24 | Rivet & Sons | refund | 5883.69 | rejected
TXN-1191 | 2025-02-01 | Quillon Software | adjustment | 1793.59 | pending
TXN-1192 | 2025-06-26 | Rivet & Sons | refund | 9709.95 | pending
TXN-1193 | 2025-03-19 | Saltmarsh Catering | refund | 8748.08 | rejected
TXN-1194 | 2025-04-01 | Rivet & Sons | refund | 5120.28 | approved
TXN-1195 | 2025-04-22 | Rivet & Sons | refund | 2865.59 | approved
TXN-1196 | 2025-04-19 | Northwind Supplies | invoice | 6723.80 | rejected
TXN-1197 | 2025-06-13 | Cortez Logistics | refund | 1628.82 | rejected
TXN-1198 | 2025-04-16 | Cortez Logistics | adjustment | 2069.08 | pending
TXN-1199 | 2025-05-11 | Halloway Instruments | invoice | 2122.83 | pending
TXN-1200 | 2025-06-30 | Rivet & Sons | refund | 7854.70 | rejected
TXN-1201 | 2025-05-24 | Saltmarsh Catering | invoice | 2954.85 | pending
TXN-1202 | 2025-04-22 | Quillon Software | invoice | 658.78 | approved
TXN-1203 | 2025-05-10 | Saltmarsh Catering | adjustment | 7204.19 | approved
TXN-1204 | 2025-04-05 | Saltmarsh Catering | refund | 4543.62 | approved
TXN-1205 | 2025-01-24 | Pemberton Freight | invoice | 5669.51 | approved
TXN-1206 | 2025-01-29 | Halloway Instruments | refund | 8590.77 | pending
TXN-1207 | 2025-04-06 | Cortez Logistics | adjustment | 705.79 | rejected
TXN-1208 | 2025-05-15 | Halloway Instruments | invoice | 525.14 | rejected
TXN-1209 | 2025-05-29 | Pemberton Freight | refund | 8866.64 | pending
TXN-1210 | 2025-01-08 | Saltmarsh Catering | adjustment | 3914.09 | pending
TXN-1211 | 2025-04-21 | Cortez Logistics | invoice | 2999.68 | approved
TXN-1212 | 2025-05-15 | Halloway Instruments | refund | 4110.14 | approved
TXN-1213 | 2025-01-05 | Cortez Logistics | refund | 1918.13 | pending
TXN-1214 | 2025-05-05 | Quillon Software | refund | 328.95 | approved
TXN-1215 | 2025-02-15 | Saltmarsh Catering | adjustment | 7800.90 | rejected
TXN-1216 | 2025-04-11 | Halloway Instruments | adjustment | 2368.13 | pending
TXN-1217 | 2025-04-06 | Rivet & Sons | refund | 7422.52 | pending
TXN-1218 | 2025-03-21 | Saltmarsh Catering | invoice | 3255.99 | pending
TXN-1219 | 2025-01-12 | Northwind Supplies | invoice | 2571.32 | approved
TXN-1220 | 2025-02-02 | Cortez Logistics | refund | 9467.87 | rejected
TXN-1221 | 2025-01-06 | Halloway Instruments | refund | 502.85 | pending
TXN-1222 | 2025-02-24 | Halloway Instruments | invoice | 1406.28 | approved
TXN-1223 | 2025-02-09 | Quillon Software | invoice | 4735.79 | approved
TXN-1224 | 2025-05-26 | Pemberton Freight | adjustment | 8253.59 | approved
TXN-1225 | 2025-02-08 | Rivet & Sons | invoice | 1669.67 | approved
TXN-1226 | 2025-03-22 | Pemberton Freight | invoice | 6885.69 | rejected
TXN-1227 | 2025-01-18 | Cortez Logistics | invoice | 3795.50 | pending
TXN-1228 | 2025-04-26 | Saltmarsh Catering | refund | 5256.35 | pending
TXN-1229 | 2025-04-27 | Pemberton Freight | adjustment | 5278.00 | pending
TXN-1230 | 2025-01-13 | Ambleside Print | invoice | 2794.51 | pending
TXN-1231 | 2025-03-20 | Quillon Software | refund | 2520.32 | pending
TXN-1232 | 2025-06-07 | Northwind Supplies | invoice | 782.86 | approved
TXN-1233 | 2025-05-11 | Pemberton Freight | invoice | 7094.52 | pending
TXN-1234 | 2025-04-15 | Quillon Software | refund | 3446.26 | pending
TXN-1235 | 2025-02-10 | Pemberton Freight | adjustment | 3447.50 | pending
TXN-1236 | 2025-06-15 | Rivet & Sons | invoice | 3456.78 | rejected
TXN-1237 | 2025-01-19 | Northwind Supplies | refund | 3044.97 | rejected
TXN-1238 | 2025-04-21 | Quillon Software | refund | 1954.14 | approved
TXN-1239 | 2025-01-21 | Halloway Instruments | refund | 316.81 | pending
TXN-1240 | 2025-06-18 | Ambleside Print | invoice | 1960.70 | approved
TXN-1241 | 2025-06-25 | Rivet & Sons | adjustment | 474.11 | pending
TXN-1242 | 2025-04-09 | Ambleside Print | invoice | 6997.12 | rejected
TXN-1243 | 2025-06-26 | Ambleside Print | invoice | 6247.04 | approved
TXN-1244 | 2025-06-15 | Cortez Logistics | refund | 516.29 | pending
TXN-1245 | 2025-02-09 | Northwind Supplies | adjustment | 3109.55 | approved
TXN-1246 | 2025-01-14 | Northwind Supplies | adjustment | 3870.38 | rejected
TXN-1247 | 2025-05-18 | Northwind Supplies | invoice | 127.99 | rejected
TXN-1248 | 2025-04-08 | Quillon Software | invoice | 381.51 | approved
TXN-1249 | 2025-01-09 | Rivet & Sons | adjustment | 2562.59 | rejected
TXN-1250 | 2025-04-25 | Halloway Instruments | refund | 6727.73 | approved
TXN-1251 | 2025-02-06 | Pemberton Freight | refund | 4340.50 | approved
TXN-1252 | 2025-01-30 | Rivet & Sons | invoice | 541.29 | rejected
TXN-1253 | 2025-01-08 | Rivet & Sons | adjustment | 887.88 | pending
TXN-1254 | 2025-06-20 | Quillon Software | invoice | 4648.39 | approved
TXN-1255 | 2025-04-03 | Halloway Instruments | refund | 5290.11 | approved
TXN-1256 | 2025-04-20 | Pemberton Freight | refund | 5051.81 | pending
TXN-1257 | 2025-06-24 | Cortez Logistics | invoice | 6400.99 | pending
TXN-1258 | 2025-03-27 | Halloway Instruments | adjustment | 3283.32 | pending
TXN-1259 | 2025-06-05 | Ambleside Print | adjustment | 2148.66 | pending
TXN-1260 | 2025-05-03 | Saltmarsh Catering | invoice | 8585.68 | pending
TXN-1261 | 2025-03-13 | Northwind Supplies | invoice | 3533.84 | pending
TXN-1262 | 2025-04-04 | Ambleside Print | adjustment | 5148.27 | rejected
TXN-1263 | 2025-02-16 | Ambleside Print | invoice | 2030.42 | approved
TXN-1264 | 2025-06-26 | Pemberton Freight | refund | 7686.80 | pending
TXN-1265 | 2025-03-29 | Northwind Supplies | adjustment | 2959.89 | pending
TXN-1266 | 2025-04-09 | Ambleside Print | refund | 9550.87 | approved
TXN-1267 | 2025-02-22 | Pemberton Freight | invoice | 8313.71 | rejected
TXN-1268 | 2025-03-18 | Ambleside Print | adjustment | 2709.52 | rejected
TXN-1269 | 2025-01-29 | Quillon Software | adjustment | 5252.12 | approved
TXN-1270 | 2025-05-24 | Cortez Logistics | invoice | 4936.23 | pending
TXN-1271 | 2025-03-25 | Ambleside Print | invoice | 8949.21 | rejected
TXN-1272 | 2025-04-17 | Rivet & Sons | invoice | 6836.15 | approved
TXN-1273 | 2025-05-14 | Northwind Supplies | refund | 3417.45 | approved
TXN-1274 | 2025-05-08 | Ambleside Print | invoice | 3093.35 | rejected
TXN-1275 | 2025-04-29 | Quillon Software | refund | 8506.65 | pending
TXN-1276 | 2025-04-21 | Northwind Supplies | refund | 5519.69 | approved
TXN-1277 | 2025-02-03 | Halloway Instruments | adjustment | 3553.08 | pending
TXN-1278 | 2025-02-13 | Pemberton Freight | refund | 898.64 | pending
TXN-1279 | 2025-06-16 | Rivet & Sons | adjustment | 2971.70 | rejected
TXN-1280 | 2025-03-24 | Halloway Instruments | adjustment | 5040.68 | pending
TXN-1281 | 2025-03-05 | Rivet & Sons | adjustment | 4500.72 | rejected
TXN-1282 | 2025-04-11 | Halloway Instruments | invoice | 7942.19 | pending
TXN-1283 | 2025-01-05 | Quillon Software | adjustment | 116.77 | pending
TXN-1284 | 2025-06-25 | Cortez Logistics | invoice | 299.61 | approved
TXN-1285 | 2025-05-03 | Northwind Supplies | adjustment | 7229.40 | approved
TXN-1286 | 2025-06-19 | Ambleside Print | invoice | 3686.15 | approved
TXN-1287 | 2025-01-24 | Quillon Software | invoice | 2694.43 | rejected
TXN-1288 | 2025-03-12 | Quillon Software | refund | 4875.93 | rejected
TXN-1289 | 2025-04-26 | Pemberton Freight | invoice | 109.63 | pending
TXN-1290 | 2025-05-29 | Ambleside Print | adjustment | 404.98 | pending
TXN-1291 | 2025-01-23 | Northwind Supplies | adjustment | 2764.03 | approved
TXN-1292 | 2025-06-04 | Cortez Logistics | adjustment | 5331.36 | approved
TXN-1293 | 2025-03-30 | Northwind Supplies | invoice | 2031.27 | rejected
TXN-1294 | 2025-03-19 | Northwind Supplies | invoice | 4546.84 | approved
TXN-1295 | 2025-06-04 | Rivet & Sons | adjustment | 4912.24 | rejected
TXN-1296 | 2025-02-13 | Rivet & Sons | refund | 1236.14 | rejected
TXN-1297 | 2025-05-24 | Quillon Software | adjustment | 7517.59 | rejected
TXN-1298 | 2025-03-24 | Quillon Software | adjustment | 3804.19 | rejected
TXN-1299 | 2025-01-21 | Ambleside Print | invoice | 4702.00 | rejected
TXN-1300 | 2025-04-10 | Rivet & Sons | invoice | 5586.94 | rejected
TXN-1301 | 2025-05-08 | Pemberton Freight | refund | 1268.15 | rejected
TXN-1302 | 2025-06-25 | Rivet & Sons | refund | 5024.60 | pending
TXN-1303 | 2025-01-02 | Quillon Software | adjustment | 2214.83 | rejected
TXN-1304 | 2025-03-06 | Rivet & Sons | refund | 7027.76 | approved
TXN-1305 | 2025-05-31 | Halloway Instruments | invoice | 2069.20 | approved
TXN-1306 | 2025-04-25 | Ambleside Print | invoice | 1132.10 | approved
TXN-1307 | 2025-06-26 | Rivet & Sons | adjustment | 4753.75 | pending
TXN-1308 | 2025-02-12 | Northwind Supplies | invoice | 665.00 | rejected
TXN-1309 | 2025-02-22 | Halloway Instruments | adjustment | 753.21 | approved
TXN-1310 | 2025-05-11 | Pemberton Freight | refund | 4137.71 | rejected
TXN-1311 | 2025-05-24 | Rivet & Sons | refund | 1611.95 | approved
TXN-1312 | 2025-03-13 | Rivet & Sons | adjustment | 4579.85 | rejected
TXN-1313 | 2025-06-29 | Halloway Instruments | adjustment | 7112.18 | pending
TXN-1314 | 2025-06-02 | Quillon Software | adjustment | 9304.54 | pending
TXN-1315 | 2025-02-22 | Rivet & Sons | adjustment | 1807.47 | rejected
TXN-1316 | 2025-04-01 | Cortez Logistics | invoice | 1978.92 | pending
TXN-1317 | 2025-05-25 | Saltmarsh Catering | adjustment | 5334.30 | pending
TXN-1318 | 2025-01-23 | Rivet & Sons | invoice | 4898.14 | rejected
TXN-1319 | 2025-02-10 | Pemberton Freight | refund | 5293.29 | pending
TXN-1320 | 2025-06-08 | Cortez Logistics | invoice | 5816.40 | approved
TXN-1321 | 2025-06-19 | Cortez Logistics | refund | 4512.30 | approved
TXN-1322 | 2025-03-16 | Rivet & Sons | adjustment | 6987.94 | rejected
TXN-1323 | 2025-01-15 | Quillon Software | refund | 8847.30 | approved
TXN-1324 | 2025-02-04 | Cortez Logistics | refund | 9517.10 | rejected
TXN-1325 | 2025-04-06 | Quillon Software | adjustment | 352.95 | approved
TXN-1326 | 2025-02-01 | Quillon Software | refund | 1176.99 | approved
TXN-1327 | 2025-03-19 | Quillon Software | adjustment | 4858.69 | approved
TXN-1328 | 2025-01-19 | Quillon Software | refund | 5280.80 | approved
TXN-1329 | 2025-03-22 | Quillon Software | refund | 5759.02 | pending
TXN-1330 | 2025-02-19 | Ambleside Print | adjustment | 1426.53 | approved
TXN-1331 | 2025-02-13 | Halloway Instruments | invoice | 1324.78 | pending
TXN-1332 | 2025-06-27 | Halloway Instruments | refund | 1405.68 | approved
TXN-1333 | 2025-03-17 | Halloway Instruments | adjustment | 238.47 | pending
TXN-1334 | 2025-03-13 | Rivet & Sons | refund | 2537.98 | approved
TXN-1335 | 2025-01-07 | Rivet & Sons | adjustment | 4812.69 | rejected
TXN-1336 | 2025-04-20 | Cortez Logistics | adjustment | 850.88 | approved
TXN-1337 | 2025-05-07 | Halloway Instruments | invoice | 9466.72 | pending
TXN-1338 | 2025-02-27 | Quillon Software | invoice | 9743.35 | pending
TXN-1339 | 2025-01-27 | Ambleside Print | invoice | 4350.83 | rejected
TXN-1340 | 2025-02-07 | Cortez Logistics | invoice | 1490.61 | rejected
TXN-1341 | 2025-04-14 | Pemberton Freight | adjustment | 8239.90 | pending
TXN-1342 | 2025-06-04 | Quillon Software | refund | 6861.43 | rejected
TXN-1343 | 2025-06-06 | Saltmarsh Catering | invoice | 9344.65 | approved
TXN-1344 | 2025-02-11 | Cortez Logistics | invoice | 2049.45 | rejected
TXN-1345 | 2025-05-28 | Saltmarsh Catering | adjustment | 7913.59 | approved
TXN-1346 | 2025-06-01 | Quillon Software | adjustment | 5364.38 | rejected
TXN-1347 | 2025-02-19 | Rivet & Sons | invoice | 3879.85 | pending
TXN-1348 | 2025-06-05 | Pemberton Freight | adjustment | 9481.77 | rejected
TXN-1349 | 2025-06-04 | Quillon Software | adjustment | 8544.85 | rejected
TXN-1350 | 2025-05-03 | Halloway Instruments | adjustment | 1011.67 | rejected
TXN-1351 | 2025-05-28 | Quillon Software | invoice | 1522.76 | approved
TXN-1352 | 2025-04-25 | Cortez Logistics | adjustment | 1703.31 | pending
TXN-1353 | 2025-04-03 | Quillon Software | adjustment | 2562.45 | rejected
TXN-1354 | 2025-01-13 | Halloway Instruments | refund | 3449.68 | pending
TXN-1355 | 2025-04-01 | Pemberton Freight | refund | 4225.58 | rejected
TXN-1356 | 2025-02-04 | Pemberton Freight | adjustment | 432.30 | rejected
TXN-1357 | 2025-04-02 | Halloway Instruments | refund | 4634.47 | rejected
TXN-1358 | 2025-05-13 | Rivet & Sons | invoice | 1006.10 | rejected
TXN-1359 | 2025-03-13 | Pemberton Freight | invoice | 5012.57 | rejected
TXN-1360 | 2025-04-28 | Cortez Logistics | invoice | 6022.05 | rejected
TXN-1361 | 2025-01-18 | Quillon Software | refund | 9448.61 | rejected
TXN-1362 | 2025-03-19 | Northwind Supplies | refund | 1507.18 | approved
TXN-1363 | 2025-03-01 | Halloway Instruments | invoice | 1591.18 | pending
TXN-1364 | 2025-02-04 | Rivet & Sons | adjustment | 6755.11 | pending
TXN-1365 | 2025-05-12 | Quillon Software | refund | 5301.84 | rejected
TXN-1366 | 2025-03-06 | Cortez Logistics | invoice | 9779.20 | rejected
TXN-1367 | 2025-06-17 | Quillon Software | invoice | 3632.03 | rejected
TXN-1368 | 2025-06-12 | Pemberton Freight | refund | 1649.36 | approved
TXN-1369 | 2025-01-18 | Saltmarsh Catering | refund | 2086.44 | approved
TXN-1370 | 2025-02-12 | Saltmarsh Catering | refund | 2012.55 | pending
TXN-1371 | 2025-01-12 | Saltmarsh Catering | invoice | 2770.06 | approved
TXN-1372 | 2025-05-26 | Northwind Supplies | invoice | 8112.64 | approved
TXN-1373 | 2025-06-16 | Cortez Logistics | adjustment | 4619.69 | approved
TXN-1374 | 2025-01-22 | Ambleside Print | adjustment | 8898.38 | rejected
TXN-1375 | 2025-04-27 | Pemberton Freight | refund | 8779.51 | rejected
TXN-1376 | 2025-02-26 | Quillon Software | adjustment | 3643.78 | approved
TXN-1377 | 2025-01-10 | Northwind Supplies | refund | 2845.23 | rejected
TXN-1378 | 2025-02-22 | Halloway Instruments | refund | 1790.09 | pending
TXN-1379 | 2025-03-31 | Northwind Supplies | refund | 6611.25 | approved
TXN-1380 | 2025-01-31 | Halloway Instruments | adjustment | 4578.29 | rejected
TXN-1381 | 2025-03-29 | Quillon Software | adjustment | 5095.68 | pending
TXN-1382 | 2025-06-24 | Rivet & Sons | invoice | 3600.11 | rejected
TXN-1383 | 2025-02-02 | Saltmarsh Catering | adjustment | 5988.20 | pending
TXN-1384 | 2025-04-13 | Rivet & Sons | invoice | 6813.23 | rejected
TXN-1385 | 2025-02-16 | Pemberton Freight | invoice | 5690.52 | approved
TXN-1386 | 2025-04-10 | Rivet & Sons | refund | 6389.33 | rejected
TXN-1387 | 2025-06-26 | Rivet & Sons | refund | 487.49 | pending
TXN-1388 | 2025-01-10 | Rivet & Sons | invoice | 2451.12 | pending
TXN-1389 | 2025-03-17 | Quillon Software | invoice | 6598.60 | approved
TXN-1390 | 2025-01-01 | Saltmarsh Catering | adjustment | 6915.69 | approved
TXN-1391 | 2025-02-02 | Pemberton Freight | refund | 1762.92 | approved
TXN-1392 | 2025-04-04 | Northwind Supplies | invoice | 65.85 | approved
TXN-1393 | 2025-02-11 | Halloway Instruments | invoice | 3447.18 | pending
TXN-1394 | 2025-06-20 | Cortez Logistics | invoice | 2126.33 | pending
TXN-1395 | 2025-02-09 | Rivet & Sons | refund | 3652.69 | rejected
TXN-1396 | 2025-05-19 | Halloway Instruments | adjustment | 2969.04 | rejected
TXN-1397 | 2025-06-24 | Ambleside Print | refund | 5215.84 | pending
TXN-1398 | 2025-03-14 | Rivet & Sons | adjustment | 2546.71 | approved
TXN-1399 | 2025-02-15 | Northwind Supplies | adjustment | 8926.42 | pending

## Repeat 1 (seed 0)

- Score: 10.0/10 — All five ledger aggregates match the ground truth and output is a bare JSON object with correct keys and no extra text.
- Tokens/s: 111.5
- Prefill tok/s: 1610.6
- TTFT: 107.74s
- Gen Time: 108.33s
- Output Tokens: 11011
- answer read from `answer_tag`

<details><summary>Reasoning (19119 chars, not scored)</summary>

Let me work through each question carefully.

**Question 1: Total amount of all rows that are vendor "Halloway Instruments", kind "invoice", and status "approved"**

Let me go through the ledger and find all rows matching:
- Vendor: Halloway Instruments
- Kind: invoice
- Status: approved

TXN-1029 | 2025-01-13 | Halloway Instruments | invoice | 4230.28 | approved ✓
TXN-1052 | 2025-06-28 | Halloway Instruments | invoice | 9564.00 | pending ✗
TXN-1059 | 2025-02-08 | Halloway Instruments | invoice | 2134.49 | pending ✗
TXN-1115 | 2025-04-16 | Halloway Instruments | invoice | 2416.43 | rejected ✗
TXN-1126 | 2025-06-30 | Halloway Instruments | invoice | 1267.81 | approved ✓
TXN-1139 | 2025-01-20 | Halloway Instruments | invoice | 4330.47 | approved ✓
TXN-1169 | 2025-03-19 | Halloway Instruments | invoice | 3888.03 | pending ✗
TXN-1199 | 2025-05-11 | Halloway Instruments | invoice | 2122.83 | pending ✗
TXN-1208 | 2025-05-15 | Halloway Instruments | invoice | 525.14 | rejected ✗
TXN-1222 | 2025-02-24 | Halloway Instruments | invoice | 1406.28 | approved ✓
TXN-1282 | 2025-04-11 | Halloway Instruments | invoice | 7942.19 | pending ✗
TXN-1305 | 2025-05-31 | Halloway Instruments | invoice | 2069.20 | approved ✓
TXN-1331 | 2025-02-13 | Halloway Instruments | invoice | 1324.78 | pending ✗
TXN-1337 | 2025-05-07 | Halloway Instruments | invoice | 9466.72 | pending ✗
TXN-1363 | 2025-03-19 | Halloway Instruments | invoice | 1591.18 | pending ✗
TXN-1393 | 2025-02-11 | Halloway Instruments | invoice | 3447.18 | pending ✗

So the approved invoices for Halloway Instruments:
- TXN-1029: 4230.28
- TXN-1126: 1267.81
- TXN-1139: 4330.47
- TXN-1222: 1406.28
- TXN-1305: 2069.20

Total: 4230.28 + 1267.81 + 4330.47 + 1406.28 + 2069.20 = 13304.04

Let me double-check:
4230.28 + 1267.81 = 5498.09
5498.09 + 4330.47 = 9828.56
9828.56 + 1406.28 = 11234.84
11234.84 + 2069.20 = 13304.04

**Question 2: How many rows have status "rejected"?**

Let me count all rejected rows:

TXN-1002 | rejected
TXN-1003 | rejected
TXN-1004 | rejected
TXN-1007 | rejected
TXN-1008 | rejected
TXN-1016 | rejected
TXN-1020 | rejected
TXN-1028 | rejected
TXN-1031 | rejected
TXN-1032 | rejected
TXN-1034 | rejected
TXN-1035 | rejected
TXN-1037 | rejected
TXN-1039 | rejected
TXN-1043 | rejected
TXN-1044 | rejected
TXN-1046 | rejected
TXN-1050 | rejected
TXN-1053 | rejected
TXN-1056 | rejected
TXN-1057 | rejected
TXN-1058 | rejected
TXN-1063 | rejected
TXN-1066 | rejected
TXN-1067 | rejected
TXN-1071 | rejected
TXN-1074 | rejected
TXN-1076 | rejected
TXN-1078 | rejected
TXN-1081 | rejected
TXN-1082 | rejected
TXN-1087 | rejected
TXN-1091 | rejected
TXN-1094 | rejected
TXN-1097 | rejected
TXN-1100 | rejected
TXN-1103 | rejected
TXN-1109 | rejected
TXN-1114 | rejected
TXN-1115 | rejected
TXN-1118 | rejected
TXN-1119 | rejected
TXN-1120 | rejected
TXN-1125 | rejected
TXN-1130 | rejected
TXN-1132 | rejected
TXN-1134 | rejected
TXN-1135 | rejected
TXN-1136 | rejected
TXN-1140 | rejected
TXN-1144 | rejected
TXN-1145 | rejected
TXN-1148 | rejected
TXN-1151 | rejected
TXN-1152 | rejected
TXN-1157 | rejected
TXN-1158 | rejected
TXN-1159 | rejected
TXN-1167 | rejected
TXN-1170 | rejected
TXN-1171 | rejected
TXN-1176 | rejected
TXN-1177 | rejected
TXN-1179 | rejected
TXN-1182 | rejected
TXN-1185 | rejected
TXN-1186 | rejected
TXN-1188 | rejected
TXN-1190 | rejected
TXN-1193 | rejected
TXN-1196 | rejected
TXN-1197 | rejected
TXN-1200 | rejected
TXN-1207 | rejected
TXN-1208 | rejected
TXN-1215 | rejected
TXN-1220 | rejected
TXN-1226 | rejected
TXN-1236 | rejected
TXN-1237 | rejected
TXN-1242 | rejected
TXN-1246 | rejected
TXN-1247 | rejected
TXN-1249 | rejected
TXN-1252 | rejected
TXN-1262 | rejected
TXN-1267 | rejected
TXN-1268 | rejected
TXN-1271 | rejected
TXN-1274 | rejected
TXN-1279 | rejected
TXN-1281 | rejected
TXN-1287 | rejected
TXN-1288 | rejected
TXN-1293 | rejected
TXN-1295 | rejected
TXN-1296 | rejected
TXN-1297 | rejected
TXN-1298 | rejected
TXN-1299 | rejected
TXN-1300 | rejected
TXN-1301 | rejected
TXN-1303 | rejected
TXN-1308 | rejected
TXN-1310 | rejected
TXN-1312 | rejected
TXN-1315 | rejected
TXN-1318 | rejected
TXN-1322 | rejected
TXN-1324 | rejected
TXN-1335 | rejected
TXN-1339 | rejected
TXN-1340 | rejected
TXN-1342 | rejected
TXN-1344 | rejected
TXN-1346 | rejected
TXN-1348 | rejected
TXN-1349 | rejected
TXN-1350 | rejected
TXN-1353 | rejected
TXN-1355 | rejected
TXN-1356 | rejected
TXN-1357 | rejected
TXN-1358 | rejected
TXN-1359 | rejected
TXN-1360 | rejected
TXN-1361 | rejected
TXN-1365 | rejected
TXN-1366 | rejected
TXN-1367 | rejected
TXN-1374 | rejected
TXN-1375 | rejected
TXN-1377 | rejected
TXN-1380 | rejected
TXN-1382 | rejected
TXN-1384 | rejected
TXN-1386 | rejected
TXN-1395 | rejected
TXN-1396 | rejected

Let me count these more carefully by going through the list:

1. TXN-1002
2. TXN-1003
3. TXN-1004
4. TXN-1007
5. TXN-1008
6. TXN-1016
7. TXN-1020
8. TXN-1028
9. TXN-1031
10. TXN-1032
11. TXN-1034
12. TXN-1035
13. TXN-1037
14. TXN-1039
15. TXN-1043
16. TXN-1044
17. TXN-1046
18. TXN-1050
19. TXN-1053
20. TXN-1056
21. TXN-1057
22. TXN-1058
23. TXN-1063
24. TXN-1066
25. TXN-1067
26. TXN-1071
27. TXN-1074
28. TXN-1076
29. TXN-1078
30. TXN-1081
31. TXN-1082
32. TXN-1087
33. TXN-1091
34. TXN-1094
35. TXN-1097
36. TXN-1100
37. TXN-1103
38. TXN-1109
39. TXN-1114
40. TXN-1115
41. TXN-1118
42. TXN-1119
43. TXN-1120
44. TXN-1125
45. TXN-1130
46. TXN-1132
47. TXN-1134
48. TXN-1135
49. TXN-1136
50. TXN-1140
51. TXN-1144
52. TXN-1145
53. TXN-1148
54. TXN-1151
55. TXN-1152
56. TXN-1157
57. TXN-1158
58. TXN-1159
59. TXN-1167
60. TXN-1170
61. TXN-1171
62. TXN-1176
63. TXN-1177
64. TXN-1179
65. TXN-1182
66. TXN-1185
67. TXN-1186
68. TXN-1188
69. TXN-1190
70. TXN-1193
71. TXN-1196
72. TXN-1197
73. TXN-1200
74. TXN-1207
75. TXN-1208
76. TXN-1215
77. TXN-1220
78. TXN-1226
79. TXN-1236
80. TXN-1237
81. TXN-1242
82. TXN-1246
83. TXN-1247
84. TXN-1249
85. TXN-1252
86. TXN-1262
87. TXN-1267
88. TXN-1268
89. TXN-1271
90. TXN-1274
91. TXN-1279
92. TXN-1281
93. TXN-1287
94. TXN-1288
95. TXN-1293
96. TXN-1295
97. TXN-1296
98. TXN-1297
99. TXN-1298
100. TXN-1299
101. TXN-1300
102. TXN-1301
103. TXN-1303
104. TXN-1308
105. TXN-1310
106. TXN-1312
107. TXN-1315
108. TXN-1318
109. TXN-1322
110. TXN-1324
111. TXN-1335
112. TXN-1339
113. TXN-1340
114. TXN-1342
115. TXN-1344
116. TXN-1346
117. TXN-1348
118. TXN-1349
119. TXN-1350
120. TXN-1353
121. TXN-1355
122. TXN-1356
123. TXN-1357
124. TXN-1358
125. TXN-1359
126. TXN-1360
127. TXN-1361
128. TXN-1365
129. TXN-1366
130. TXN-1367
131. TXN-1374
132. TXN-1375
133. TXN-1377
134. TXN-1380
135. TXN-1382
136. TXN-1384
137. TXN-1386
138. TXN-1395
139. TXN-1396

That's 139 rejected rows.

Wait, let me re-check. Let me go through the ledger more carefully, row by row, and mark rejected ones.

TXN-1000: approved
TXN-1001: approved
TXN-1002: rejected (1)
TXN-1003: rejected (2)
TXN-1004: rejected (3)
TXN-1005: pending
TXN-1006: approved
TXN-1007: rejected (4)
TXN-1008: rejected (5)
TXN-1009: approved
TXN-1010: approved
TXN-1011: pending
TXN-1012: approved
TXN-1013: approved
TXN-1014: pending
TXN-1015: pending
TXN-1016: rejected (6)
TXN-1017: approved
TXN-1018: pending
TXN-1019: approved
TXN-1020: rejected (7)
TXN-1021: approved
TXN-1022: approved
TXN-1023: pending
TXN-1024: approved
TXN-1025: approved
TXN-1026: pending
TXN-1027: approved
TXN-1028: rejected (8)
TXN-1029: approved
TXN-1030: pending
TXN-1031: rejected (9)
TXN-1032: rejected (10)
TXN-1033: pending
TXN-1034: rejected (11)
TXN-1035: rejected (12)
TXN-1036: approved
TXN-1037: rejected (13)
TXN-1038: pending
TXN-1039: rejected (14)
TXN-1040: pending
TXN-1041: approved
TXN-1042: approved
TXN-1043: rejected (15)
TXN-1044: rejected (16)
TXN-1045: pending
TXN-1046: rejected (17)
TXN-1047: approved
TXN-1048: approved
TXN-1049: approved
TXN-1050: rejected (18)
TXN-1051: approved
TXN-1052: pending
TXN-1053: rejected (19)
TXN-1054: approved
TXN-1055: pending
TXN-1056: rejected (20)
TXN-1057: rejected (21)
TXN-1058: rejected (22)
TXN-1059: pending
TXN-1060: pending
TXN-1061: approved
TXN-1062: pending
TXN-1063: rejected (23)
TXN-1064: approved
TXN-1065: pending
TXN-1066: rejected (24)
TXN-1067: rejected (25)
TXN-1068: approved
TXN-1069: pending
TXN-1070: pending
TXN-1071: rejected (26)
TXN-1072: approved
TXN-1073: pending
TXN-1074: rejected (27)
TXN-1075: pending
TXN-1076: rejected (28)
TXN-1077: pending
TXN-1078: rejected (29)
TXN-1079: pending
TXN-1080: approved
TXN-1081: rejected (30)
TXN-1082: rejected (31)
TXN-1083: pending
TXN-1084: pending
TXN-1085: pending
TXN-1086: pending
TXN-1087: rejected (32)
TXN-1088: pending
TXN-1089: approved
TXN-1090: approved
TXN-1091: rejected (33)
TXN-1092: pending
TXN-1093: approved
TXN-1094: rejected (34)
TXN-1095: approved
TXN-1096: pending
TXN-1097: rejected (35)
TXN-1098: pending
TXN-1099: pending
TXN-1100: rejected (36)
TXN-1101: pending
TXN-1102: pending
TXN-1103: rejected (37)
TXN-1104: approved
TXN-1105: approved
TXN-1106: pending
TXN-1107: approved
TXN-1108: pending
TXN-1109: rejected (38)
TXN-1110: pending
TXN-1111: approved
TXN-1112: pending
TXN-1113: approved
TXN-1114: rejected (39)
TXN-1115: rejected (40)
TXN-1116: approved
TXN-1117: approved
TXN-1118: rejected (41)
TXN-1119: rejected (42)
TXN-1120: rejected (43)
TXN-1121: pending
TXN-1122: pending
TXN-1123: approved
TXN-1124: approved
TXN-1125: rejected (44)
TXN-1126: approved
TXN-1127: pending
TXN-1128: pending
TXN-1129: pending
TXN-1130: rejected (45)
TXN-1131: pending
TXN-1132: rejected (46)
TXN-1133: approved
TXN-1134: rejected (47)
TXN-1135: rejected (48)
TXN-1136: rejected (49)
TXN-1137: pending
TXN-1138: approved
TXN-1139: approved
TXN-1140: rejected (50)
TXN-1141: pending
TXN-1142: approved
TXN-1143: pending
TXN-1144: rejected (51)
TXN-1145: rejected (52)
TXN-1146: approved
TXN-1147: pending
TXN-1148: rejected (53)
TXN-1149: pending
TXN-1150: pending
TXN-1151: rejected (54)
TXN-1152: rejected (55)
TXN-1153: pending
TXN-1154: pending
TXN-1155: approved
TXN-1156: approved
TXN-1157: rejected (56)
TXN-1158: rejected (57)
TXN-1159: rejected (58)
TXN-1160: pending
TXN-1161: approved
TXN-1162: pending
TXN-1163: approved
TXN-1164: pending
TXN-1165: approved
TXN-1166: approved
TXN-1167: rejected (59)
TXN-1168: approved
TXN-1169: pending
TXN-1170: rejected (60)
TXN-1171: rejected (61)
TXN-1172: pending
TXN-1173: pending
TXN-1174: approved
TXN-1175: pending
TXN-1176: rejected (62)
TXN-1177: rejected (63)
TXN-1178: approved
TXN-1179: rejected (64)
TXN-1180: approved
TXN-1181: pending
TXN-1182: rejected (65)
TXN-1183: pending
TXN-1184: pending
TXN-1185: rejected (66)
TXN-1186: rejected (67)
TXN-1187: pending
TXN-1188: rejected (68)
TXN-1189: approved
TXN-1190: rejected (69)
TXN-1191: pending
TXN-1192: pending
TXN-1193: rejected (70)
TXN-1194: approved
TXN-1195: approved
TXN-1196: rejected (71)
TXN-1197: rejected (72)
TXN-1198: pending
TXN-1199: pending
TXN-1200: rejected (73)
TXN-1201: pending
TXN-1202: approved
TXN-1203: approved
TXN-1204: approved
TXN-1205: approved
TXN-1206: pending
TXN-1207: rejected (74)
TXN-1208: rejected (75)
TXN-1209: pending
TXN-1210: pending
TXN-1211: approved
TXN-1212: approved
TXN-1213: pending
TXN-1214: approved
TXN-1215: rejected (76)
TXN-1216: pending
TXN-1217: pending
TXN-1218: pending
TXN-1219: approved
TXN-1220: rejected (77)
TXN-1221: pending
TXN-1222: approved
TXN-1223: approved
TXN-1224: approved
TXN-1225: approved
TXN-1226: rejected (78)
TXN-1227: pending
TXN-1228: pending
TXN-1229: pending
TXN-1230: pending
TXN-1231: pending
TXN-1232: approved
TXN-1233: pending
TXN-1234: pending
TXN-1235: pending
TXN-1236: rejected (79)
TXN-1237: rejected (80)
TXN-1238: approved
TXN-1239: pending
TXN-1240: approved
TXN-1241: pending
TXN-1242: rejected (81)
TXN-1243: approved
TXN-1244: pending
TXN-1245: approved
TXN-1246: rejected (82)
TXN-1247: rejected (83)
TXN-1248: approved
TXN-1249: rejected (84)
TXN-1250: approved
TXN-1251: approved
TXN-1252: rejected (85)
TXN-1253: pending
TXN-1254: approved
TXN-1255: approved
TXN-1256: pending
TXN-1257: pending
TXN-1258: pending
TXN-1259: pending
TXN-1260: pending
TXN-1261: pending
TXN-1262: rejected (86)
TXN-1263: approved
TXN-1264: pending
TXN-1265: pending
TXN-1266: approved
TXN-1267: rejected (87)
TXN-1268: rejected (88)
TXN-1269: approved
TXN-1270: pending
TXN-1271: rejected (89)
TXN-1272: approved
TXN-1273: approved
TXN-1274: rejected (90)
TXN-1275: pending
TXN-1276: approved
TXN-1277: pending
TXN-1278: pending
TXN-1279: rejected (91)
TXN-1280: pending
TXN-1281: rejected (92)
TXN-1282: pending
TXN-1283: pending
TXN-1284: approved
TXN-1285: approved
TXN-1286: approved
TXN-1287: rejected (93)
TXN-1288: rejected (94)
TXN-1289: pending
TXN-1290: pending
TXN-1291: approved
TXN-1292: approved
TXN-1293: rejected (95)
TXN-1294: approved
TXN-1295: rejected (96)
TXN-1296: rejected (97)
TXN-1297: rejected (98)
TXN-1298: rejected (99)
TXN-1299: rejected (100)
TXN-1300: rejected (101)
TXN-1301: rejected (102)
TXN-1302: pending
TXN-1303: rejected (103)
TXN-1304: approved
TXN-1305: approved
TXN-1306: approved
TXN-1307: pending
TXN-1308: rejected (104)
TXN-1309: approved
TXN-1310: rejected (105)
TXN-1311: approved
TXN-1312: rejected (106)
TXN-1313: pending
TXN-1314: pending
TXN-1315: rejected (107)
TXN-1316: pending
TXN-1317: pending
TXN-1318: rejected (108)
TXN-1319: pending
TXN-1320: approved
TXN-1321: approved
TXN-1322: rejected (109)
TXN-1323: approved
TXN-1324: rejected (110)
TXN-1325: approved
TXN-1326: approved
TXN-1327: approved
TXN-1328: approved
TXN-1329: pending
TXN-1330: approved
TXN-1331: pending
TXN-1332: approved
TXN-1333: pending
TXN-1334: approved
TXN-1335: rejected (111)
TXN-1336: approved
TXN-1337: pending
TXN-1338: pending
TXN-1339: rejected (112)
TXN-1340: rejected (113)
TXN-1341: pending
TXN-1342: rejected (114)
TXN-1343: approved
TXN-1344: rejected (115)
TXN-1345: approved
TXN-1346: rejected (116)
TXN-1347: pending
TXN-1348: rejected (117)
TXN-1349: rejected (118)
TXN-1350: rejected (119)
TXN-1351: approved
TXN-1352: pending
TXN-1353: rejected (120)
TXN-1354: pending
TXN-1355: rejected (121)
TXN-1356: rejected (122)
TXN-1357: rejected (123)
TXN-1358: rejected (124)
TXN-1359: rejected (125)
TXN-1360: rejected (126)
TXN-1361: rejected (127)
TXN-1362: approved
TXN-1363: pending
TXN-1364: pending
TXN-1365: rejected (128)
TXN-1366: rejected (129)
TXN-1367: rejected (130)
TXN-1368: approved
TXN-1369: approved
TXN-1370: pending
TXN-1371: approved
TXN-1372: approved
TXN-1373: approved
TXN-1374: rejected (131)
TXN-1375: rejected (132)
TXN-1376: approved
TXN-1377: rejected (133)
TXN-1378: pending
TXN-1379: approved
TXN-1380: rejected (134)
TXN-1381: pending
TXN-1382: rejected (135)
TXN-1383: pending
TXN-1384: rejected (136)
TXN-1385: approved
TXN-1386: rejected (137)
TXN-1387: pending
TXN-1388: pending
TXN-1389: approved
TXN-1390: approved
TXN-1391: approved
TXN-1392: approved
TXN-1393: pending
TXN-1394: pending
TXN-1395: rejected (138)
TXN-1396: rejected (139)
TXN-1397: pending
TXN-1398: approved
TXN-1399: pending

So the total count of rejected is 139.

**Question 3: Of the rows with kind "refund" and status "approved", which has the largest amount? Give its ID.**

Let me find all refund + approved rows:

TXN-1001 | refund | 6193.24 | approved
TXN-1009 | refund | 6048.33 | approved
TXN-1010 | refund | 5791.53 | approved
TXN-1017 | refund | 4038.76 | approved
TXN-1021 | refund | 1754.33 | approved
TXN-1022 | refund | 1852.33 | approved
TXN-1042 | refund | 497.10 | approved
TXN-1054 | refund | 222.37 | approved
TXN-1061 | refund | 5734.89 | approved
TXN-1072 | refund | 6151.78 | approved
TXN-1093 | refund | 5664.31 | approved
TXN-1104 | refund | 4463.44 | approved
TXN-1113 | refund | 8219.43 | approved
TXN-1117 | refund | 536.48 | approved
TXN-1123 | refund | 8581.30 | approved
TXN-1142 | refund | 4871.97 | approved
TXN-1156 | refund | 5059.53 | approved
TXN-1166 | refund | 2625.14 | approved
TXN-1180 | refund | 2248.48 | approved
TXN-1189 | refund | 8588.85 | approved
TXN-1194 | refund | 5120.28 | approved
TXN-1195 | refund | 2865.59 | approved
TXN-1204 | refund | 4543.62 | approved
TXN-1212 | refund | 4110.14 | approved
TXN-1214 | refund | 328.95 | approved
TXN-1238 | refund | 1954.14 | approved
TXN-1250 | refund | 6727.73 | approved
TXN-1251 | refund | 4340.50 | approved
TXN-1255 | refund | 5290.11 | approved
TXN-1266 | refund | 9550.87 | approved
TXN-1273 | refund | 3417.45 | approved
TXN-1276 | refund | 5519.69 | approved
TXN-1304 | refund | 7027.76 | approved
TXN-1311 | refund | 1611.95 | approved
TXN-1321 | refund | 4512.30 | approved
TXN-1323 | refund | 8847.30 | approved
TXN-1326 | refund | 1176.99 | approved
TXN-1328 | refund | 5280.80 | approved
TXN-1332 | refund | 1405.68 | approved
TXN-1334 | refund | 2537.98 | approved
TXN-1362 | refund | 1507.18 | approved
TXN-1368 | refund | 1649.36 | approved
TXN-1369 | refund | 2086.44 | approved
TXN-1379 | refund | 6611.25 | approved
TXN-1391 | refund | 1762.92 | approved

Now let me find the largest amount:
- TXN-1266: 9550.87
- TXN-1323: 8847.30
- TXN-1189: 8588.85
- TXN-1123: 8581.30
- TXN-1113: 8219.43
- TXN-1304: 7027.76
- TXN-1250: 6727.73
- TXN-1379: 6611.25
- TXN-1001: 6193.24
- TXN-1072: 6151.78
- TXN-1009: 6048.33
- TXN-1010: 5791.53
- TXN-1061: 5734.89
- TXN-1093: 5664.31
- TXN-1276: 5519.69
- TXN-1328: 5280.80
- TXN-1255: 5290.11
- TXN-1194: 5120.28
- TXN-1156: 5059.53
- TXN-1142: 4871.97
- TXN-1204: 4543.62
- TXN-1321: 4512.30
- TXN-1104: 4463.44
- TXN-1251: 4340.50
- TXN-1212: 4110.14
- TXN-1017: 4038.76
- TXN-1273: 3417.45
- TXN-1195: 2865.59
- TXN-1166: 2625.14
- TXN-1334: 2537.98
- TXN-1180: 2248.48
- TXN-1369: 2086.44
- TXN-1238: 1954.14
- TXN-1022: 1852.33
- TXN-1021: 1754.33
- TXN-1391: 1762.92
- TXN-1311: 1611.95
- TXN-1362: 1507.18
- TXN-1332: 1405.68
- TXN-1326: 1176.99
- TXN-1368: 1649.36
- TXN-1042: 497.10
- TXN-1117: 536.48
- TXN-1214: 328.95
- TXN-1054: 222.37

The largest is TXN-1266 with 9550.87.

**Question 4: How many distinct vendors appear anywhere in the ledger?**

Let me list all vendors:
- Ambleside Print
- Pemberton Freight
- Northwind Supplies
- Quillon Software
- Rivet & Sons
- Cortez Logistics
- Halloway Instruments
- Saltmarsh Catering

That's 8 distinct vendors.

**Question 5: How many distinct vendors have at least one row with status "pending"?**

Let me check which vendors have at least one pending row:

Ambleside Print: TXN-1005 (pending) ✓
Pemberton Freight: TXN-1079 (pending) ✓
Northwind Supplies: TXN-1011 (pending) ✓
Quillon Software: TXN-1015 (pending) ✓
Rivet & Sons: TXN-1014 (pending) ✓
Cortez Logistics: TXN-1069 (pending) ✓
Halloway Instruments: TXN-1026 (pending) ✓
Saltmarsh Catering: TXN-1018 (pending) ✓

All 8 vendors have at least one pending row.

So the answer is 8.

Let me double-check question 1:
- TXN-1029: 4230.28
- TXN-1126: 1267.81
- TXN-1139: 4330.47
- TXN-1222: 1406.28
- TXN-1305: 2069.20

Sum: 4230.28 + 1267.81 = 5498.09
5498.09 + 4330.47 = 9828.56
9828.56 + 1406.28 = 11234.84
11234.84 + 2069.20 = 13304.04

Final answers:
- approved_invoice_total: 13304.04
- rejected_count: 139
- largest_refund_id: "TXN-1266"
- distinct_vendors: 8
- vendors_with_pending: 8

</details>

**Answer (as scored):**

{"approved_invoice_total": 13304.04, "rejected_count": 139, "largest_refund_id": "TXN-1266", "distinct_vendors": 8, "vendors_with_pending": 8}

