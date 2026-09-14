# Datasets

All raw files live on `hdd2` — paths and provenance in `config.md`. This file
documents what each dataset actually contains: what it is, its column
schema, and a real 5-row sample. Where a dataset ships as many files with an
identical schema, one representative file is shown in full and the rest are
listed by name only.

---

## CIC-IDS-2018

**What it is.** 10 days of simulated enterprise network traffic (AWS-hosted
victim infrastructure + a separate attacker infrastructure), captured by the
Canadian Institute for Cybersecurity. Each day isolates one attack family
(brute force, DoS, web attacks, infiltration, botnet) against several hours
of background benign traffic. Distributed as CICFlowMeter output — every row
is one bidirectional flow, already feature-engineered (not raw packets). No
source/destination IP columns in these CSVs — CIC stripped them from this
release.

**Files (10, identical 80-column schema):**
`Wednesday-14-02-2018` (FTP/SSH brute force) · `Thursday-15-02-2018` (DoS
GoldenEye/Slowloris) · `Friday-16-02-2018` (DoS SlowHTTPTest/Hulk) ·
`Thuesday-20-02-2018` (DDoS LOIC-HTTP) · `Wednesday-21-02-2018` (DDoS
LOIC-UDP/HOIC) · `Thursday-22-02-2018` / `Friday-23-02-2018` (web brute
force, XSS, SQL injection) · `Wednesday-28-02-2018` / `Thursday-01-03-2018`
(Infiltration) · `Friday-02-03-2018` (Bot).

**Header** (`Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv`):
```
Dst Port,Protocol,Timestamp,Flow Duration,Tot Fwd Pkts,Tot Bwd Pkts,TotLen Fwd Pkts,TotLen Bwd Pkts,Fwd Pkt Len Max,Fwd Pkt Len Min,Fwd Pkt Len Mean,Fwd Pkt Len Std,Bwd Pkt Len Max,Bwd Pkt Len Min,Bwd Pkt Len Mean,Bwd Pkt Len Std,Flow Byts/s,Flow Pkts/s,Flow IAT Mean,Flow IAT Std,Flow IAT Max,Flow IAT Min,Fwd IAT Tot,Fwd IAT Mean,Fwd IAT Std,Fwd IAT Max,Fwd IAT Min,Bwd IAT Tot,Bwd IAT Mean,Bwd IAT Std,Bwd IAT Max,Bwd IAT Min,Fwd PSH Flags,Bwd PSH Flags,Fwd URG Flags,Bwd URG Flags,Fwd Header Len,Bwd Header Len,Fwd Pkts/s,Bwd Pkts/s,Pkt Len Min,Pkt Len Max,Pkt Len Mean,Pkt Len Std,Pkt Len Var,FIN Flag Cnt,SYN Flag Cnt,RST Flag Cnt,PSH Flag Cnt,ACK Flag Cnt,URG Flag Cnt,CWE Flag Count,ECE Flag Cnt,Down/Up Ratio,Pkt Size Avg,Fwd Seg Size Avg,Bwd Seg Size Avg,Fwd Byts/b Avg,Fwd Pkts/b Avg,Fwd Blk Rate Avg,Bwd Byts/b Avg,Bwd Pkts/b Avg,Bwd Blk Rate Avg,Subflow Fwd Pkts,Subflow Fwd Byts,Subflow Bwd Pkts,Subflow Bwd Byts,Init Fwd Win Byts,Init Bwd Win Byts,Fwd Act Data Pkts,Fwd Seg Size Min,Active Mean,Active Std,Active Max,Active Min,Idle Mean,Idle Std,Idle Max,Idle Min,Label
```

**5 rows:**
```
443,6,28/02/2018 08:22:13,94658,6,7,708,3718,387,0,118,159.2846508613,1460,0,531.1428571429,673.1182235367,46757.8017705846,137.3365167234,7888.1666666667,11130.0425943262,24325,0,72880,14576,12590.3839695221,24385,363,72178,12029.6666666667,13189.2575176416,24718,0,0,0,0,0,132,152,63.3860846416,73.9504320818,0,1460,316.1428571429,519.2058813734,269574.747252747,0,0,1,1,0,0,0,1,1,340.4615384615,118,531.1428571429,0,0,0,0,0,0,6,708,7,3718,8192,7484,3,20,0,0,0,0,0,0,0,0,Benign
443,6,28/02/2018 08:22:13,206,2,0,0,0,0,0,0,0,0,0,0,0,0,9708.7378640777,206,0,206,206,206,206,0,206,206,0,0,0,0,0,0,0,0,0,40,0,9708.7378640777,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,258,-1,0,20,0,0,0,0,0,0,0,0,Benign
445,6,28/02/2018 08:22:15,165505,3,1,0,0,0,0,0,0,0,0,0,0,0,24.1684541253,55168.3333333333,95478.1464908768,165417,35,165505,82752.5,116980.210345597,165470,35,0,0,0,0,0,0,0,0,0,72,32,18.1263405939,6.0421135313,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,1,0,8192,8192,0,20,0,0,0,0,0,0,0,0,Benign
443,6,28/02/2018 08:22:16,102429,6,7,708,3718,387,0,118,159.2846508613,1460,0,531.1428571429,673.1182235367,43210.4189243281,126.9171816575,8535.75,10956.6377235238,24473,0,80271,16054.2,14269.7541569573,31379,366,79733,13288.8333333333,14753.4827266875,30931,0,0,0,0,0,132,152,58.577160765,68.3400208925,0,1460,316.1428571429,519.2058813734,269574.747252747,0,0,1,1,0,0,0,1,1,340.4615384615,118,531.1428571429,0,0,0,0,0,0,6,708,7,3718,8192,7484,3,20,0,0,0,0,0,0,0,0,Benign
443,6,28/02/2018 08:22:16,167,2,0,0,0,0,0,0,0,0,0,0,0,0,11976.0479041916,167,0,167,167,167,167,0,167,167,0,0,0,0,0,0,0,0,0,40,0,11976.0479041916,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,258,-1,0,20,0,0,0,0,0,0,0,0,Benign
```

---

## CIC-IDS-2017

**What it is.** The predecessor to CIC-IDS-2018 — 5 days (Mon–Fri) of the
same style of simulated enterprise traffic, some days split into AM/PM
segments per attack family. Same CICFlowMeter feature tradition, ~78
columns, but **does include** `Flow ID`, `Source IP`, `Source Port`,
`Destination IP` fields in the raw CSV header (note: they don't appear in
the header line captured below because CIC's public zip's `MachineLearningCVE`
CSVs already have those columns stripped in this release too, mirroring
2018 — see the actual header dump).

**Files (9, identical schema):**
`Monday-WorkingHours` (benign only) · `Tuesday-WorkingHours` (brute force) ·
`Wednesday-workingHours` (DoS/Heartbleed) ·
`Thursday-WorkingHours-Morning-WebAttacks` ·
`Thursday-WorkingHours-Afternoon-Infilteration` ·
`Friday-WorkingHours-Morning` (benign) ·
`Friday-WorkingHours-Afternoon-PortScan` ·
`Friday-WorkingHours-Afternoon-DDos`.

**Header** (`Wednesday-workingHours.pcap_ISCX.csv`):
```
Destination Port, Flow Duration, Total Fwd Packets, Total Backward Packets,Total Length of Fwd Packets, Total Length of Bwd Packets, Fwd Packet Length Max, Fwd Packet Length Min, Fwd Packet Length Mean, Fwd Packet Length Std,Bwd Packet Length Max, Bwd Packet Length Min, Bwd Packet Length Mean, Bwd Packet Length Std,Flow Bytes/s, Flow Packets/s, Flow IAT Mean, Flow IAT Std, Flow IAT Max, Flow IAT Min,Fwd IAT Total, Fwd IAT Mean, Fwd IAT Std, Fwd IAT Max, Fwd IAT Min,Bwd IAT Total, Bwd IAT Mean, Bwd IAT Std, Bwd IAT Max, Bwd IAT Min,Fwd PSH Flags, Bwd PSH Flags, Fwd URG Flags, Bwd URG Flags, Fwd Header Length, Bwd Header Length,Fwd Packets/s, Bwd Packets/s, Min Packet Length, Max Packet Length, Packet Length Mean, Packet Length Std, Packet Length Variance,FIN Flag Count, SYN Flag Count, RST Flag Count, PSH Flag Count, ACK Flag Count, URG Flag Count, CWE Flag Count, ECE Flag Count, Down/Up Ratio, Average Packet Size, Avg Fwd Segment Size, Avg Bwd Segment Size, Fwd Header Length,Fwd Avg Bytes/Bulk, Fwd Avg Packets/Bulk, Fwd Avg Bulk Rate, Bwd Avg Bytes/Bulk, Bwd Avg Packets/Bulk,Bwd Avg Bulk Rate,Subflow Fwd Packets, Subflow Fwd Bytes, Subflow Bwd Packets, Subflow Bwd Bytes,Init_Win_bytes_forward, Init_Win_bytes_backward, act_data_pkt_fwd, min_seg_size_forward,Active Mean, Active Std, Active Max, Active Min,Idle Mean, Idle Std, Idle Max, Idle Min, Label
```

**5 rows:**
```
80,38308,1,1,6,6,6,6,6,0,6,6,6,0,313.250496,52.208416,38308,0,38308,38308,0,0,0,0,0,0,0,0,0,0,0,0,0,0,20,20,26.104208,26.104208,6,6,6,0,0,0,0,0,0,1,1,0,0,1,9,6,6,20,0,0,0,0,0,0,1,6,1,6,255,946,0,20,0,0,0,0,0,0,0,0,BENIGN
389,479,11,5,172,326,79,0,15.63636364,31.4492376,163,0,65.2,89.27877687,1039665.971,33402.92276,31.93333333,25.51040871,73,0,479,47.9,38.9428356,109,1,401,100.25,101.7361784,237,3,0,0,0,0,368,176,22964.50939,10438.41336,0,163,29.29411765,56.52959922,3195.595588,0,0,0,1,0,0,0,0,0,31.125,15.63636364,65.2,368,0,0,0,0,0,0,11,172,5,326,29200,260,4,32,0,0,0,0,0,0,0,0,BENIGN
88,1095,10,6,3150,3150,1575,0,315,632.5616351,1575,0,525,813.3265027,5753424.658,14611.87215,73,204.9609719,810,1,1095,121.6666667,298.7461297,915,1,995,199,345.5350923,810,3,0,0,0,0,336,208,9132.420091,5479.452055,0,1575,370.5882353,671.7515406,451250.1324,0,0,0,1,0,0,0,0,0,393.75,315,525,336,0,0,0,0,0,0,10,3150,6,3150,29200,2081,3,32,0,0,0,0,0,0,0,0,BENIGN
389,15206,17,12,3452,6660,1313,0,203.0588235,425.7784739,3069,0,555,977.4803416,665000.6576,1907.141918,543.0714286,2519.931377,13391,0,15206,950.375,3322.417812,13391,2,15112,1373.818182,4176.449588,13961,3,0,0,0,0,560,388,1117.979745,789.1621728,0,3069,337.0666667,704.6540816,496537.3747,0,0,0,1,0,0,0,0,0,348.6896552,203.0588235,555,560,0,0,0,0,0,0,17,3452,12,6660,29200,0,10,32,0,0,0,0,0,0,0,0,BENIGN
88,1092,9,6,3150,3152,1575,0,350,694.5097192,1576,0,525.3333333,813.8429005,5771062.271,13736.26374,78,207.000929,794,1,1092,136.5,313.850738,910,1,1015,203,333.2401536,794,3,0,0,0,0,304,208,8241.758242,5494.505495,0,1576,393.875,704.585067,496440.1167,0,0,0,1,0,0,0,0,0,420.1333333,350,525.3333333,304,0,0,0,0,0,0,9,3150,6,3152,29200,2081,2,32,0,0,0,0,0,0,0,0,BENIGN
```

---

## CTU-13

**What it is.** 13 independently-captured, real (not simulated) botnet
scenarios from CTU University, each a mix of genuine botnet traffic
(Neris, Rbot, Virut, Menti, Sogou, Murlo, NSIS.ay malware families),
normal traffic, and background traffic. Distributed as **Argus
bidirectional netflow** (`.binetflow`) — a coarser, different flow format
than CICFlowMeter: no TCP flag counts, no active/idle sub-flow stats, but
**real source/destination IP** and **real TTL** on every flow (CIC's CSVs
mostly zero these out).

**Files (13, identical 33-column schema)** — scenario number : malware
family: `42`/`43`/`50` Neris · `44`/`45`/`51`/`52` Rbot · `46`/`54` Virut ·
`47` Menti · `48` Sogou · `49` Murlo · `53` NSIS.ay.

**Header** (`scenario42_capture20110810.binetflow.csv`):
```
SrcAddr,DstAddr,Proto,Sport,Dport,State,sTos,dTos,SrcWin,DstWin,sHops,dHops,StartTime,LastTime,sTtl,dTtl,TcpRtt,SynAck,AckDat,SrcPkts,DstPkts,SrcBytes,DstBytes,SAppBytes,DAppBytes,Dur,TotPkts,TotBytes,TotAppByte,Rate,SrcRate,DstRate,Label
```

**5 rows:**
```
94.44.127.113,147.32.84.59,tcp,1577,6881,RST,0,0,131070,0,14,1,2011/08/10 09:46:59.607825,2011/08/10 09:47:00.634364,114,63,0.000000,0.000000,0.000000,2,2,156,120,0,0,1.026539,4,276,0,2.922441,0.974147,1.900205,flow=Background-Established-cmpgw-CVUT
94.44.127.113,147.32.84.59,tcp,1577,6881,RST,0,0,131070,0,14,1,2011/08/10 09:47:00.634364,2011/08/10 09:47:01.643959,114,63,0.000000,0.000000,0.000000,2,2,156,120,0,0,1.009595,4,276,0,2.971488,0.990496,1.999236,flow=Background-Established-cmpgw-CVUT
147.32.86.89,77.75.73.33,tcp,4768,80,RST,0,0,65535,6886,1,8,2011/08/10 09:47:48.185538,2011/08/10 09:47:51.242124,127,56,0.000000,0.000000,0.000000,2,1,122,60,0,0,3.056586,3,182,0,0.654325,0.327162,0.000000,flow=Background-TCP-Attempt
147.32.86.89,77.75.73.33,tcp,4788,80,RST,0,0,65535,6897,1,8,2011/08/10 09:47:48.230897,2011/08/10 09:47:51.342666,127,56,0.000000,0.000000,0.000000,2,1,122,60,0,0,3.111769,3,182,0,0.642721,0.321361,0.000000,flow=Background-TCP-Attempt
147.32.86.89,77.75.73.33,tcp,4850,80,RST,0,0,65535,6809,1,8,2011/08/10 09:47:48.963351,2011/08/10 09:47:52.046762,127,56,0.000000,0.000000,0.000000,2,1,122,60,0,0,3.083411,3,182,0,0.648632,0.324316,0.000000,flow=Background-TCP-Attempt
```
`Label` is free text — `flow=Background-*` / `flow=Normal-*` /
`flow=From-Botnet-*` / `flow=To-Botnet-*` — parsed by substring match, not
a clean category column.

---

## CICIoT2023

**What it is.** Traffic from 105 real IoT devices in a dedicated testbed,
covering 33 attack classes across 7 families (DDoS, DoS, Recon, Web-based,
Brute Force, Spoofing, Mirai) plus benign — the largest and most recent
dataset here (2023), purpose-built for IoT botnet/recon behavior rather
than enterprise traffic. Each row is a per-flow/window statistical summary
(41 columns), not raw packets.

**Files:** 63 `MergedNN.csv` files (`Merged01.csv`…`Merged63.csv`), each an
arbitrary shard of the full merged dataset — same schema throughout, no
per-file meaning (unlike the day-named files in the other datasets).

**Header** (`Merged01.csv`):
```
Header_Length,Protocol Type,Time_To_Live,Rate,fin_flag_number,syn_flag_number,rst_flag_number,psh_flag_number,ack_flag_number,ece_flag_number,cwr_flag_number,ack_count,syn_count,fin_count,rst_count,HTTP,HTTPS,DNS,Telnet,SMTP,SSH,IRC,TCP,UDP,DHCP,ARP,ICMP,IGMP,IPv,LLC,Tot sum,Min,Max,AVG,Std,Tot size,IAT,Number,Variance,Label
```

**5 rows:**
```
19.92,6,63.36,25893.962217557724,0.0,0.0,0.0,0.99,0.99,0.0,0.0,99,0,0,0,0.0,0.01,0.0,0.0,0.0,0.0,0.0,0.99,0.0,0.0,0.01,0.0,0.0,0.99,0.99,6421,60,481,64.21,42.09999999999997,64.21,3.861904144287109E-5,100,1772.4099999999978,DDOS-PSHACK_FLOOD
0.0,47,64.0,3703.841330954946,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0,0,0,0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.01,0.0,1.0,1.0,57320,98,578,573.2,48.00000000000007,573.2,2.707815170288E-4,100,2304.000000000007,MIRAI-GREIP_FLOOD
7.92,17,65.91,19673.095684803004,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0,0,0,0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.99,0.0,0.0,0.01,0.0,1.0,1.0,6010,60,70,60.1,1.0000000000000009,60.1,5.74493408203125E-5,100,1.0000000000000018,DOS-UDP_FLOOD
20.4,6,110.5,261.6648262868622,0.1,0.0,0.3,0.2,0.4,0.0,0.0,4,0,1,3,0.0,0.5,0.0,0.0,0.0,0.0,0.0,0.7,0.2,0.0,0.1,0.0,0.0,0.9,0.9,2223,54,1500,222.3,451.5966858455304,222.3,0.0047659873962402,10,203939.56666666668,DNS_SPOOFING
0.32,1,63.96,28944.19984818163,0.0,0.0,0.0,0.0,0.01,0.0,0.0,1,0,0,0,0.0,0.01,0.0,0.0,0.0,0.0,0.0,0.01,0.0,0.0,0.0,0.99,0.0,1.0,1.0,6006,60,66,60.06,0.5999999999999994,60.06,3.455877304077148E-5,100,0.3599999999999992,DDOS-ICMP_FLOOD
```

---

## UNSW-NB15

**What it is.** Synthetic hybrid of real modern benign traffic and
synthetic attacks (Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic,
Reconnaissance, Shellcode, Worms) generated with IXIA PerfectStorm at
UNSW's Cyber Range Lab. What's downloaded here is the standard **training
partition** (~175K rows), the same one virtually all UNSW-NB15 papers use.

**Files:** `UNSW_NB15_training-set.csv` (data) + `NUSW-NB15_features.csv`
(feature dictionary, 49 rows describing every column — genuinely useful
since several column names are non-obvious, e.g. `sttl`/`dttl`, `ct_*`
counts).

**Header:**
```
id,dur,proto,service,state,spkts,dpkts,sbytes,dbytes,rate,sttl,dttl,sload,dload,sloss,dloss,sinpkt,dinpkt,sjit,djit,swin,stcpb,dtcpb,dwin,tcprtt,synack,ackdat,smean,dmean,trans_depth,response_body_len,ct_srv_src,ct_state_ttl,ct_dst_ltm,ct_src_dport_ltm,ct_dst_sport_ltm,ct_dst_src_ltm,is_ftp_login,ct_ftp_cmd,ct_flw_http_mthd,ct_src_ltm,ct_srv_dst,is_sm_ips_ports,attack_cat,label
```

**5 rows:**
```
1,0.121478,tcp,-,FIN,6,4,258,172,74.08749,252,254,14158.94238,8495.365234,0,0,24.2956,8.375,30.177547,11.830604,255,621772692,2202533631,255,0,0,0,43,43,0,0,1,0,1,1,1,1,0,0,0,1,1,0,Normal,0
2,0.649902,tcp,-,FIN,14,38,734,42014,78.473372,62,252,8395.112305,503571.3125,2,17,49.915,15.432865,61.426934,1387.77833,255,1417884146,3077387971,255,0,0,0,52,1106,0,0,43,1,1,1,1,2,0,0,0,1,6,0,Normal,0
3,1.623129,tcp,-,FIN,8,16,364,13186,14.170161,62,252,1572.271851,60929.23047,1,6,231.875571,102.737203,17179.58686,11420.92623,255,2116150707,2963114973,255,0.111897,0.061458,0.050439,46,824,0,0,7,1,2,1,1,3,0,0,0,2,6,0,Normal,0
4,1.681642,tcp,ftp,FIN,12,12,628,770,13.677108,62,252,2740.178955,3358.62207,1,3,152.876547,90.235726,259.080172,4991.784669,255,1107119177,1047442890,255,0,0,0,52,64,0,0,1,1,2,1,1,3,1,1,0,2,1,0,Normal,0
5,0.449454,tcp,-,FIN,10,6,534,268,33.373826,254,252,8561.499023,3987.059814,2,1,47.750333,75.659602,2415.837634,115.807,255,2436137549,1977154190,255,0.128381,0.071147,0.057234,53,45,0,0,43,1,2,2,1,40,0,0,0,2,39,0,Normal,0
```
`attack_cat = Normal` and `label = 0` for these first 5 rows — the file is
not shuffled, benign rows come first.

---

## NSL-KDD (DARPA'98/99 → KDD Cup 99 successor)

**What it is.** The rebalanced, de-duplicated successor to the 1999
DARPA/KDD Cup 99 intrusion dataset — every paper that cites "the DARPA
dataset" today virtually always means this. Derived from 1998 DARPA
Lincoln Lab simulated traffic, labeled into 4 attack categories (DoS, Probe,
R2L, U2R) plus normal. **Files ship with no header row** — column names
come from the dataset's fixed, documented 41-feature + label + difficulty
schema, reproduced below since it isn't in the file itself.

**Files:** `KDDTrain+.txt`, `KDDTest+.txt` — same 43-field schema.

**Column names (in order, not present in the file):**
```
duration,protocol_type,service,flag,src_bytes,dst_bytes,land,wrong_fragment,urgent,hot,num_failed_logins,logged_in,num_compromised,root_shell,su_attempted,num_root,num_file_creations,num_shells,num_access_files,num_outbound_cmds,is_host_login,is_guest_login,count,srv_count,serror_rate,srv_serror_rate,rerror_rate,srv_rerror_rate,same_srv_rate,diff_srv_rate,srv_diff_host_rate,dst_host_count,dst_host_srv_count,dst_host_same_srv_rate,dst_host_diff_srv_rate,dst_host_same_src_port_rate,dst_host_srv_diff_host_rate,dst_host_serror_rate,dst_host_srv_serror_rate,dst_host_rerror_rate,dst_host_srv_rerror_rate,class,difficulty_level
```

**5 rows** (`KDDTrain+.txt`):
```
0,tcp,ftp_data,SF,491,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,0.00,0.00,0.00,0.00,1.00,0.00,0.00,150,25,0.17,0.03,0.17,0.00,0.00,0.00,0.05,0.00,normal,20
0,udp,other,SF,146,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,13,1,0.00,0.00,0.00,0.00,0.08,0.15,0.00,255,1,0.00,0.60,0.88,0.00,0.00,0.00,0.00,0.00,normal,15
0,tcp,private,S0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,123,6,1.00,1.00,0.00,0.00,0.05,0.07,0.00,255,26,0.10,0.05,0.00,0.00,1.00,1.00,0.00,0.00,neptune,19
0,tcp,http,SF,232,8153,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,5,5,0.20,0.20,0.00,0.00,1.00,0.00,0.00,30,255,1.00,0.00,0.03,0.04,0.03,0.01,0.00,0.01,normal,21
0,tcp,http,SF,199,420,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,30,32,0.00,0.00,0.00,0.00,1.00,0.00,0.09,255,255,1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,normal,21
```

---

## LANL Comprehensive Multi-Source Cyber-Security Events ("Authentication Dataset")

**What it is.** 58 consecutive days of real, de-identified internal network
activity from Los Alamos National Laboratory: Windows authentication events,
network flows, DNS lookups, process events, and ground-truth red-team
compromise events, across ~12,425 users / 17,684 computers. The only dataset
here with genuine **lateral-movement ground truth** (`redteam.txt.gz`).

**Status:** `flows.txt.gz` (1.1 GB) and `redteam.txt.gz` (4.8 KB, 749 events
total — the complete ground-truth compromise log) are fully downloaded.
`auth.txt.gz` (7.2 GB) is still downloading in the background — LANL's
server throttles to roughly 100 KB/s, so this takes many hours; sample rows
below are read from the partial file (gzip streams, so this is real data,
just not necessarily representative of the full 58 days yet).

**Files fetched:** `auth.txt.gz`, `flows.txt.gz`, `redteam.txt.gz` (not
`proc.txt.gz` or `dns.txt.gz` — see `config.md`). **No header row in any
file** — comma-delimited, columns per LANL's documentation:

**`auth.txt.gz`** — `time, source_user@domain, destination_user@domain, source_computer, destination_computer, authentication_type, logon_type, authentication_orientation, success/failure`
```
1,ANONYMOUS LOGON@C586,ANONYMOUS LOGON@C586,C1250,C586,NTLM,Network,LogOn,Success
1,ANONYMOUS LOGON@C586,ANONYMOUS LOGON@C586,C586,C586,?,Network,LogOff,Success
1,C101$@DOM1,C101$@DOM1,C988,C988,?,Network,LogOff,Success
1,C1020$@DOM1,SYSTEM@C1020,C1020,C1020,Negotiate,Service,LogOn,Success
1,C1021$@DOM1,C1021$@DOM1,C1021,C625,Kerberos,Network,LogOn,Success
```
Note `ANONYMOUS LOGON` and NTLM-over-network in row 1 — worth remembering
when picking a "normal-looking" example for a demo; the very first rows of
the file are already a mildly suspicious pattern (unauthenticated network
logon), which is realistic for 58 days of real enterprise activity.

**`flows.txt.gz`** — `time, duration, source_computer, source_port, destination_computer, destination_port, protocol, packet_count, byte_count` (129,977,412 rows total)
```
1,0,C1065,389,C3799,N10451,6,10,5323
1,0,C1423,N1136,C1707,N1,6,5,847
1,0,C1423,N1142,C1707,N1,6,5,847
1,0,C14909,N8191,C5720,2049,6,1,52
1,0,C14909,N8192,C5720,2049,6,1,52
```

**`redteam.txt.gz`** — `time, user@domain, source_computer, destination_computer` (749 ground-truth compromise events, complete file)
```
150885,U620@DOM1,C17693,C1003
151036,U748@DOM1,C17693,C305
151648,U748@DOM1,C17693,C728
151993,U6115@DOM1,C17693,C1173
153792,U636@DOM1,C17693,C294
```

All fields are de-identified integers/tokens (`C123` = computer, `U123` =
user, `N123` = non-well-known port) except well-known ports and system
accounts, per LANL's anonymization policy. `time` is seconds since an
undisclosed epoch, resolution 1s.

---

## Knowledge bases (reference/enrichment, not training data)

These support the MITRE ATT&CK stage-mapping and explainability layer —
they don't have a "row" structure like the flow datasets above.

### MITRE ATT&CK (Enterprise, STIX 2.1)

`attack-stix-enterprise.json` — one JSON object per STIX object, 26,086
objects total: 858 `attack-pattern` (techniques), 733 `malware`, 268
`course-of-action` (mitigations), 191 `intrusion-set` (threat groups), plus
21,262 `relationship` objects linking them all together, and newer
`x-mitre-analytic` / `x-mitre-detection-strategy` objects (detection
guidance per technique).

Sample `attack-pattern` object (truncated):
```json
{
  "type": "attack-pattern",
  "name": "Extra Window Memory Injection",
  "description": "Adversaries may inject malicious code into process via Extra Window Memory (EWM) in order to evade process-based defenses as well as possibly elevate privileges. EWM injection is a method of executing arbitrary code in the address space of a separate live process. ..."
}
```
Each technique also carries `external_references` with its `T-number`
(e.g. `T1055`) — this is what `attack_map.py`-style code joins dataset
attack-type labels against.

### CAPEC (attack pattern catalog)

`capec_latest.xml` — MITRE's Common Attack Pattern Enumeration and
Classification, XML, one `<Attack_Pattern>` element per pattern with an
`ID`, `Name`, `Abstraction`, `Status`, and free-text `<Description>`:
```xml
<Attack_Pattern ID="1" Name="Accessing Functionality Not Properly Constrained by ACLs"
                Abstraction="Standard" Status="Draft">
   <Description>In applications, particularly web applications, access to
   functionality is mitigated by an authorization framework...</Description>
```
CAPEC patterns sit one level more concrete than ATT&CK techniques (a
technique like T1190 "Exploit Public-Facing Application" maps to several
CAPEC patterns describing the actual exploitation mechanics) — useful for
enriching an explanation beyond just the ATT&CK stage name.

### CVE / NVD

`nvd-json-data-feeds/` — one JSON file per CVE (386K files, sharded into
`CVE-<year>/CVE-<year>-<NNxx>/CVE-<id>.json` directories), full current NVD
record per CVE:
```json
{
  "id": "CVE-2026-0496",
  "sourceIdentifier": "cna@sap.com",
  "published": "2026-01-13T02:15:51.990",
  "vulnStatus": "Deferred",
  "descriptions": [
    {"lang": "en", "value": "SAP Fiori App Intercompany Balance Reconciliation allows an attacker with high privileges to upload any file..."}
  ],
  "affected": [ ... ]
}
```
Not directly joinable to any flow dataset here (none of the flow-level
datasets carry service/software version info) — its use is scoped to
enriching the demo/explainability UI when a flagged flow's destination port
maps to a well-known service, not to model training.

---

## Usage plan — what each dataset is actually for

Not all ten sources feed the same pipeline the same way. Each plays a
distinct, specific role in training or evaluating the regime-switching
world model (per `docs/literature_survey_world_model_architecture.md`)
over the flat-vector state representation (per
`docs/literature_survey_network_representation.md`).

| dataset | role |
|---|---|
| **CIC-IDS-2018** | **Primary training corpus.** Multi-day, multi-family, and — critically — the two Infiltration days are the only source here with a genuine, labeled multi-stage kill chain (exploit delivery → internal port scan from the compromised host). This is what the world model's dynamics (`P(S_t+1\|S_t)`) actually get trained on. Temporal splits (train on one part of a day, test on the rest / a held-out day) per the evaluation design already agreed. |
| **CIC-IDS-2017** | **Cross-dataset generalization test, not training data.** Same CICFlowMeter tradition and overlapping attack categories (brute force, DoS, web attacks, infiltration, port scan, DDoS) but a different capture — different network, different day, two years apart. Train on 2018, evaluate zero-shot on 2017 (or vice versa): if performance survives, that's real evidence of learned dynamics rather than memorized 2018-specific artifacts (the exact failure mode §1 of the representation survey warns about). |
| **CTU-13** | **Leave-one-botnet-family-out generalization sweep**, extending the "one held-out C2 source" anecdote (CIC's single Bot day) into a real n=13 test: train on all-but-one family (Neris/Rbot/Virut/Menti/Sogou/Murlo/NSIS.ay), hold one out, repeat, pool. Also the only flow dataset with **real IPs and real TTL** — if we ever revisit host-graph features, this is the only source that supports them honestly. |
| **CICIoT2023** | **Two roles.** (1) Domain-transfer test — enterprise-trained model against IoT botnet/DDoS traffic, relevant to the PS's Critical Infrastructure framing. (2) Fills a real gap: CICIoT2023 has explicit `RECON-*` attack labels (port scan, OS scan, ping sweep, vulnerability scan), which CIC-IDS-2018/2017 don't cleanly provide — useful training signal for the **Reconnaissance** ATT&CK stage, which is otherwise underrepresented. Its heavy DDoS/Mirai skew also makes it a good **specificity test**: these are real attacks that are *not* progressing toward infiltration, so a well-calibrated model should mostly not fire the infiltration alarm on them. |
| **UNSW-NB15** | **Second independent generalization axis** — a different traffic generator entirely (synthetic IXIA PerfectStorm, not CIC's testbed), so it stress-tests overfitting to CIC-specific traffic-generator artifacts rather than dataset-specific attack artifacts. Also has an explicit, clean `Reconnaissance` category in `attack_cat`, and genuine packet-level-adjacent fields (`sttl`/`dttl`, `sjit`/`djit`, window sizes) that CIC's CSVs lack. |
| **NSL-KDD (DARPA)** | **Not used for world-model training** — rows are individual connection records with no timestamp field, so there's no way to build the time-windowed sequences the dynamics model needs. Its role is a classical-ML sanity baseline only: confirms the logistic-regression baseline behaves reasonably on the textbook dataset, and stands as an honest note in the report ("included because the PS names it; unsuitable for sequence modeling, and here's why") rather than silently dropped. |
| **LANL Authentication Dataset** | **Different modality, different role.** Not network flow data — host authentication events. Its value is `redteam.txt.gz`: the only **direct, ground-truth lateral-movement labels** in the whole collection (every other dataset's "Lateral Movement" mapping is inferred from attack-type strings, not observed compromise events). Realistic use: validate/calibrate the Lateral-Movement stage definition against real red-team episodes, and — time permitting — build a small auxiliary user→computer login-graph state channel from `auth.txt.gz`+`flows.txt.gz` as a case study, rather than merging it into the primary flow-feature pipeline (its schema has no packet/byte/flag statistics to merge with the CICFlowMeter-family feature vector). |
| **MITRE ATT&CK** | Not training data — the taxonomy the attack-type→stage mapping table (`cwm/attack_map.py`-style) is built and validated against, plus technique descriptions surfaced in the explainability UI next to a flagged prediction. |
| **CAPEC** | Optional explainability enrichment, one level more concrete than ATT&CK — e.g. attaching the actual attack-pattern mechanics text to a flagged SYN-flood sequence. Secondary to SHAP/attention; polish, not core. |
| **CVE/NVD** | Optional UI enrichment only — surfacing known CVEs for a flagged flow's destination port/service, when relevant. Not joinable to any dataset here at the record level (none carry service/version fingerprints), so scoped strictly to demo-time context, never model training. |
