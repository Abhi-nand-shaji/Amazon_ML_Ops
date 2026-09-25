# Edge-case diagnostics (validation entities)

Decision config: `{"mode": "global", "global_threshold": 0.74, "s2_threshold": 0.74, "s3_threshold": 0.74, "gate": null, "validation_macro_f05": 0.962783109836692, "oracle_given_blocking": 0.9875558449573475, "holdout_macro_f05": 0.9646786045680446}`. Validation entities: 7,500; true pairs: 25,998; overall macro F0.5 = 0.9628.

## Failure taxonomy (all validation entities)

| type | count | share of true pairs |
|---|---|---|
| A blocking (true match not a candidate) | 894 | 3.44% |
| B matching: true candidate rejected | 1,164 | 4.48% |
| B matching: false merge on an entity with matches | 160 | 0.62% |
| C singleton with a predicted match | 14 | 0.05% |

## Per edge case

Entities can belong to several cases. Precision / recall are pooled over the case's entities; F0.5 is the macro mean.

| edge case | S1 entities | candidate recall | precision | recall | macro F0.5 | A blocking failures | B matching failures | C singleton failures |
|---|---|---|---|---|---|---|---|---|
| 1 true singleton | 412 |  | 0.000 |  | 0.966 | 0 | 0 | 14 |
| 2 one-to-one match | 420 | 0.971 | 0.972 | 0.898 | 0.887 | 12 | 42 | 0 |
| 3 multiple matches | 6668 | 0.966 | 0.994 | 0.921 | 0.967 | 882 | 1,282 | 0 |
| 4 cross-source match (S2 and S3) | 6043 | 0.966 | 0.994 | 0.922 | 0.968 | 812 | 1,203 | 0 |
| 5 multiple matches within one source | 5759 | 0.966 | 0.994 | 0.922 | 0.969 | 809 | 1,170 | 0 |
| 6 name-only strong (name>=.85, address weak/missing) | 1342 | 0.970 | 0.994 | 0.887 | 0.951 | 169 | 492 | 0 |
| 7 address-only strong (address>=.85, name weak/missing) | 601 | 0.965 | 0.992 | 0.898 | 0.957 | 89 | 191 | 0 |
| 8 name+address both noisy (both < .7) | 849 | 0.969 | 0.994 | 0.896 | 0.954 | 110 | 281 | 0 |
| 9 name collision (false candidate with name>=.9) | 2511 | 0.942 | 0.986 | 0.890 | 0.944 | 493 | 541 | 8 |
| 10 address collision (false candidate with address>=.9) | 2047 | 0.964 | 0.991 | 0.922 | 0.964 | 244 | 341 | 5 |
| 11 missing name (either side) | 0 |  |  |  |  |  |  |  |
| 12 missing address (either side) | 865 | 0.976 | 0.993 | 0.841 | 0.929 | 86 | 510 | 0 |
| 13 multiple missing fields | 0 |  |  |  |  |  |  |  |
| 14 normalization-only (identical after normalization) | 4656 | 0.975 | 0.994 | 0.935 | 0.976 | 471 | 843 | 0 |
| 15 token-order (token-sort>=.95, plain ratio<.9) | 1429 | 0.976 | 0.995 | 0.943 | 0.980 | 147 | 228 | 0 |
| 16 typo (name JW>=.9 but not identical) | 5454 | 0.975 | 0.994 | 0.932 | 0.974 | 525 | 1,043 | 0 |
| 17 transliteration / non-Latin name | 821 | 0.930 | 0.988 | 0.857 | 0.933 | 219 | 261 | 0 |
| 22 ambiguous (true candidate AND a false candidate p>=.3) | 621 | 0.957 | 0.926 | 0.907 | 0.907 | 96 | 269 | 0 |

## Source 2 vs Source 3

| source | true pairs in candidates | mean name similarity (true pairs) | mean address similarity (true pairs) | address missing | non-Latin name | name identical after normalization | precision | recall (of candidate true pairs) |
|---|---|---|---|---|---|---|---|---|
| S2 | 12079 | 0.837 | 0.849 | 0.037 | 0.077 | 0.294 | 0.992 | 0.954 |
| S3 | 13025 | 0.836 | 0.775 | 0.036 | 0.042 | 0.299 | 0.993 | 0.953 |

## Not measurable / structural cases

- 18 country conflict: 0 of 7,638,365 training true pairs cross a country boundary and candidates are generated per country partition, so no cross-country candidate exists; country mismatch is treated as a hard partition because the data prove it safe.
- 19 unseen country (France): test only; `country` is not a model feature and partitions are formed from whatever labels occur. See `inference_summary.json` for per-country prediction statistics.
- 20 true match missed by blocking: 894 pairs (row A above); 21 true candidate rejected: 1,164 pairs (row B).
- 23 duplicate candidates: 0 duplicated (S1, candidate) rows in the validation candidate set (candidates are the union of several rules, deduplicated by construction).
- 24 empty ground truth parsed as zero matches: 412 validation singletons.

## D. Output / pipeline checks on output/*.tsv (cases 25-27)

- rows: matching 1,732,544, candidate 1,732,544; duplicate S1 rows: 0 / 0
- S1/other-source ids inside predicted lists: 0 entities
- duplicate ids inside a predicted list: 0 entities; inside a candidate list: 0 entities
- subset invariant (every match is a candidate) violated for 0 entities

## Representative examples (worst entities of each case)

- [1 true singleton] false merge p=0.99: S1 `Internal Medicine Legacy Health Inc. | 6355 Shedd Road, Eloy, AZ`  <->  `Internal Medicine Legacy Health Inc | 6376 Shedd Road, Eloy, Arizona`
- [1 true singleton] false merge p=0.76: S1 `Anand Investments Private Limited | A-128, Tower P5 Ashiana Palm Court, Raj Nagar Extn, Ghaziabad, Uttar Pradesh`  <->  `आनंद मैनेजमेंट प्राइवेट लिमिटेड | A-139, TOWER P5 ASHIANA PALM COURT, RAJ NAGAR EXTN, GHAZIABAD, RAE BARELI, Uttar Pradesh`
- [1 true singleton] false merge p=0.89: S1 `VHI Floral LLP | Great Social Bldg 5Th Flr 60Sir Pm Rd Vth Floor Fort, Mumbai, Maharashtra`  <->  `VHI Flribal LLP | (no address)`
- [2 one-to-one match] missed true match p=0.67: S1 `Super Impex Pvt Ltd | 8-2-269/10, #501, 5Th Floor, Trendset Towers, Road No:2 Banjara Hills, Hyderabad, Telangana`  <->  `సూపర్ ఇంపెక్స్ ప్రైవేట్ లిమిటెడ్ | 8-2-269/10, #501, 5TH FLOOR, TRENDSET TOWERS, ROAD NO:2 BANJARA HILLS, HYDERABAD, Andhra Pradesh`
- [2 one-to-one match] missed true match p=0.23: S1 `Rinam Social Care | Jyoti Smiriti Appartment Flat No 304 Yar Pur D.V.C Road Gardanibagh, Patna, Bihar`  <->  `M/s Rinam Sdocbia Care | (no address)`
- [3 multiple matches] missed true match p=0.21: S1 `Krishna Business | Prajay Princetion Towers, Unitno507 Plot No:/3And1/4Chitralayout, L.B Nagar, S, Telangana, Hyderabad, Aroornagar, 5Th Floor`  <->  `కృష్ణ బిజినెస్ | NO A-5TH FLOOR, HYDERABAD, AROORNAGAR, Andhra Pradesh`
- [3 multiple matches] missed true match p=0.47: S1 `Dream Foundation | Palam Vihar, Gurgaon, Sector-23, Haryana, House No.3993`  <->  `ड्रीम फाउंडेशन | GURGAON, SECTOR-23, PLOT 235 HOUSE NO.3993, हरियाणा`
- [4 cross-source match (S2 and S3)] missed true match p=0.28: S1 `Red Foundation Private Limited | 109, Ground Floor, Shakti Khand 1St Indirapuram, Ghaziabad, Uttar Pradesh`  <->  `रेड फाउंडेशन प्राइवेट लिमिटेड | 09, Ground Floor, Shakti Khand 1St Indirapuram, Ghaziabad, UP`
- [4 cross-source match (S2 and S3)] missed true match p=0.47: S1 `Dream Foundation | Palam Vihar, Gurgaon, Sector-23, Haryana, House No.3993`  <->  `ड्रीम फाउंडेशन | GURGAON, SECTOR-23, PLOT 235 HOUSE NO.3993, हरियाणा`
- [5 multiple matches within one source] missed true match p=0.21: S1 `Krishna Business | Prajay Princetion Towers, Unitno507 Plot No:/3And1/4Chitralayout, L.B Nagar, S, Telangana, Hyderabad, Aroornagar, 5Th Floor`  <->  `కృష్ణ బిజినెస్ | NO A-5TH FLOOR, HYDERABAD, AROORNAGAR, Andhra Pradesh`
- [5 multiple matches within one source] missed true match p=0.16: S1 `Buckner & Rowe Consulting LP | 3200 Forest Hills Road, Petersburg City, VA`  <->  `Buckner & Rowe Consulting Inc | VA, PETERSBURG CITY, 3200 FROEST HILLS ROAD, PMB 7141`
- [6 name-only strong (name>=.85, address weak/missing)] missed true match p=0.30: S1 `Bren Coffman, DDS | 511 Walnut Street, OH, Leetonia`  <->  `Bren Coffman, | (no address)`
- [6 name-only strong (name>=.85, address weak/missing)] missed true match p=0.03: S1 `Regional Retail Industries | 2607 Garfield Avenue, Des Moines, IA`  <->  `REGIONAL RETAIL INDUSTRIES INC | DES MOINES TOWNSHIP, IA, 260 GARFIELD AVE`
- [6 name-only strong (name>=.85, address weak/missing)] missed true match p=0.55: S1 `Falls Institutions Corp | 900 Locust Road, Pottsboro, TX`  <->  `falls institutions corp | (no address)`
- [7 address-only strong (address>=.85, name weak/missing)] missed true match p=0.67: S1 `Super Impex Pvt Ltd | 8-2-269/10, #501, 5Th Floor, Trendset Towers, Road No:2 Banjara Hills, Hyderabad, Telangana`  <->  `సూపర్ ఇంపెక్స్ ప్రైవేట్ లిమిటెడ్ | 8-2-269/10, #501, 5TH FLOOR, TRENDSET TOWERS, ROAD NO:2 BANJARA HILLS, HYDERABAD, Andhra Pradesh`
- [7 address-only strong (address>=.85, name weak/missing)] missed true match p=0.65: S1 `Super Healthcare Private Limited | Prof C N R Rao Block, Tumkur University, Tumkur, Karnataka`  <->  `ಸೂಪರ್ ಹೆಲ್ತ್‌ಕೇರ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್ | HN ##413 PROF C N R RAO BLCOK, TUMKUR UNIVERSITY, TUMKUR, Karnataka`
- [7 address-only strong (address>=.85, name weak/missing)] missed true match p=0.13: S1 `Travel Consultants | Kpp Vii/114 B Puthur, Payannur, Kannur, Kerala`  <->  `Shri Nylabelobelo | PLOT 505 KPP VII/114 B PUTHUR, PAYANNUR, KANNUR, Keralam`
- [8 name+address both noisy (both < .7)] missed true match p=0.21: S1 `Krishna Business | Prajay Princetion Towers, Unitno507 Plot No:/3And1/4Chitralayout, L.B Nagar, S, Telangana, Hyderabad, Aroornagar, 5Th Floor`  <->  `కృష్ణ బిజినెస్ | NO A-5TH FLOOR, HYDERABAD, AROORNAGAR, Andhra Pradesh`
- [8 name+address both noisy (both < .7)] missed true match p=0.50: S1 `Sophic Solutions Private Limited | Gorrayhatti Estate, Bedaguli Post, Chamarajanagar., Karnataka`  <->  `Avisyn | Gorrayhatti Estate, Chamrajanagara, Chamarajanagar., ಕರ್ನಾಟಕ`
- [8 name+address both noisy (both < .7)] missed true match p=0.01: S1 `Indian Developers Private Limited | Block-Dn, 7 Th Floor, Unit No. 712 Salt Lake, Kolkata, Kolkata, Plot 51, Howrah, West Bengal`  <->  `ইন্ডিয়ান ডেভেলপারস প্রাইভেট লিমিটেড | PLOT #51, HOWRAH, KOLKATA, পশ্চিমবঙ্গ`
- [9 name collision (false candidate with name>=.9)] missed true match p=0.01: S1 `Indian Developers Private Limited | Block-Dn, 7 Th Floor, Unit No. 712 Salt Lake, Kolkata, Kolkata, Plot 51, Howrah, West Bengal`  <->  `ইন্ডিয়ান ডেভেলপারস প্রাইভেট লিমিটেড | PLOT #51, HOWRAH, KOLKATA, পশ্চিমবঙ্গ`
- [10 address collision (false candidate with address>=.9)] false merge p=0.89: S1 `VHI Floral LLP | Great Social Bldg 5Th Flr 60Sir Pm Rd Vth Floor Fort, Mumbai, Maharashtra`  <->  `VHI Flribal LLP | (no address)`
- [10 address collision (false candidate with address>=.9)] false merge p=0.76: S1 `Anand Investments Private Limited | A-128, Tower P5 Ashiana Palm Court, Raj Nagar Extn, Ghaziabad, Uttar Pradesh`  <->  `आनंद मैनेजमेंट प्राइवेट लिमिटेड | A-139, TOWER P5 ASHIANA PALM COURT, RAJ NAGAR EXTN, GHAZIABAD, RAE BARELI, Uttar Pradesh`
- [12 missing address (either side)] missed true match p=0.03: S1 `One Enterprises Private Limited | Beside Sri G M Siddaramanna Girls Hostel, Panduranganagar, Chickpete, Tumkur, Karnataka`  <->  `One Enterprises Prívate Limited | (no address)`
- [12 missing address (either side)] missed true match p=0.55: S1 `Falls Institutions Corp | 900 Locust Road, Pottsboro, TX`  <->  `falls institutions corp | (no address)`
- [12 missing address (either side)] missed true match p=0.21: S1 `Krishna Business | Prajay Princetion Towers, Unitno507 Plot No:/3And1/4Chitralayout, L.B Nagar, S, Telangana, Hyderabad, Aroornagar, 5Th Floor`  <->  `కృష్ణ బిజినెస్ | NO A-5TH FLOOR, HYDERABAD, AROORNAGAR, Andhra Pradesh`
- [14 normalization-only (identical after normalization)] missed true match p=0.01: S1 `Gujarat Consultancy Private Limited | 6-3-570/1 To &7, Unit 403, 4Th Floor, Diamond Block, Lumbini Rockdale, Khairatabad, Hyderabad, Telangana`  <->  `Gujarat Consultancy Private Ltd | (no address)`
- [14 normalization-only (identical after normalization)] false merge p=0.81: S1 `Gujarat Consultancy Private Limited | 6-3-570/1 To &7, Unit 403, 4Th Floor, Diamond Block, Lumbini Rockdale, Khairatabad, Hyderabad, Telangana`  <->  `గుజరాత్ ఇన్వెస్ట్‌మెంట్స్ ప్రైవేట్ లిమిటెడ్ | DOOR NO 6-3-570/3 TO &7, KHAIRATABAD, HYDERABAD, Telangana`
- [14 normalization-only (identical after normalization)] missed true match p=0.52: S1 `Young Continental Sachs, LLC | 13 S Bearwood Drive, Fluvanna County, VA`  <->  `Young Continental Sachs, Llc | (no address)`
- [14 normalization-only (identical after normalization)] missed true match p=0.65: S1 `Super Healthcare Private Limited | Prof C N R Rao Block, Tumkur University, Tumkur, Karnataka`  <->  `ಸೂಪರ್ ಹೆಲ್ತ್‌ಕೇರ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್ | HN ##413 PROF C N R RAO BLCOK, TUMKUR UNIVERSITY, TUMKUR, Karnataka`
- [15 token-order (token-sort>=.95, plain ratio<.9)] missed true match p=0.32: S1 `New Business Private Limited | Groundfloor, Warehouse, Nea, R, Shivagiri, Byappanahalli, Bangalore, Karnataka, Bangalore South`  <->  `private new business limited | KA, Hn 866 Groundfloor, Bangalore`
- [15 token-order (token-sort>=.95, plain ratio<.9)] missed true match p=0.43: S1 `Shiva Developers Private Limited | Karnataka, Dharwad, Biotechnology Building, Kle Technological University, Bvb Campus, Vidyanagar, Hubli, Dharwar, Ground Floor`  <->  `Shiva Deve1opers Developers Private [Limited] | H.NO 275 GROUND FLOOR, BIOTECHNOLOGY BUILDING, KLE TECHNOLOGICAL UNIVERSITY, BVB CAMPUS, VIDYANAGAR, HUBLI, DHARWAD, ಕರ್ನಾಟಕ`
- [15 token-order (token-sort>=.95, plain ratio<.9)] false merge p=0.92: S1 `Shiva Developers Private Limited | Karnataka, Dharwad, Biotechnology Building, Kle Technological University, Bvb Campus, Vidyanagar, Hubli, Dharwar, Ground Floor`  <->  `ಬಾಂಬೆ ಕನ್ಸಲ್ಟೆಂಟ್ಸ್ ಲಿಮಿಟೆಡ್ | Kle Technological University Bvb Campus, Vidyanagar, Hubli, Dharwad, Dharwar, KA`
- [15 token-order (token-sort>=.95, plain ratio<.9)] false merge p=0.80: S1 `All Media Private Limited | Bangalore, Yarandahalli, Anekal, No. 80, Karnataka`  <->  `ಲಕ್ಷ್ಮಿ ಅಗ್ರೋ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್ | BANGALORE, Karnataka, # 80`
- [16 typo (name JW>=.9 but not identical)] missed true match p=0.21: S1 `Krishna Business | Prajay Princetion Towers, Unitno507 Plot No:/3And1/4Chitralayout, L.B Nagar, S, Telangana, Hyderabad, Aroornagar, 5Th Floor`  <->  `కృష్ణ బిజినెస్ | NO A-5TH FLOOR, HYDERABAD, AROORNAGAR, Andhra Pradesh`
- [16 typo (name JW>=.9 but not identical)] missed true match p=0.20: S1 `Reliable Packaging Partners | 5123 Meadowbrook Drive, Fort Worth, TX`  <->  `Reliable Packaging Partners Co | Texas, 5116 Meadowbrook Drive, Ft Worth`
- [16 typo (name JW>=.9 but not identical)] missed true match p=0.56: S1 `Grissel Shank, D.C. | 233 Indian Hill Trail, Glastonbury, CT`  <->  `Grissel Shank, D.C. Co | Indian Hill Trl, Glastonbury, Connecticut`
- [17 transliteration / non-Latin name] missed true match p=0.21: S1 `Krishna Business | Prajay Princetion Towers, Unitno507 Plot No:/3And1/4Chitralayout, L.B Nagar, S, Telangana, Hyderabad, Aroornagar, 5Th Floor`  <->  `కృష్ణ బిజినెస్ | NO A-5TH FLOOR, HYDERABAD, AROORNAGAR, Andhra Pradesh`
- [17 transliteration / non-Latin name] missed true match p=0.67: S1 `Super Impex Pvt Ltd | 8-2-269/10, #501, 5Th Floor, Trendset Towers, Road No:2 Banjara Hills, Hyderabad, Telangana`  <->  `సూపర్ ఇంపెక్స్ ప్రైవేట్ లిమిటెడ్ | 8-2-269/10, #501, 5TH FLOOR, TRENDSET TOWERS, ROAD NO:2 BANJARA HILLS, HYDERABAD, Andhra Pradesh`
- [17 transliteration / non-Latin name] missed true match p=0.33: S1 `Royal Services Private Limited | 56C Mirza Galib Street, Kolkata, Howrah, West Bengal`  <->  `রয়্যাল সার্ভিসেস প্রাইভেট লিমিটেড | 56C/2 MIRZA GALIB STREET, KOLKATA, HOWRAH, West Bengal`
- [22 ambiguous (true candidate AND a false candidate p>=.3)] missed true match p=0.21: S1 `Krishna Business | Prajay Princetion Towers, Unitno507 Plot No:/3And1/4Chitralayout, L.B Nagar, S, Telangana, Hyderabad, Aroornagar, 5Th Floor`  <->  `కృష్ణ బిజినెస్ | NO A-5TH FLOOR, HYDERABAD, AROORNAGAR, Andhra Pradesh`
- [22 ambiguous (true candidate AND a false candidate p>=.3)] missed true match p=0.67: S1 `Super Impex Pvt Ltd | 8-2-269/10, #501, 5Th Floor, Trendset Towers, Road No:2 Banjara Hills, Hyderabad, Telangana`  <->  `సూపర్ ఇంపెక్స్ ప్రైవేట్ లిమిటెడ్ | 8-2-269/10, #501, 5TH FLOOR, TRENDSET TOWERS, ROAD NO:2 BANJARA HILLS, HYDERABAD, Andhra Pradesh`
- [22 ambiguous (true candidate AND a false candidate p>=.3)] missed true match p=0.32: S1 `New Business Private Limited | Groundfloor, Warehouse, Nea, R, Shivagiri, Byappanahalli, Bangalore, Karnataka, Bangalore South`  <->  `private new business limited | KA, Hn 866 Groundfloor, Bangalore`