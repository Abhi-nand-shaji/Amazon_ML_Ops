# Edge-case diagnostics (validation entities)

Decision config: `{"mode": "global", "global_threshold": 0.73}`. Validation entities: 7,500; true pairs: 25,998; overall macro F0.5 = 0.9802.

## Failure taxonomy (all validation entities)

| type | count | share of true pairs |
|---|---|---|
| A blocking (true match not a candidate) | 894 | 3.44% |
| B matching: true candidate rejected | 435 | 1.67% |
| B matching: false merge on an entity with matches | 32 | 0.12% |
| C singleton with a predicted match | 1 | 0.00% |

## Per edge case

Entities can belong to several cases. Precision / recall are pooled over the case's entities; F0.5 is the macro mean.

| edge case | S1 entities | candidate recall | precision | recall | macro F0.5 | A blocking failures | B matching failures | C singleton failures |
|---|---|---|---|---|---|---|---|---|
| 1 true singleton | 412 |  | 0.000 |  | 0.998 | 0 | 0 | 1 |
| 2 one-to-one match | 420 | 0.971 | 0.993 | 0.948 | 0.944 | 12 | 13 | 0 |
| 3 multiple matches | 6668 | 0.966 | 0.999 | 0.949 | 0.981 | 882 | 454 | 0 |
| 4 cross-source match (S2 and S3) | 6043 | 0.966 | 0.999 | 0.950 | 0.982 | 812 | 416 | 0 |
| 5 multiple matches within one source | 5759 | 0.966 | 0.999 | 0.949 | 0.982 | 809 | 421 | 0 |
| 6 name-only strong (name>=.85, address weak/missing) | 1342 | 0.970 | 0.999 | 0.918 | 0.968 | 169 | 293 | 0 |
| 7 address-only strong (address>=.85, name weak/missing) | 601 | 0.965 | 0.999 | 0.946 | 0.983 | 89 | 53 | 0 |
| 8 name+address both noisy (both < .7) | 849 | 0.969 | 0.999 | 0.955 | 0.988 | 110 | 52 | 0 |
| 9 name collision (false candidate with name>=.9) | 2511 | 0.942 | 0.998 | 0.921 | 0.969 | 493 | 193 | 1 |
| 10 address collision (false candidate with address>=.9) | 2047 | 0.964 | 0.998 | 0.947 | 0.980 | 244 | 130 | 0 |
| 11 missing name (either side) | 0 |  |  |  |  |  |  |  |
| 12 missing address (either side) | 865 | 0.976 | 0.999 | 0.879 | 0.949 | 86 | 356 | 0 |
| 13 multiple missing fields | 0 |  |  |  |  |  |  |  |
| 14 normalization-only (identical after normalization) | 4656 | 0.975 | 0.999 | 0.957 | 0.986 | 471 | 342 | 0 |
| 15 token-order (token-sort>=.95, plain ratio<.9) | 1429 | 0.976 | 0.999 | 0.964 | 0.990 | 147 | 77 | 0 |
| 16 typo (name JW>=.9 but not identical) | 5454 | 0.975 | 0.999 | 0.958 | 0.987 | 525 | 391 | 0 |
| 17 transliteration / non-Latin name | 821 | 0.930 | 1.000 | 0.915 | 0.976 | 219 | 45 | 0 |
| 22 ambiguous (true candidate AND a false candidate p>=.3) | 203 | 0.968 | 0.955 | 0.947 | 0.954 | 23 | 47 | 0 |

## Source 2 vs Source 3

| source | true pairs in candidates | mean name similarity (true pairs) | mean address similarity (true pairs) | address missing | non-Latin name | name identical after normalization | precision | recall (of candidate true pairs) |
|---|---|---|---|---|---|---|---|---|
| S2 | 12079 | 0.837 | 0.849 | 0.037 | 0.077 | 0.294 | 0.998 | 0.983 |
| S3 | 13025 | 0.836 | 0.775 | 0.036 | 0.042 | 0.299 | 0.999 | 0.982 |

## Not measurable / structural cases

- 18 country conflict: 0 of 7,638,365 training true pairs cross a country boundary and candidates are generated per country partition, so no cross-country candidate exists; country mismatch is treated as a hard partition because the data prove it safe.
- 19 unseen country (France): test only; `country` is not a model feature and partitions are formed from whatever labels occur. See `inference_summary.json` for per-country prediction statistics.
- 20 true match missed by blocking: 894 pairs (row A above); 21 true candidate rejected: 435 pairs (row B).
- 23 duplicate candidates: 0 duplicated (S1, candidate) rows in the validation candidate set (candidates are the union of several rules, deduplicated by construction).
- 24 empty ground truth parsed as zero matches: 412 validation singletons.

## D. Output / pipeline checks on output/*.tsv (cases 25-27)

- rows: matching 1,732,544, candidate 1,732,544; duplicate S1 rows: 0 / 0
- S1/other-source ids inside predicted lists: 0 entities
- duplicate ids inside a predicted list: 0 entities; inside a candidate list: 0 entities
- subset invariant (every match is a candidate) violated for 0 entities

## Representative examples (worst entities of each case)

- [1 true singleton] false merge p=0.99: S1 `Lyra Inc | IL, Oak Park, 1445 Harlem Avenue, Unit # A`  <->  `LYRASYN | HARLEM AVENUE, IL, OAK PARK`
- [2 one-to-one match] missed true match p=0.05: S1 `Dee's Petroleum | 101 Chaney Avenue, Unit 1/2, Jacksonville, NC`  <->  `HAL0KORYUMA | 0328 CHANEY AVE, JACKSONVILLE, NC`
- [3 multiple matches] missed true match p=0.16: S1 `Karishma Finance Private Limited | Building No 1085/13 Mulamoottil Building, Kozhenchery, Pathanamthitta, Kerala`  <->  `Karishma Finance | (no address)`
- [3 multiple matches] missed true match p=0.17: S1 `Dermatology Desert Center LLC | 750 Elm Grove Road, City Of Brookfield, WI`  <->  `Dermatology Desert  Center LLC | (no address)`
- [4 cross-source match (S2 and S3)] missed true match p=0.17: S1 `Dermatology Desert Center LLC | 750 Elm Grove Road, City Of Brookfield, WI`  <->  `Dermatology Desert  Center LLC | (no address)`
- [6 name-only strong (name>=.85, address weak/missing)] missed true match p=0.02: S1 `Ridge Association | 5544 Shady Trail, TN, Old Hickory`  <->  `Ridge Association Corp | (no address)`
- [6 name-only strong (name>=.85, address weak/missing)] missed true match p=0.17: S1 `Dermatology Desert Center LLC | 750 Elm Grove Road, City Of Brookfield, WI`  <->  `Dermatology Desert  Center LLC | (no address)`
- [6 name-only strong (name>=.85, address weak/missing)] missed true match p=0.20: S1 `Bren Coffman, DDS | 511 Walnut Street, OH, Leetonia`  <->  `Bren Coffman, | (no address)`
- [7 address-only strong (address>=.85, name weak/missing)] missed true match p=0.26: S1 `North Global Private Limited | 105, 1St Floor, Plot 15D, C Wing, Kalpak Estate, Shaikh Misree Road, Mumbai, Maharashtra`  <->  `Gildpyra | 105, 1St Floor, Plot 15D, C Wing, Kalpak Estate, Shaikh Misree Road, Mumbai, Maharashtra`
- [7 address-only strong (address>=.85, name weak/missing)] missed true match p=0.11: S1 `Galaxy International Pvt Ltd | 4-10/2, Annapurna Nilayam, Satrampadu, Eluru, West Godavari, Andhra Pradesh`  <->  `Galaxy International Pvt [Ltd] | (no address)`
- [8 name+address both noisy (both < .7)] missed true match p=0.05: S1 `Krishna Business | Prajay Princetion Towers, Unitno507 Plot No:/3And1/4Chitralayout, L.B Nagar, S, Telangana, Hyderabad, Aroornagar, 5Th Floor`  <->  `KRISHNA BUSINESS Enterprises | (no address)`
- [9 name collision (false candidate with name>=.9)] missed true match p=0.04: S1 `One Enterprises Private Limited | Beside Sri G M Siddaramanna Girls Hostel, Panduranganagar, Chickpete, Tumkur, Karnataka`  <->  `One Enterprises Prívate Limited | (no address)`
- [9 name collision (false candidate with name>=.9)] missed true match p=0.06: S1 `Economic Development Coalition Corp | 357 Yale Avenue, Baltimore, MD`  <->  `ECONOMIC DÉVELOPMENT COALITION CORP | (no address)`
- [12 missing address (either side)] missed true match p=0.04: S1 `One Enterprises Private Limited | Beside Sri G M Siddaramanna Girls Hostel, Panduranganagar, Chickpete, Tumkur, Karnataka`  <->  `One Enterprises Prívate Limited | (no address)`
- [12 missing address (either side)] missed true match p=0.09: S1 `Main Street Cafe Inc. | 16356 Thompson Peak Parkway, Unit APT 2091, Scottsdale, AZ`  <->  `MAIN STREET CAFE INC. Enterprises | (no address)`
- [12 missing address (either side)] missed true match p=0.20: S1 `Bren Coffman, DDS | 511 Walnut Street, OH, Leetonia`  <->  `Bren Coffman, | (no address)`
- [14 normalization-only (identical after normalization)] missed true match p=0.60: S1 `Falls Institutions Corp | 900 Locust Road, Pottsboro, TX`  <->  `falls institutions corp | (no address)`
- [14 normalization-only (identical after normalization)] missed true match p=0.43: S1 `Young Continental Sachs, LLC | 13 S Bearwood Drive, Fluvanna County, VA`  <->  `Young Continental Sachs, Llc | (no address)`
- [14 normalization-only (identical after normalization)] missed true match p=0.17: S1 `Dermatology Desert Center LLC | 750 Elm Grove Road, City Of Brookfield, WI`  <->  `Dermatology Desert  Center LLC | (no address)`
- [15 token-order (token-sort>=.95, plain ratio<.9)] false merge p=1.00: S1 `Better Patriot Entertainment LLC | 7525 Cameron Drive, Peoria, AZ`  <->  `Better  Patriot Entertainment LLC | (no address)`
- [16 typo (name JW>=.9 but not identical)] missed true match p=0.65: S1 `Regional Retail Industries | 2607 Garfield Avenue, Des Moines, IA`  <->  `REGIONAL RETAIL INDUSTRIES INC | DES MOINES TOWNSHIP, IA, 260 GARFIELD AVE`
- [16 typo (name JW>=.9 but not identical)] missed true match p=0.09: S1 `Main Street Cafe Inc. | 16356 Thompson Peak Parkway, Unit APT 2091, Scottsdale, AZ`  <->  `MAIN STREET CAFE INC. Enterprises | (no address)`
- [16 typo (name JW>=.9 but not identical)] missed true match p=0.00: S1 `Reliable Packaging Partners | 5123 Meadowbrook Drive, Fort Worth, TX`  <->  `Reliable Packaging Partners Corp | 5116 Meadowbrook Drive, Fort Worth, Texas`
- [17 transliteration / non-Latin name] missed true match p=0.11: S1 `Galaxy International Pvt Ltd | 4-10/2, Annapurna Nilayam, Satrampadu, Eluru, West Godavari, Andhra Pradesh`  <->  `Galaxy International Pvt [Ltd] | (no address)`
- [17 transliteration / non-Latin name] missed true match p=0.05: S1 `Krishna Business | Prajay Princetion Towers, Unitno507 Plot No:/3And1/4Chitralayout, L.B Nagar, S, Telangana, Hyderabad, Aroornagar, 5Th Floor`  <->  `KRISHNA BUSINESS Enterprises | (no address)`
- [17 transliteration / non-Latin name] missed true match p=0.05: S1 `Raj Software Private Limited | Building No.18, Vidya Colony, Behind Court, Camp, Amravati, Maharashtra`  <->  `Raj Software Limited Services | (no address)`
- [22 ambiguous (true candidate AND a false candidate p>=.3)] false merge p=0.90: S1 `Family Program | 376 1/2 Colorado Street, Fl 1, Chandler, AZ`  <->  `Avilum Program | 376 1/2 Colorado St, Floor 1, Chandler, Arizona`
- [22 ambiguous (true candidate AND a false candidate p>=.3)] false merge p=0.95: S1 `Andheri East Services Limited | 24-10/12/2013Marolmaroshi Rd, Gaondevi, Talao, Near, Mapkanschool, Andheri East, Mumbai City, Maharashtra`  <->  `Andheri East  Services | (no address)`
- [22 ambiguous (true candidate AND a false candidate p>=.3)] false merge p=0.81: S1 `Secure Diamond Imperial, L.L.C. | 1049 Woodsia Way, Unit 205, Oak Island, NC`  <->  `Secure Diamond Irpria, [L.L.C.] | (no address)`