# Blocking (Candidate Generation) Report - TRAIN sample

Generated 2026-09-25 23:17:18

Configuration: `{"df_cap": 300, "k_rare_name": 3, "k_rare_addr": 4, "exact_key_cap": 300, "max_candidates": 30, "expansion_budget": 6000000, "cap_addr_bigram": 300, "cap_name_bigram": 300, "prefix_lens": [8], "cap_prefix": 300, "rescore": true, "dense_width_name": 8, "dense_width_addr": 16}`; S1 sample = 50000 (seed 42)

Candidate filter: `{"model": "6663d8f251ca", "min_score": 0.015, "max_candidates": 30, "scores": "out-of-fold"}`

## Volume

- S1 entities blocked: 50,000; true singletons among them: 2,849 (5.70%)
- candidate pairs: 304,054; per entity mean=6.08 median=6 p90=9 max=25
- entities with ZERO candidates: 38 (0.076%)
- candidates by source: {'S3': np.int64(154749), 'S2': np.int64(149305)}
- candidate reduction ratio vs. same-country cross product: 0.999999 (304,054 of 268,595,513,739 possible same-country pairs kept)

## Recall (fraction of true (S1, match) pairs present in the candidate set)

- true pairs among sampled entities: 172,575
- **overall candidate recall: 96.549%**
  - S2: 96.546% (n=83,349)
  - S3: 96.553% (n=89,226)
  - country India: 94.768% (n=68,504)
  - country US: 97.722% (n=104,071)
- entities whose EVERY true match is a candidate: 89.97% of entities with >=1 true match

### Step by step (share of ALL true pairs of the sampled entities that survive)

| step | candidates / entity | recall |
|---|---|---|
| 1. retrieval (inverted indices) + learned ranker, shortlist of the best 30 | 29.78 | 96.978% |
| 2. candidate filter (probability >= 0.015) = **final candidate set** | 6.08 | 96.549% |

### Shortlist recall if the ranker cut were k

| k | recall | mean shortlisted/entity |
|---|---|---|
| 5 | 88.859% | 5.0 |
| 10 | 96.010% | 10.0 |
| 15 | 96.601% | 15.0 |
| 20 | 96.810% | 19.9 |
| 30 | 96.978% | 29.8 |

### Candidate filter: size / recall trade-off (out-of-fold probabilities, all sampled entities)

| probability floor | candidates / entity | recall of all true pairs | entities without candidates | true singletons without candidates |
|---|---|---|---|---|
| 0 (no filter) | 29.78 | 96.978% | 0.01% | 0.1% |
| 0.001 | 10.99 | 96.939% | 0.01% | 0.1% |
| 0.002 | 9.18 | 96.896% | 0.01% | 0.1% |
| 0.003 | 8.36 | 96.867% | 0.01% | 0.1% |
| 0.005 | 7.51 | 96.809% | 0.01% | 0.1% |
| 0.0075 | 6.94 | 96.746% | 0.02% | 0.3% |
| 0.01 | 6.58 | 96.679% | 0.04% | 0.6% |
| 0.0125 | 6.31 | 96.612% | 0.06% | 1.0% |
| 0.015 **(chosen)** | 6.08 | 96.549% | 0.08% | 1.3% |
| 0.02 | 5.72 | 96.387% | 0.12% | 2.0% |
| 0.025 | 5.45 | 96.236% | 0.17% | 2.8% |
| 0.03 | 5.25 | 96.118% | 0.23% | 3.8% |
| 0.04 | 4.94 | 95.889% | 0.39% | 6.3% |
| 0.05 | 4.73 | 95.677% | 0.56% | 9.2% |

Floor chosen on the matcher's validation entities: the largest grid value that loses at most 0.5% of their shortlisted true pairs. Filter features by gain: `block_rank` 61.6%, `ov_addr` 11.3%, `addr_token_set` 4.2%, `core_ratio` 3.6%, `name_partial` 3.4%, `ov_name` 3.3%, `n_strategies` 2.3%, `housenum_match` 2.2%.

### Contribution of each blocking rule (recall of true pairs carrying that rule's evidence)

- rare NAME-token rule: 35.455%
- rare ADDRESS-token rule: 48.856%
- ADDRESS-bigram rule: 85.632%; NAME-bigram rule: 66.191%; name-prefix rule: 47.193%
- exact normalized name: 28.426%; exact compact name: 29.540%; postal equal: 5.044%
- found ONLY via name rule: 11.538%
- found ONLY via address rule: 20.942%
- found by exactly one rule (fragile): 10.361%; by >=2 rules: 86.189%
- among FOUND true pairs, rank by block score: median=1, p90=4, p99=7

## Why true matches are missed (5,955 pairs = 3.451%)

Lost at step 1 (never shortlisted): 5,216 pairs; removed by the candidate filter (step 2): 739 pairs.

| category of missed pair | count | share of misses | of which removed by the filter |
|---|---|---|---|
| non-Latin/accented name (name tokens can't match) | 2,276 | 38.2% | 198 |
| shares name+address tokens but dropped (rare-token selection / cap / df_cap) | 1,573 | 26.4% | 109 |
| candidate address missing (only name tokens available) | 1,138 | 19.1% | 255 |
| no shared name token (address tokens shared) | 899 | 15.1% | 157 |
| non-Latin/accented name AND missing address | 65 | 1.1% | 18 |
| no shared address token (name tokens shared) | 4 | 0.1% | 2 |

Missed pairs by source: {'S3': np.int64(3076), 'S2': np.int64(2879)}; by country: {'India': np.int64(3584), 'US': np.int64(2371)}

Examples (S1 name | S1 address  -->  missed candidate name | address):

- [candidate address missing (only na] `Oncology Physicians of Columbus` | `1530 Genessee Avenue, Unit C, Columbus, OH`  -->  `Oncology Physicians of Columbus Ltd` | ``
- [candidate address missing (only na] `Solutions Indian Export Limited` | `Flat No 201 2Nd Floor Khasra No.104, Village Gijhore, Noida, Gautam Buddha Nagar, Uttar Pradesh`  -->  `Solutions Indian Limited Services` | ``
- [non-Latin/accented name AND missin] `Angel Welfare Society` | `H1-309, Jasmine Grove, Khasra No. 959, Mehrauli, Ghaziabad, Uttar Pradesh`  -->  `Angel Society Wélfare` | ``
- [shares name+address tokens but dro] `Krishna Business` | `B-36 1St Floor Panchsheel Enclave, New Delhi, South Delhi, Delhi`  -->  `Krishna Bslses` | `B-3-6 1ST FLOOR PANCHSHEEL ENCLAVE, NEW DELHI, Delhi`
- [non-Latin/accented name (name toke] `Dream Estate Private Limited` | `No.21/2, 1St Floor, Pratibha Complex, Uttaradi Mutt Road, Shankarapuram, Bangalore, Karnataka`  -->  `ಡ್ರೀಮ್ ಎಸ್ಟೇಟ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್` | `NO.21, BANGALORE, Karnataka`
- [no shared name token (address toke] `Prime Developers Private Limited` | `C 13, First Floor, Gali No 52 Chanakya Place, New Delhi, West Delhi, Delhi`  -->  `dpprivate.com` | `C 13, New Delhi, DL`
- [shares name+address tokens but dro] `Roberts and Merritt` | `1739 Heather Lane, Frederick, MD`  -->  `Roberts &-Meirtt` | `1740 Heather Ln, Frederick, Maryland`
- [shares name+address tokens but dro] `Gold Consulting Private Limited` | `30 Rafi Ahmed Kidwai Road, Kolkata, Howrah, West Bengal`  -->  `Gold Cosuilting Private Limited` | `West Bengal, 1  RAFI AHMED KIDWAI ROAD, KOLKATA, KOLKATA`
- [non-Latin/accented name (name toke] `Tech Foods Private Limited` | `A/804, Donum Dei, No 1 Off Kanakia, Thane, Maharashtra`  -->  `टेक फूड्स प्राइवेट लिमिटेड` | `A/804, THANE, महाराष्ट्र`
- [non-Latin/accented name (name toke] `White Finance Pvt Ltd` | `B 17, Sector 55 Noida, Lorik Yadav, Noida, Gautam Buddha Nagar, Uttar Pradesh`  -->  `व्हाइट फाइनेंस प्रा. लि.` | `B 17, NOIDA, GAUTAM BUDDHA NAGAR, Uttar Pradesh`
- [shares name+address tokens but dro] `Mumbai Infratek Private Limited` | `5, Rameshwar Rifle Range, L.B.S. Marg, Near North Bombay School, Mumbai, Mumbai City, Maharashtra`  -->  `Mumbai Itrek Private Limited` | `Mumbai, Mumbai City, MH, 5`
- [candidate address missing (only na] `Vision Specialists PC` | `444 Fordham Place, Roselle, IL`  -->  `Vision Specialists Industries` | ``
- [no shared name token (address toke] `Shrivision Vidyalaya` | `Apt 203, Sterling Shalom, Kundalahalli, Brookfields, Bangalore, Bangalore, Karnataka`  -->  `vidyalayashrivision.com` | `Apt 20, Bangalore, KA`
- [candidate address missing (only na] `Asn Brothers Pvt Ltd` | `House No.2/46, Ramadevi, Chauraha, Ring Road, Rama Devi Chauraha, Kanpur Nagar, Uttar Pradesh`  -->  `Asn Bfhofers Pvt Ltd` | ``
- [shares name+address tokens but dro] `Blue Cleaning Service` | `105 Main Street, Sanford, NC`  -->  `Blue Cleaning` | `05 MAIN ST, SANFORD, NC`
- [shares name+address tokens but dro] `Community Chiropractic LLC` | `419 Sycamore Avenue, NM, Roswell`  -->  `Community Chiorpfacitc LLC` | `19 Sycamore Ave, Roswell, New Mexico`
- [non-Latin/accented name (name toke] `Sky Consulting Private Limited` | `New No .188, Old No183 Chaturyana, Jhansi, Uttar Pradesh`  -->  `स्काई कंसल्टिंग प्राइवेट लिमिटेड` | `New No .188, Jhansi, UP`
- [candidate address missing (only na] `Atlantic Family Office LLC` | `11204 Highway 88, Maury City, TN`  -->  `Atlantic Family Trading` | ``
- [non-Latin/accented name (name toke] `North Business Private Limited` | `H.No. 831, Khasra No. 27/19/20, Behind Fun Food Village, Kapashera, New Delhi, New Delhi, South West Delhi, Delhi`  -->  `नॉर्थ बिजनेस प्राइवेट लिमिटेड` | `Doer No 83, New Delhi, South West Delhi, दिल्ली`
- [shares name+address tokens but dro] `West Trading` | `Shop 2, 3 S Y No 15, Kolan Raja Reddy Heights, Near Nizampet Panchayat Office, Qutubullapur, Hyderabad, Telangana`  -->  `West Trading Trading` | `Shop 2, Ghmc M Corp Og, TG`
- [shares name+address tokens but dro] `Morgan Media Inc` | `301 Adams Street, Unit C, Annapolis, MD`  -->  `Morgan Inc  Services` | `01 ADAMS STREET, ANNAPOLIS, MD`
- [non-Latin/accented name (name toke] `Sai Healthcare Limited` | `South West Delhi, Tyagi Enclave, South West Delhi, Delhi, House No A-1`  -->  `साईं हेल्थकेयर लिमिटेड` | `Delhi, SOUTH WEST DELHI, HOUSE NO A-1, SOUTH WEST DELHI`
- [candidate address missing (only na] `Producer Girraj Construction Private Limited` | `4-Da, Dedna 4Th Floor97 Queen'S Above Roop Mills, Mumbai, Maharashtra`  -->  `Producer  Gonrraj Construction Private Limited` | ``
- [non-Latin/accented name (name toke] `Balaji Consultancy Private Limited` | `Plot No.74, Pioneer Residency Park, Somalwada Wardha Road, Nagpur, Maharashtra`  -->  `बालाजी कंसल्टेंसी प्राइवेट लिमिटेड` | `H.NO 74, NAGPUR, Maharashtra`
- [non-Latin/accented name (name toke] `Shivam Media Private Limited` | `Office No. 2, Ward No. 16 Deendayal Nagar, Kanpur Dehat, Uttar Pradesh`  -->  `शिवम मीडिया प्राइवेट लिमिटेड` | `OFFICE NO. 2, KANPUR DEHAT, उत्तर प्रदेश`
- [non-Latin/accented name (name toke] `My Infrastructure` | `Phase - 1, Near Raj Public School Aya Nagar, Delhi, New Delhi, B-29`  -->  `माय इंफ्रास्ट्रक्चर` | `B-29, NEW DELHI, Delhi`
- [candidate address missing (only na] `Jai Management Private Limited` | `1864 J, Trichy Road, Vasantham Colony, Ramanathapuram, Coimbatore, Tamil Nadu`  -->  `Mr Jai Management` | ``
- [non-Latin/accented name (name toke] `Jain Industries` | `No.101, 1St Floor, Vyshak Center, No.1027, 24Th Main, 11Th Cross, Sector 1, Hsr Lay, Out, Bangalore, Karnataka`  -->  `ಜೈನ್ ಇಂಡಸ್ಟ್ರೀಸ್` | `00101, BANGALORE, AGARA, Karnataka`
- [non-Latin/accented name (name toke] `Red Foundation Private Limited` | `A 1903, Majaswadi Sarvoday Nagar Chs Ltd, Majas Village, Sarvodaya Nagar, Jogeshwari (E), Mumbai, Mumbai, Mumbai City, Maharashtra`  -->  `रेड फाउंडेशन प्राइवेट लिमिटेड` | `A 1902, Vasai, MH`
- [no shared name token (address toke] `Equinox` | `10835 Pitch Circle, MD, Monrovia`  -->  `Yumakelo` | `0835 Pitch Cir, PO Box 5884, Monrvia, Maryland`
- [candidate address missing (only na] `Wildlife Fellowship` | `1608 Dunmore Loop, Crossett, AR`  -->  `Wildlife Fhelglowfship` | ``
- [non-Latin/accented name (name toke] `Dream Trading Private Limited` | `Flat No 1, Vini Apartment, S No 2093, Pune City, Pune, Maharashtra`  -->  `ड्रीम ट्रेडिंग प्राइवेट लिमिटेड` | `FLAT NO ##1, PUNE CITY, PUNE, महाराष्ट्र`
- [non-Latin/accented name (name toke] `Laxmi City Producer Private Limited` | `47-A, Karnataka, Bangalore, Cp Residency, I Floor, 9Th Main, 1St Cross, Hal Iii Stage, Indira Nagar Opp Bsnl Tel, Exchange`  -->  `ಲಕ್ಷ್ಮಿ ಸಿಟಿ ಪ್ರೊಡ್ಯೂಸರ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್` | `47-A, Bangalore, KA`
- [non-Latin/accented name (name toke] `All Consultancy` | `27, Sri Nilaya, Iii Floor, Between 10Th & 11Th Cross Margosa Road, Malleswaram, Bangalore, Karnataka`  -->  `ಆಲ್ ಕನ್ಸಲ್ಟೆನ್ಸಿ` | `BANGALORE, 27, BANGALORE, Karnataka`
- [shares name+address tokens but dro] `Orthopedic Health LLC` | `1556 Wayne Street, Toledo, OH`  -->  `Orthopedic (Health)` | `1563 Wayne Street, Tooledo CITY, Ohio`
- [candidate address missing (only na] `Vision Health` | `3303 Archibald Avenue, Unit UNIT 127, Ontario, CA`  -->  `Vision [Health]` | ``
- [non-Latin/accented name (name toke] `Surya Solutions Limited` | `S-5, Second Floor 11/41, West Punjabi Bagh, Delhi, New Delhi, Delhi`  -->  `सूर्य सॉल्यूशंस लिमिटेड` | `S-5, NEW DELHI, Delhi`
- [non-Latin/accented name (name toke] `Krishna Services Limited` | `Office No. 25, 1St Floor, B/H Real Plaza 1 Village: Lalpar, Morbi, Rajkot, Gujarat`  -->  `કૃષ્ણ સર્વિસીસ લિમિટેડ` | `Ofnice No. 25, Morbi, Rajkot, GJ`
- [shares name+address tokens but dro] `Rockwell & Brothers` | `No 1503/27, Room No. 104, 40Th Cross, Ground Floor, 4Th ÂTâ Block, Jayanagar, Bengaluru, Bangalore, Karnataka`  -->  `R0ckwell & Brothers` | `Bangalore, Bengaluru, ಕರ್ನಾಟಕ`
- [no shared name token (address toke] `Davidson Select Hawaii Inc` | `4914 3, Mexico, NY`  -->  `davidsonselecthawaii.com` | `914 3, Centrral Square, New York`