# Data Analysis Report

Generated: 2026-09-25 02:53:07


## 1. Per-source profile (TRAIN)


### train_source1

- rows: 2,206,821
- unique entity_id: 2,206,821 (is_unique=True)
- null business_name: 0; empty-string: 0
- null business_address: 0; empty-string: 0
- null country: 0
- country distribution: {'US': 1323633, 'India': 883188}
- name length: mean=24.0 median=24.0 p90=34.0 max=105
- address length: mean=52.1 median=41.0 p90=90.0 max=256
- non-ASCII business_name: 0 (0.00%)
- exact duplicate (name,address) rows: 0 (0.00%)
- address contains a 5-digit token: 6.60%; 6-digit token: 0.08%; starts with digits: 61.89%

### train_source2

- rows: 5,034,616
- unique entity_id: 5,034,616 (is_unique=True)
- null business_name: 2; empty-string: 2
- null business_address: 168,967; empty-string: 168,967
- null country: 0
- country distribution: {'US': 3016817, 'India': 2017799}
- name length: mean=25.1 median=25.0 p90=37.0 max=104
- address length: mean=46.2 median=37.0 p90=83.0 max=249
- non-ASCII business_name: 764,608 (15.19%)
- exact duplicate (name,address) rows: 25,891 (0.51%)
- address contains a 5-digit token: 6.50%; 6-digit token: 0.83%; starts with digits: 51.00%

### train_source3

- rows: 5,285,603
- unique entity_id: 5,285,603 (is_unique=True)
- null business_name: 13; empty-string: 13
- null business_address: 175,916; empty-string: 175,916
- null country: 0
- country distribution: {'US': 3170056, 'India': 2115547}
- name length: mean=25.2 median=25.0 p90=37.0 max=123
- address length: mean=46.7 median=42.0 p90=77.0 max=240
- non-ASCII business_name: 606,737 (11.48%)
- exact duplicate (name,address) rows: 18,881 (0.36%)
- address contains a 5-digit token: 6.50%; 6-digit token: 0.80%; starts with digits: 51.19%

## 2. Per-source profile (TEST)


### test_source1

- rows: 1,732,544
- unique entity_id: 1,732,544 (is_unique=True)
- null business_name: 0; empty-string: 0
- null business_address: 0; empty-string: 0
- null country: 0
- country distribution: {'India': 809986, 'US': 663106, 'France': 259452}
- name length: mean=23.8 median=24.0 p90=34.0 max=92
- address length: mean=57.2 median=50.0 p90=93.0 max=268
- non-ASCII business_name: 40,789 (2.35%)
- exact duplicate (name,address) rows: 0 (0.00%)
- address contains a 5-digit token: 4.32%; 6-digit token: 0.05%; starts with digits: 57.82%

### test_source2

- rows: 4,887,273
- unique entity_id: 4,887,273 (is_unique=True)
- null business_name: 46; empty-string: 46
- null business_address: 129,408; empty-string: 129,408
- null country: 0
- country distribution: {'India': 2312565, 'US': 1871330, 'France': 703378}
- name length: mean=25.7 median=25.0 p90=38.0 max=102
- address length: mean=50.4 median=43.0 p90=87.0 max=269
- non-ASCII business_name: 928,158 (18.99%)
- exact duplicate (name,address) rows: 22,642 (0.46%)
- address contains a 5-digit token: 4.52%; 6-digit token: 0.55%; starts with digits: 46.87%

### test_source3

- rows: 5,082,316
- unique entity_id: 5,082,316 (is_unique=True)
- null business_name: 59; empty-string: 59
- null business_address: 136,098; empty-string: 136,098
- null country: 0
- country distribution: {'India': 2405000, 'US': 1945701, 'France': 731615}
- name length: mean=25.7 median=25.0 p90=38.0 max=103
- address length: mean=48.7 median=43.0 p90=81.0 max=267
- non-ASCII business_name: 737,515 (14.51%)
- exact duplicate (name,address) rows: 16,305 (0.32%)
- address contains a 5-digit token: 4.51%; 6-digit token: 0.53%; starts with digits: 46.93%

## 3. Train vs test distribution differences

- source1 train country%: {'US': 59.98, 'India': 40.02}
- source1 test  country%: {'India': 46.75, 'US': 38.27, 'France': 14.98}
- source2 train country%: {'US': 59.92, 'India': 40.08}
- source2 test  country%: {'India': 47.32, 'US': 38.29, 'France': 14.39}
- source3 train country%: {'US': 59.98, 'India': 40.02}
- source3 test  country%: {'India': 47.32, 'US': 38.28, 'France': 14.4}

## 4. Ground truth analysis

- rows: 2,206,821; unique source1_entity_id: 2,206,821
- singleton (no match) S1 entities: 123,247 (5.58%)
- match count distribution:
```
matched_entity_ids
0     123247
1     119157
2     375212
3     530841
4     484115
5     321957
6     164868
7      63968
8      18680
9       4205
10       534
11        37
```
- mean matches/entity (incl singletons): 3.461; mean given >=1 match: 3.666; max: 11
- S1 with S2-only matches: 143,029
- S1 with S3-only matches: 164,498
- S1 with BOTH S2 and S3 matches: 1,776,047
- S1 with multiple S2 matches (>1): 1,129,968; multiple S3 matches (>1): 1,224,128
- total positive pairs: 7,638,365 (S2 side: 3,693,619, S3 side: 3,944,746)
- distinct S2 ids used as a true match: 3,693,619 / 5,034,616 rows (73.36%)
- distinct S3 ids used as a true match: 3,944,746 / 5,285,603 rows (74.63%)

## 5. True-pair similarity analysis (join gt with record fields)

- exact RAW name match among true pairs: 4.64%
- exact NORMALIZED name match among true pairs: 21.85%
- exact NORMALIZED address match among true pairs: 8.28%
- country match among true pairs: 100.0000% (mismatches: 0)
- (sampled 200,000) name token-Jaccard quantiles: [0.0, 0.0, 0.5, 0.667, 1.0, 1.0, 1.0]
- (sampled 200,000) addr token-Jaccard quantiles: [0.0, 0.25, 0.429, 0.667, 0.818, 1.0, 1.0]
- frac true pairs with name_jaccard==0 (sampled): 14.36%
- frac true pairs with addr_jaccard==0 (sampled): 4.46%
- frac true pairs with BOTH name_jaccard<0.2 AND addr_jaccard<0.2 (sampled): 0.33%
  - [S2 true pairs] name_jac median=0.667, addr_jac median=0.714, n=96,444
  - [S3 true pairs] name_jac median=0.667, addr_jac median=0.500, n=103,556
- true pairs where exactly one side's name is non-ASCII (possible transliteration case): 1,060,781 (13.888%)
- true pairs where BOTH sides' name are non-ASCII: 0 (0.000%)

## 6. Source2 vs Source3 noise comparison (independent, not just via true pairs)

- S2: legal-suffix-token presence in name: 47.21%
- S3: legal-suffix-token presence in name: 49.54%

_EDA total runtime: 579.1s_
