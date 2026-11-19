# Data Dictionary

> Column definitions for `applicant_base.parquet` and `eval_attrs.parquet`.

---

## `applicant_base.parquet`

One row per applicant (`SK_ID_CURR`). Built from `application_train.csv` with
presence flags joined from bureau and lender tables.

### Identity & Target

| Column | Type | Meaning | Source |
|--------|------|---------|--------|
| `SK_ID_CURR` | int64 | Unique applicant ID (primary key) | application_train |
| `TARGET` | int64 | Default flag: 1 = defaulted, 0 = repaid | application_train |

### Contract & Demographics

| Column | Type | Meaning | Source |
|--------|------|---------|--------|
| `NAME_CONTRACT_TYPE` | str | Cash loans / Revolving loans | application_train |
| `CODE_GENDER` | str | Gender: M, F, XNA | application_train |
| `FLAG_OWN_CAR` | str | Owns a car: Y / N | application_train |
| `FLAG_OWN_REALTY` | str | Owns real estate: Y / N | application_train |
| `CNT_CHILDREN` | int64 | Number of children | application_train |
| `NAME_TYPE_SUITE` | str | Who accompanied the applicant | application_train |
| `NAME_INCOME_TYPE` | str | Income category (Working, Pensioner, etc.) | application_train |
| `NAME_EDUCATION_TYPE` | str | Education level | application_train |
| `NAME_FAMILY_STATUS` | str | Marital / family status | application_train |
| `NAME_HOUSING_TYPE` | str | Housing situation | application_train |
| `OCCUPATION_TYPE` | str | Occupation (high missingness) | application_train |
| `ORGANIZATION_TYPE` | str | Employer organisation type | application_train |
| `CNT_FAM_MEMBERS` | float64 | Family size | application_train |

### Financial Amounts

| Column | Type | Meaning | Source |
|--------|------|---------|--------|
| `AMT_INCOME_TOTAL` | float64 | Total annual income | application_train |
| `AMT_CREDIT` | float64 | Credit amount of the loan | application_train |
| `AMT_ANNUITY` | float64 | Loan annuity (periodic payment) | application_train |
| `AMT_GOODS_PRICE` | float64 | Price of the goods for which the loan is given | application_train |

### Time-Related (days before application, negative)

| Column | Type | Meaning | Source |
|--------|------|---------|--------|
| `DAYS_BIRTH` | int64 | Age in days (negative; use `abs()/365.25` for years) | application_train |
| `DAYS_EMPLOYED` | float64 | Days employed before application (sentinel 365243 → NaN) | application_train |
| `DAYS_REGISTRATION` | float64 | Days since registration | application_train |
| `DAYS_ID_PUBLISH` | int64 | Days since ID document was published | application_train |
| `OWN_CAR_AGE` | float64 | Age of car in years (NaN if no car) | application_train |
| `DAYS_LAST_PHONE_CHANGE` | float64 | Days since last phone change | application_train |

### Region & Address Flags

| Column | Type | Meaning | Source |
|--------|------|---------|--------|
| `REGION_POPULATION_RELATIVE` | float64 | Normalised population of applicant's region | application_train |
| `REGION_RATING_CLIENT` | int64 | Region rating (1–3) | application_train |
| `REGION_RATING_CLIENT_W_CITY` | int64 | Region rating (city-weighted) | application_train |
| `REG_REGION_NOT_LIVE_REGION` | int64 | Registration ≠ living region | application_train |
| `REG_REGION_NOT_WORK_REGION` | int64 | Registration ≠ work region | application_train |
| `LIVE_REGION_NOT_WORK_REGION` | int64 | Living ≠ work region | application_train |
| `REG_CITY_NOT_LIVE_CITY` | int64 | Registration ≠ living city | application_train |
| `REG_CITY_NOT_WORK_CITY` | int64 | Registration ≠ work city | application_train |
| `LIVE_CITY_NOT_WORK_CITY` | int64 | Living ≠ work city | application_train |

### External Sources

| Column | Type | Meaning | Source |
|--------|------|---------|--------|
| `EXT_SOURCE_1` | float64 | External data source 1 score (high missingness) | application_train |
| `EXT_SOURCE_2` | float64 | External data source 2 score | application_train |
| `EXT_SOURCE_3` | float64 | External data source 3 score | application_train |

### Contact & Process Flags

| Column | Type | Meaning | Source |
|--------|------|---------|--------|
| `FLAG_MOBIL` | int64 | Mobile phone provided | application_train |
| `FLAG_EMP_PHONE` | int64 | Work phone provided | application_train |
| `FLAG_WORK_PHONE` | int64 | Work phone reachable | application_train |
| `FLAG_CONT_MOBILE` | int64 | Mobile reachable | application_train |
| `FLAG_PHONE` | int64 | Home phone provided | application_train |
| `FLAG_EMAIL` | int64 | Email provided | application_train |
| `WEEKDAY_APPR_PROCESS_START` | str | Day of week application started | application_train |
| `HOUR_APPR_PROCESS_START` | int64 | Hour of day application started | application_train |

### Housing Characteristics (AVG / MODE / MEDI variants)

Normalised building info about the applicant's residence. Each base feature
(`APARTMENTS`, `BASEMENTAREA`, `YEARS_BEGINEXPLUATATION`, `YEARS_BUILD`,
`COMMONAREA`, `ELEVATORS`, `ENTRANCES`, `FLOORSMAX`, `FLOORSMIN`, `LANDAREA`,
`LIVINGAPARTMENTS`, `LIVINGAREA`, `NONLIVINGAPARTMENTS`, `NONLIVINGAREA`) has
three variants: `_AVG`, `_MODE`, `_MEDI`. All are `float64`, high missingness
(~50–69%).

| Column | Type | Meaning | Source |
|--------|------|---------|--------|
| `FONDKAPREMONT_MODE` | str | Capital repair fund mode | application_train |
| `HOUSETYPE_MODE` | str | House type mode | application_train |
| `TOTALAREA_MODE` | float64 | Total area mode | application_train |
| `WALLSMATERIAL_MODE` | str | Walls material mode | application_train |
| `EMERGENCYSTATE_MODE` | str | Emergency state mode | application_train |

### Social Circle

| Column | Type | Meaning | Source |
|--------|------|---------|--------|
| `OBS_30_CNT_SOCIAL_CIRCLE` | float64 | Observable 30-day defaults in social circle | application_train |
| `DEF_30_CNT_SOCIAL_CIRCLE` | float64 | Defaulted 30-day in social circle | application_train |
| `OBS_60_CNT_SOCIAL_CIRCLE` | float64 | Observable 60-day defaults in social circle | application_train |
| `DEF_60_CNT_SOCIAL_CIRCLE` | float64 | Defaulted 60-day in social circle | application_train |

### Document Flags

| Column | Type | Meaning | Source |
|--------|------|---------|--------|
| `FLAG_DOCUMENT_2` … `FLAG_DOCUMENT_21` | int64 | Whether document N was provided (1/0) | application_train |

### Credit Bureau Enquiries

| Column | Type | Meaning | Source |
|--------|------|---------|--------|
| `AMT_REQ_CREDIT_BUREAU_HOUR` | float64 | Bureau enquiries in last hour | application_train |
| `AMT_REQ_CREDIT_BUREAU_DAY` | float64 | Bureau enquiries in last day | application_train |
| `AMT_REQ_CREDIT_BUREAU_WEEK` | float64 | Bureau enquiries in last week | application_train |
| `AMT_REQ_CREDIT_BUREAU_MON` | float64 | Bureau enquiries in last month | application_train |
| `AMT_REQ_CREDIT_BUREAU_QRT` | float64 | Bureau enquiries in last quarter | application_train |
| `AMT_REQ_CREDIT_BUREAU_YEAR` | float64 | Bureau enquiries in last year | application_train |

### Derived Columns (Phase 1)

| Column | Type | Meaning | Source |
|--------|------|---------|--------|
| `has_bureau` | bool | ≥ 1 record in `bureau.csv` | Derived (bureau) |
| `has_lender_history` | bool | ≥ 1 record in any of previous_application, installments_payments, POS_CASH_balance, credit_card_balance | Derived (lender tables) |
| `segment` | str | `full_file` / `thin_file` / `no_file` | Derived |

---

## `eval_attrs.parquet`

Evaluation-only attributes for fairness analysis. **Must not be used as model input.**

| Column | Type | Meaning | Source |
|--------|------|---------|--------|
| `SK_ID_CURR` | int64 | Applicant ID (FK → applicant_base) | applicant_base |
| `gender` | str | Gender: M, F, XNA | applicant_base.CODE_GENDER |
| `gender_xna` | bool | True if gender is XNA | Derived |
| `age_years` | float64 | Age in years (from DAYS_BIRTH) | Derived |
| `age_bin` | category | Age group: `<30`, `30-44`, `45-59`, `60+` | Derived |

---

## Split Files (`data/splits/`)

Each file contains the following columns:

| Column | Type | Meaning |
|--------|------|---------|
| `SK_ID_CURR` | int64 | Applicant ID |
| `TARGET` | int64 | Default flag |
| `segment` | str | Applicant segment |
