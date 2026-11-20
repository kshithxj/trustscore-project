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

### Bureau Features (Model A & B)

Aggregated per applicant from `bureau.csv`.

| Column | Type | Meaning | Formula / Source |
|--------|------|---------|------------------|
| `bureau_n_credits` | float64 | Total number of bureau records | `count()` of bureau rows |
| `bureau_n_active` | float64 | Number of active credits | `sum(CREDIT_ACTIVE == "Active")` |
| `bureau_days_since_first_credit` | float64 | Days since earliest credit opened | `-min(DAYS_CREDIT)` |
| `bureau_days_since_last_credit` | float64 | Days since most recent credit opened | `-max(DAYS_CREDIT)` |
| `bureau_total_overdue` | float64 | Total credit sum currently overdue | `sum(AMT_CREDIT_SUM_OVERDUE)` (NA -> 0) |
| `bureau_total_debt` | float64 | Total current debt across bureau credits | `sum(AMT_CREDIT_SUM_DEBT)` (NA -> 0) |

### Behavioral Lender-History Features (Model B)

Aggregated per applicant from lender history tables before joining. Unlinked applicants remain NaN.

#### From `previous_application.csv`

| Column | Type | Meaning | Formula / Source |
|--------|------|---------|------------------|
| `prev_n_apps` | float64 | Total past loan applications with Home Credit | `count()` of application rows |
| `prev_approved_rate` | float64 | Approval rate for previous applications | `mean(NAME_CONTRACT_STATUS == "Approved")` |
| `prev_refused_rate` | float64 | Refusal rate for previous applications | `mean(NAME_CONTRACT_STATUS == "Refused")` |
| `prev_mean_amt_credit` | float64 | Average credit amount of past applications | `mean(AMT_CREDIT)` |
| `prev_mean_amt_annuity` | float64 | Average annuity of past applications | `mean(AMT_ANNUITY)` |
| `prev_n_apps_last_365d` | float64 | Number of applications in the last 365 days | `sum(DAYS_DECISION >= -365)` |

#### From `installments_payments.csv`

| Column | Type | Meaning | Formula / Source |
|--------|------|---------|------------------|
| `inst_days_late_mean` | float64 | Mean payment delay in days (positive = late) | `mean(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT)` where entry not NaN |
| `inst_days_late_max` | float64 | Maximum payment delay in days | `max(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT)` |
| `inst_days_late_std` | float64 | Sample standard deviation of payment delays | `std(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT)` |
| `inst_frac_late` | float64 | Fraction of installments paid past due date | `mean(days_late > 0)` |
| `inst_frac_early` | float64 | Fraction of installments paid strictly ahead of due date | `mean(days_late < 0)` |
| `inst_payment_ratio_mean` | float64 | Mean payment-to-due ratio (per-row approximation) | `mean(AMT_PAYMENT / AMT_INSTALMENT)` for rows with `AMT_INSTALMENT > 0` |

#### From `POS_CASH_balance.csv`

| Column | Type | Meaning | Formula / Source |
|--------|------|---------|------------------|
| `pos_n_records` | float64 | Total monthly snapshot records for POS/cash loans | `count()` of monthly records |
| `pos_dpd_mean` | float64 | Average days past due across monthly snapshots | `mean(SK_DPD)` |
| `pos_dpd_max` | float64 | Maximum days past due observed across monthly snapshots | `max(SK_DPD)` |
| `pos_months_with_dpd` | float64 | Number of monthly snapshots with days past due > 0 | `sum(SK_DPD > 0)` |

#### From `credit_card_balance.csv`

| Column | Type | Meaning | Formula / Source |
|--------|------|---------|------------------|
| `cc_n_records` | float64 | Total monthly snapshot records for revolving cards | `count()` of monthly records |
| `cc_util_mean` | float64 | Average credit card credit utilization | `mean(AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL)` where limit > 0 |
| `cc_util_max` | float64 | Peak credit card credit utilization observed | `max(AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL)` where limit > 0 |
| `cc_months_with_dpd` | float64 | Number of monthly snapshots with card DPD > 0 | `sum(SK_DPD > 0)` |

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
