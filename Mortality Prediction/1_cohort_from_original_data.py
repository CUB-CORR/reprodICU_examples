# Cohort Extraction from Original SQL Databases

# Demographics:
#   Age (years), Gender (1 female / 0 male), ICU mortality,
#   admission weight (kg), admission height (cm), BMI

# Vital Signs:
#   Heart rate (min/mean/max), Systolic blood pressure (min/mean/max),
#   Diastolic blood pressure (min/mean/max), Temperature (min/mean/max)

# Laboratory Results:
#   Hemoglobin (min/max), Hematocrit (min/max), Platelets (min/max), WBC (min/max),
#   Albumin (min/max), Anion gap (min/max), Bicarbonate (min/max), BUN (min/max),
#   Calcium (min/max), Chloride (min/max), Creatinine (min/max), Glucose (min/max),
#   Sodium (min/max), Potassium (min/max), Bilirubin (min/max),
#   Lactate (min/max), pH (min/max), SO2 (min/max), PO2 (min/max),
#   PCO2 (min/max), Base excess (min/max)

# ------------------------------------------------------------------------------

# Source databases: AmsterdamUMCdb, eICU-CRD, MIMIC-III, MIMIC-IV
# Outcome: ICU mortality
# Inclusion criteria: age >= 18, ICU LOS >= 1 day
# Coverage: first 24 hours of ICU admission

################################################################################

from pathlib import Path

import pandas as pd
import polars as pl
from sqlalchemy import create_engine

# region config
# ==============================================================================

DB_CONFIG = {
    "MIMIC-III":       {"host": "localhost", "database": "mimic",   "user": "postgres", "password": "", "port": 5432},
    "MIMIC-IV":        {"host": "localhost", "database": "mimiciv", "user": "postgres", "password": "", "port": 5432},
    "eICU-CRD":        {"host": "localhost", "database": "eicu",    "user": "postgres", "password": "", "port": 5432},
    "AmsterdamUMCdb":  {"host": "localhost", "database": "umcdb",   "user": "postgres", "password": "", "port": 5432},
} # fmt: skip

OUTPUT_DIR = Path("data_source")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("\n=== COHORT EXTRACTION FROM ORIGINAL DATABASES ===")

# region MIMIC-III
# ==============================================================================
print("\n=== MIMIC-III ===")

query_m3 = """
WITH bg_agg AS (
    SELECT
        icustay_id,
        MIN(CASE WHEN lactate > 0 THEN lactate ELSE NULL END) AS lactate_min,
        MAX(CASE WHEN lactate > 0 THEN lactate ELSE NULL END) AS lactate_max,
        MIN(CASE WHEN ph > 0 THEN ph ELSE NULL END) AS ph_min,
        MAX(CASE WHEN ph > 0 THEN ph ELSE NULL END) AS ph_max,
        MIN(CASE WHEN so2 > 0 AND so2 <= 100 THEN so2 ELSE NULL END) AS so2_min,
        MAX(CASE WHEN so2 > 0 AND so2 <= 100 THEN so2 ELSE NULL END) AS so2_max,
        MIN(CASE WHEN po2 > 0 THEN po2 ELSE NULL END) AS po2_min,
        MAX(CASE WHEN po2 > 0 THEN po2 ELSE NULL END) AS po2_max,
        MIN(CASE WHEN pco2 > 0 THEN pco2 ELSE NULL END) AS pco2_min,
        MAX(CASE WHEN pco2 > 0 THEN pco2 ELSE NULL END) AS pco2_max,
        MIN(CASE WHEN baseexcess IS NOT NULL THEN baseexcess ELSE NULL END) AS baseexcess_min,
        MAX(CASE WHEN baseexcess IS NOT NULL THEN baseexcess ELSE NULL END) AS baseexcess_max,
        MIN(CASE WHEN bicarbonate > 0 THEN bicarbonate ELSE NULL END) AS bicarbonate_min,
        MAX(CASE WHEN bicarbonate > 0 THEN bicarbonate ELSE NULL END) AS bicarbonate_max,
        MIN(CASE WHEN hemoglobin > 0 THEN hemoglobin ELSE NULL END) AS hemoglobin_min,
        MAX(CASE WHEN hemoglobin > 0 THEN hemoglobin ELSE NULL END) AS hemoglobin_max,
        MIN(CASE WHEN hematocrit > 0 AND hematocrit <= 100 THEN hematocrit ELSE NULL END) AS hematocrit_min,
        MAX(CASE WHEN hematocrit > 0 AND hematocrit <= 100 THEN hematocrit ELSE NULL END) AS hematocrit_max,
        MIN(CASE WHEN chloride > 0 THEN chloride ELSE NULL END) AS chloride_min,
        MAX(CASE WHEN chloride > 0 THEN chloride ELSE NULL END) AS chloride_max,
        MIN(CASE WHEN calcium > 0 THEN calcium ELSE NULL END) AS calcium_min,
        MAX(CASE WHEN calcium > 0 THEN calcium ELSE NULL END) AS calcium_max,
        MIN(CASE WHEN glucose > 0 THEN glucose ELSE NULL END) AS glucose_min,
        MAX(CASE WHEN glucose > 0 THEN glucose ELSE NULL END) AS glucose_max,
        MIN(CASE WHEN potassium > 0 THEN potassium ELSE NULL END) AS potassium_min,
        MAX(CASE WHEN potassium > 0 THEN potassium ELSE NULL END) AS potassium_max,
        MIN(CASE WHEN sodium > 0 THEN sodium ELSE NULL END) AS sodium_min,
        MAX(CASE WHEN sodium > 0 THEN sodium ELSE NULL END) AS sodium_max
    FROM mimiciii.blood_gas_first_day
    GROUP BY icustay_id
)
SELECT
    d.icustay_id,
    d.admission_age,
    CASE WHEN d.gender = 'F' THEN 'Female' ELSE 'Male' END AS gender,
    CASE
      WHEN adm.deathtime BETWEEN ie.intime AND ie.outtime
      THEN 1
      WHEN adm.deathtime <= ie.intime
      THEN 1
      WHEN adm.dischtime <= ie.outtime AND adm.discharge_location = 'DEAD/EXPIRED'
      THEN 1
      ELSE 0
    END AS mortality_in_icu,
    d.hospital_expire_flag AS mortality_in_hosp,
    w.weight AS admission_weight_kg,
    h.height AS admission_height_cm,
    CASE
        WHEN w.weight > 0 AND h.height > 0
        THEN w.weight / ((h.height / 100.0) ^ 2)
        ELSE NULL
    END AS bmi,
    -- vitals
    v.heartrate_min AS hr_min, v.heartrate_max AS hr_max, v.heartrate_mean AS hr_mean,
    v.sysbp_min AS sbp_min, v.sysbp_max AS sbp_max, v.sysbp_mean AS sbp_mean,
    v.diasbp_min AS dbp_min, v.diasbp_max AS dbp_max, v.diasbp_mean AS dbp_mean,
    v.tempc_min AS temp_min, v.tempc_max AS temp_max, v.tempc_mean AS temp_mean,
    -- complete blood count
    LEAST(l.hemoglobin_min, bg.hemoglobin_min) AS hemoglobin_min,
    GREATEST(l.hemoglobin_max, bg.hemoglobin_max) AS hemoglobin_max,
    LEAST(l.hematocrit_min, bg.hematocrit_min) AS hematocrit_min,
    GREATEST(l.hematocrit_max, bg.hematocrit_max) AS hematocrit_max,
    l.platelet_min, l.platelet_max,
    l.wbc_min, l.wbc_max,
    -- chemistry
    l.albumin_min, l.albumin_max,
    l.aniongap_min, l.aniongap_max,
    LEAST(l.bicarbonate_min, bg.bicarbonate_min) AS bicarbonate_min,
    GREATEST(l.bicarbonate_max, bg.bicarbonate_max) AS bicarbonate_max,
    l.bun_min, l.bun_max,
    bg.calcium_min, bg.calcium_max,
    LEAST(l.chloride_min, bg.chloride_min) AS chloride_min,
    GREATEST(l.chloride_max, bg.chloride_max) AS chloride_max,
    LEAST(l.glucose_min, bg.glucose_min) AS glucose_min,
    GREATEST(l.glucose_max, bg.glucose_max) AS glucose_max,
    LEAST(l.potassium_min, bg.potassium_min) AS potassium_min,
    GREATEST(l.potassium_max, bg.potassium_max) AS potassium_max,
    LEAST(l.sodium_min, bg.sodium_min) AS sodium_min,
    GREATEST(l.sodium_max, bg.sodium_max) AS sodium_max,
    -- enzymes
    l.bilirubin_min, l.bilirubin_max,
    l.creatinine_min, l.creatinine_max,
    -- blood gases
    bg.lactate_min, bg.lactate_max,
    bg.ph_min, bg.ph_max,
    bg.so2_min, bg.so2_max,
    bg.po2_min, bg.po2_max,
    bg.pco2_min, bg.pco2_max,
    bg.baseexcess_min, bg.baseexcess_max
FROM mimiciii.icustay_detail d
LEFT JOIN mimiciii.icustays ie ON d.icustay_id = ie.icustay_id
LEFT JOIN mimiciii.admissions adm ON d.hadm_id = adm.hadm_id
LEFT JOIN mimiciii.vitals_first_day v ON d.icustay_id = v.icustay_id
LEFT JOIN mimiciii.height_first_day h ON d.icustay_id = h.icustay_id
LEFT JOIN mimiciii.weight_first_day w ON d.icustay_id = w.icustay_id
LEFT JOIN mimiciii.labs_first_day l ON d.icustay_id = l.icustay_id
LEFT JOIN bg_agg bg ON d.icustay_id = bg.icustay_id
WHERE d.admission_age >= 18 AND d.los_icu >= 1
"""

print("Connecting to MIMIC-III...")
cfg = DB_CONFIG["MIMIC-III"]
engine_m3 = create_engine(f"postgresql://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}") # fmt: skip
print("✓ Connected")

print("Extracting and processing cohort...")
cohort_m3 = pd.read_sql_query(query_m3, engine_m3)
engine_m3.dispose()

output_m3 = OUTPUT_DIR / "cohort_MIMIC_III.csv"
cohort_m3.to_csv(output_m3, index=False)
print(f"✓ Saved {output_m3} ({len(cohort_m3)} patients)")

# region MIMIC-IV
# ==============================================================================

print("\n=== MIMIC-IV ===")

query_m4 = """
SELECT
    d.stay_id,
    d.admission_age,
    CASE WHEN d.gender = 'F' THEN 'Female' ELSE 'Male' END AS gender,
    CASE -- taken from oasis.sql -> mortality flags
        WHEN adm.deathtime BETWEEN ie.intime AND ie.outtime
        THEN 1
        WHEN adm.deathtime <= ie.intime
        THEN 1
        WHEN adm.dischtime <= ie.outtime AND adm.discharge_location = 'DIED'
        THEN 1
        ELSE 0
    END AS mortality_in_icu,
    d.hospital_expire_flag AS mortality_in_hosp,
    w.weight AS admission_weight_kg,
    h.height AS admission_height_cm,
    CASE
        WHEN w.weight > 0 AND h.height > 0
        THEN w.weight / ((h.height / 100.0) ^ 2)
        ELSE NULL
    END AS bmi,
    -- vitals
    v.heart_rate_min AS hr_min, v.heart_rate_max AS hr_max, v.heart_rate_mean AS hr_mean,
    v.sbp_min, v.sbp_max, v.sbp_mean,
    v.dbp_min, v.dbp_max, v.dbp_mean,
    v.temperature_min AS temp_min, v.temperature_max AS temp_max, v.temperature_mean AS temp_mean,
    -- complete blood count
    LEAST(l.hemoglobin_min, bg.hemoglobin_min) AS hemoglobin_min,
    GREATEST(l.hemoglobin_max, bg.hemoglobin_max) AS hemoglobin_max,
    LEAST(l.hematocrit_min, bg.hematocrit_min) AS hematocrit_min,
    GREATEST(l.hematocrit_max, bg.hematocrit_max) AS hematocrit_max,
    l.platelets_min AS platelet_min, l.platelets_max AS platelet_max,
    l.wbc_min, l.wbc_max,
    -- chemistry
    l.albumin_min, l.albumin_max,
    l.aniongap_min, l.aniongap_max,
    LEAST(l.bicarbonate_min, bg.bicarbonate_min) AS bicarbonate_min,
    GREATEST(l.bicarbonate_max, bg.bicarbonate_max) AS bicarbonate_max,
    l.bun_min, l.bun_max,
    LEAST(l.calcium_min, bg.calcium_min) AS calcium_min,
    GREATEST(l.calcium_max, bg.calcium_max) AS calcium_max,
    LEAST(l.chloride_min, bg.chloride_min) AS chloride_min,
    GREATEST(l.chloride_max, bg.chloride_max) AS chloride_max,
    l.creatinine_min, l.creatinine_max,
    LEAST(l.glucose_min, bg.glucose_min) AS glucose_min,
    GREATEST(l.glucose_max, bg.glucose_max) AS glucose_max,
    LEAST(l.potassium_min, bg.potassium_min) AS potassium_min,
    GREATEST(l.potassium_max, bg.potassium_max) AS potassium_max,
    LEAST(l.sodium_min, bg.sodium_min) AS sodium_min,
    GREATEST(l.sodium_max, bg.sodium_max) AS sodium_max,
    -- enzymes
    l.bilirubin_total_min AS bilirubin_min, l.bilirubin_total_max AS bilirubin_max,
    -- blood gases
    bg.lactate_min, bg.lactate_max,
    bg.ph_min, bg.ph_max,
    bg.so2_min, bg.so2_max,
    bg.po2_min, bg.po2_max,
    bg.pco2_min, bg.pco2_max,
    bg.baseexcess_min, bg.baseexcess_max
FROM mimiciv_derived.icustay_detail d
LEFT JOIN mimiciv_icu.icustays ie ON d.stay_id = ie.stay_id
LEFT JOIN mimiciv_hosp.admissions adm ON d.hadm_id = adm.hadm_id
LEFT JOIN mimiciv_derived.first_day_vitalsign v ON d.stay_id = v.stay_id
LEFT JOIN mimiciv_derived.first_day_height h ON d.stay_id = h.stay_id
LEFT JOIN mimiciv_derived.first_day_weight w ON d.stay_id = w.stay_id
LEFT JOIN mimiciv_derived.first_day_lab l ON d.stay_id = l.stay_id
LEFT JOIN mimiciv_derived.first_day_bg bg ON d.stay_id = bg.stay_id
WHERE d.admission_age >= 18 AND d.los_icu >= 1
"""

print("Connecting to MIMIC-IV...")
cfg = DB_CONFIG["MIMIC-IV"]
engine_m4 = create_engine(f"postgresql://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}") # fmt: skip
print("✓ Connected")

print("Extracting and processing cohort...")
cohort_m4 = pd.read_sql_query(query_m4, engine_m4)
engine_m4.dispose()

output_m4 = OUTPUT_DIR / "cohort_MIMIC_IV.csv"
cohort_m4.to_csv(output_m4, index=False)
print(f"✓ Saved {output_m4} ({len(cohort_m4)} patients)")

# region eICU-CRD
# ==============================================================================

print("\n=== eICU-CRD ===")

query_eicu = """
-- Aggregate vitals using UNION + GROUP BY (more efficient than FULL OUTER JOINs)
WITH vitals_union AS (
    SELECT patientunitstayid, heartrate, sbp, dbp, temperature
    FROM (
        SELECT patientunitstayid, heartrate,
            NULLIF(CASE WHEN ibp_systolic > 0 AND ibp_systolic < 400 THEN ibp_systolic
                        WHEN nibp_systolic > 0 AND nibp_systolic < 400 THEN nibp_systolic END, NULL) AS sbp,
            NULLIF(CASE WHEN ibp_diastolic > 0 AND ibp_diastolic < 300 THEN ibp_diastolic
                        WHEN nibp_diastolic > 0 AND nibp_diastolic < 300 THEN nibp_diastolic END, NULL) AS dbp,
            temperature
        FROM eicu_crd.pivoted_vital
        WHERE chartoffset >= -360 AND chartoffset <= 1440
        UNION ALL
        SELECT patientunitstayid, heartrate,
            CASE WHEN systemicsystolic > 0 AND systemicsystolic < 400 THEN systemicsystolic ELSE NULL END,
            CASE WHEN systemicdiastolic > 0 AND systemicdiastolic < 300 THEN systemicdiastolic ELSE NULL END,
            temperature
        FROM eicu_crd.vitalPeriodic
        WHERE observationoffset >= -360 AND observationoffset <= 1440
        UNION ALL
        SELECT patientunitstayid, NULL,
            CASE WHEN noninvasivesystolic > 0 AND noninvasivesystolic < 400 THEN noninvasivesystolic ELSE NULL END,
            CASE WHEN noninvasivediastolic > 0 AND noninvasivediastolic < 300 THEN noninvasivediastolic ELSE NULL END,
            NULL
        FROM eicu_crd.vitalAperiodic
        WHERE observationoffset >= -360 AND observationoffset <= 1440
    ) v
),
vitals_agg AS (
    SELECT patientunitstayid,
        MIN(CASE WHEN heartrate > 0 AND heartrate < 300 THEN heartrate ELSE NULL END) AS hr_min,
        MAX(CASE WHEN heartrate > 0 AND heartrate < 300 THEN heartrate ELSE NULL END) AS hr_max,
        ROUND(AVG(CASE WHEN heartrate > 0 AND heartrate < 300 THEN heartrate ELSE NULL END)::NUMERIC, 2) AS hr_mean,
        MIN(sbp) AS sbp_min, MAX(sbp) AS sbp_max, ROUND(AVG(sbp)::NUMERIC, 2) AS sbp_mean,
        MIN(dbp) AS dbp_min, MAX(dbp) AS dbp_max, ROUND(AVG(dbp)::NUMERIC, 2) AS dbp_mean,
        MIN(CASE WHEN temperature > 10 AND temperature < 50 THEN temperature ELSE NULL END) AS temp_min,
        MAX(CASE WHEN temperature > 10 AND temperature < 50 THEN temperature ELSE NULL END) AS temp_max,
        ROUND(AVG(CASE WHEN temperature > 10 AND temperature < 50 THEN temperature ELSE NULL END)::NUMERIC, 2) AS temp_mean
    FROM vitals_union
    GROUP BY patientunitstayid
),
-- labs / bg aggregated separately due to different table structures, then joined in final SELECT
labs_agg AS (
    SELECT patientunitstayid,
        MIN(CASE WHEN labname ILIKE 'o2 sat%%' AND labresult > 0 AND labresult <= 100 THEN labresult ELSE NULL END) AS so2_min,
        MAX(CASE WHEN labname ILIKE 'o2 sat%%' AND labresult > 0 AND labresult <= 100 THEN labresult ELSE NULL END) AS so2_max
    FROM eicu_crd.lab
    WHERE labresultoffset >= -360 AND labresultoffset <= 1440 AND labresult IS NOT NULL AND labname ILIKE 'o2 sat%%'
    GROUP BY patientunitstayid
),
pivoted_lab_agg AS (
    SELECT patientunitstayid,
        MIN(CASE WHEN calcium > 0 THEN calcium ELSE NULL END) AS calcium_min,
        MAX(CASE WHEN calcium > 0 THEN calcium ELSE NULL END) AS calcium_max,
        MIN(CASE WHEN glucose > 0 THEN glucose ELSE NULL END) AS glucose_min,
        MAX(CASE WHEN glucose > 0 THEN glucose ELSE NULL END) AS glucose_max
    FROM eicu_crd.pivoted_lab
    WHERE chartoffset >= -360 AND chartoffset <= 1440
    GROUP BY patientunitstayid
),
pivoted_bg_agg AS (
    SELECT patientunitstayid,
        MIN(CASE WHEN pao2 > 0 THEN pao2 ELSE NULL END) AS po2_min,
        MAX(CASE WHEN pao2 > 0 THEN pao2 ELSE NULL END) AS po2_max,
        MIN(CASE WHEN paco2 > 0 THEN paco2 ELSE NULL END) AS pco2_min,
        MAX(CASE WHEN paco2 > 0 THEN paco2 ELSE NULL END) AS pco2_max,
        MIN(CASE WHEN pH > 0 THEN pH ELSE NULL END) AS ph_min,
        MAX(CASE WHEN pH > 0 THEN pH ELSE NULL END) AS ph_max,
        MIN(CASE WHEN aniongap > 0 THEN aniongap ELSE NULL END) AS aniongap_min,
        MAX(CASE WHEN aniongap > 0 THEN aniongap ELSE NULL END) AS aniongap_max,
        MIN(CASE WHEN baseexcess IS NOT NULL THEN baseexcess ELSE NULL END) AS baseexcess_min,
        MAX(CASE WHEN baseexcess IS NOT NULL THEN baseexcess ELSE NULL END) AS baseexcess_max
    FROM eicu_crd.pivoted_bg
    WHERE chartoffset >= -360 AND chartoffset <= 1440
    GROUP BY patientunitstayid
)
SELECT
    p.patientunitstayid,
    CASE WHEN p.age = '> 89' THEN 90 WHEN p.age ~ '^[0-9]+$' THEN p.age::int ELSE NULL END AS admission_age_years,
    CASE WHEN p.gender = 'Female' THEN 'Female' ELSE 'Male' END AS gender,
    CASE WHEN lower(p.unitdischargestatus) like '%%alive%%' THEN 0
         WHEN lower(p.unitdischargestatus) like '%%expired%%' THEN 1
         ELSE NULL END AS mortality_in_icu,
    CASE WHEN lower(p.hospitaldischargestatus) like '%%alive%%' THEN 0
         WHEN lower(p.hospitaldischargestatus) like '%%expired%%' THEN 1
         ELSE NULL END AS mortality_in_hosp,
    p.admissionweight AS admission_weight_kg,
    p.admissionheight AS admission_height_cm,
    CASE
        WHEN p.admissionweight > 0 AND p.admissionheight > 0
        THEN ROUND((p.admissionweight / ((p.admissionheight / 100.0) ^ 2))::NUMERIC, 2)
        ELSE NULL
    END AS bmi,
    -- vitals (combined from joined vital sources, preferring IBP)
    va.hr_min, va.hr_max, va.hr_mean,
    va.sbp_min, va.sbp_max, va.sbp_mean,
    va.dbp_min, va.dbp_max, va.dbp_mean,
    va.temp_min, va.temp_max, va.temp_mean,
    -- complete blood count
    l.HEMOGLOBIN_min, l.HEMOGLOBIN_max,
    l.HEMATOCRIT_min, l.HEMATOCRIT_max,
    l.PLATELET_min, l.PLATELET_max,
    l.WBC_min, l.WBC_max,
    -- chemistry
    l.ALBUMIN_min, l.ALBUMIN_max,
    pbg.aniongap_min, pbg.aniongap_max,
    l.BICARBONATE_min, l.BICARBONATE_max,
    l.BUN_min, l.BUN_max,
    pla.calcium_min, pla.calcium_max,
    l.CHLORIDE_min, l.CHLORIDE_max,
    l.CREATININE_min, l.CREATININE_max,
    pla.glucose_min, pla.glucose_max,
    l.POTASSIUM_min, l.POTASSIUM_max,
    l.SODIUM_min, l.SODIUM_max,
    -- enzymes
    l.BILIRUBIN_min, l.BILIRUBIN_max,
    -- blood gases
    l.LACTATE_min, l.LACTATE_max,
    pbg.ph_min, pbg.ph_max,
    la.so2_min, la.so2_max,
    pbg.po2_min, pbg.po2_max,
    pbg.pco2_min, pbg.pco2_max,
    pbg.baseexcess_min, pbg.baseexcess_max
FROM eicu_crd.patient p
LEFT JOIN vitals_agg va ON p.patientunitstayid = va.patientunitstayid
LEFT JOIN labs_agg la ON p.patientunitstayid = la.patientunitstayid
LEFT JOIN eicu_crd.labsfirstday l ON p.patientunitstayid = l.patientunitstayid
LEFT JOIN pivoted_lab_agg pla ON p.patientunitstayid = pla.patientunitstayid
LEFT JOIN pivoted_bg_agg pbg ON p.patientunitstayid = pbg.patientunitstayid
WHERE CASE WHEN p.age = '> 89' THEN 90 WHEN p.age ~ '^[0-9]+$' THEN p.age::int ELSE NULL END >= 18
  AND ROUND(p.unitdischargeoffset::NUMERIC / 60.0 / 24.0, 2) >= 1
"""

print("Connecting to eICU-CRD...")
cfg = DB_CONFIG["eICU-CRD"]
engine_eicu = create_engine(f"postgresql://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}") # fmt: skip
print("✓ Connected")

print("Extracting and processing cohort...")
cohort_eicu = pd.read_sql_query(query_eicu, engine_eicu)
engine_eicu.dispose()

output_eicu = OUTPUT_DIR / "cohort_eICU.csv"
cohort_eicu.to_csv(output_eicu, index=False)
print(f"✓ Saved {output_eicu} ({len(cohort_eicu)} patients)")

# region AmsterdamUMCdb
# ==============================================================================

print("\n=== AmsterdamUMCdb ===")

query_umcdb = """
SET enable_seqscan = off;
WITH filtered_admissions AS (
    SELECT
        admissionid,
        admittedat,
        gender,
        destination,
        -- Derive approximate numeric values from UMCdb categorical group columns
        -- (exact age/weight/height not available in published dataset)
        -- Convention follows vasopressors_inotropes.sql
        CASE
            WHEN agegroup LIKE '18%%' THEN 30  -- 18-39
            WHEN agegroup LIKE '40%%' THEN 45  -- 40-49
            WHEN agegroup LIKE '50%%' THEN 55  -- 50-59
            WHEN agegroup LIKE '60%%' THEN 65  -- 60-69
            WHEN agegroup LIKE '70%%' THEN 75  -- 70-79
            WHEN agegroup LIKE '80%%' THEN 85  -- 80+
            ELSE 60 -- approximate mean age for entire cohort
        END AS admission_age,
        CASE
            WHEN weightgroup LIKE '59%%'  THEN  55 --  59-
            WHEN weightgroup LIKE '60%%'  THEN  65 --  60- 69
            WHEN weightgroup LIKE '70%%'  THEN  75 --  70- 79
            WHEN weightgroup LIKE '80%%'  THEN  85 --  80- 89
            WHEN weightgroup LIKE '90%%'  THEN  95 --  90- 99
            WHEN weightgroup LIKE '100%%' THEN 105 -- 100-109
            WHEN weightgroup LIKE '110%%' THEN 115 -- 110+
            ELSE 80 -- approximate mean weight for entire cohort
        END AS admission_weight_kg,
        CASE
            WHEN heightgroup LIKE '159%%' THEN 155 -- 159-
            WHEN heightgroup LIKE '160%%' THEN 165 -- 160-169
            WHEN heightgroup LIKE '170%%' THEN 175 -- 170-179
            WHEN heightgroup LIKE '180%%' THEN 185 -- 180-189
            WHEN heightgroup LIKE '190%%' THEN 195 -- 190+
            ELSE 175 -- approximate mean height for entire cohort
        END AS admission_height_cm
    FROM admissions
    WHERE (dischargedat - admittedat) >= 1000*60*60*24 -- ICU LOS >= 1 day
),
vitals_data AS (
    SELECT
        n.admissionid,
        n.itemid,
        n.value
    FROM numericitems n
    INNER JOIN filtered_admissions a
        ON n.admissionid = a.admissionid
          AND n.itemid IN (
            6640,                        -- hr
            6641, 6678, 8842,            -- sbp  -- modified `(?:ABP|bloeddruk).*gemiddeld` regex from apache_ii.ipynb
            6643, 6680, 8844,            -- dbp  -- modified `(?:ABP|bloeddruk).*gemiddeld` regex from apache_ii.ipynb
            8658, 8659, 8662, 13058,
            13059, 13060, 13061, 13062,
            13063, 13952, 16110          -- temp (from temperature.sql)
          )
          AND (n.measuredat - a.admittedat) BETWEEN -1000*60*60*6 AND 1000*60*60*24
),
labs_data AS (
    SELECT
        n.admissionid,
        n.itemid,
        n.value
    FROM numericitems n
    INNER JOIN filtered_admissions a
        ON n.admissionid = a.admissionid
          AND n.itemid IN (
            9947, 6833, 9557,                -- glucose
            9941, 6836, 14216,               -- creatinine
            9943, 6850,                      -- bun
            9960, 6778, 10286, 19703, 9553,  -- hemoglobin
            11423, 11545, 6777,              -- hematocrit
            9964, 6797, 10409, 14252,        -- platelets
            9965, 6779,                      -- wbc
            9924, 6840, 9555, 10284,         -- sodium
            9927, 9556, 6835, 10285,         -- potassium
            9937, 6801,                      -- albumin
            9945, 6813,                      -- bilirubin
            9559, 8492,                      -- anion gap
            10053, 6837, 9580,               -- lactate
            12310, 6848,                     -- pH
            6846, 9990, 21213,               -- pCO2
            7433, 9996, 21214,               -- pO2
            9994, 6807,                      -- base excess
            9992, 6810,                      -- bicarbonate
            12311, 8903,                     -- sO2
            -- manually selected labs based on itemid patterns from apache_ii.ipynb
            9933, 9560, 6817,                -- calcium
            14413, 9930, 9558, 6819          -- chloride
            )
          AND n.islabresult = B'1'
          AND (n.measuredat - a.admittedat) BETWEEN -1000*60*60*6 AND 1000*60*60*24
),
vitals_agg AS (
    -- Vitals aggregation (heart rate, systolic/diastolic BP, temperature)
    SELECT
        vd.admissionid,
        -- Heart rate
        MIN(CASE WHEN vd.itemid = 6640 THEN vd.value ELSE NULL END) AS hr_min,
        MAX(CASE WHEN vd.itemid = 6640 THEN vd.value ELSE NULL END) AS hr_max,
        ROUND(AVG(CASE WHEN vd.itemid = 6640 THEN vd.value ELSE NULL END)::NUMERIC, 2) AS hr_mean,
        -- Systolic BP
        MIN(CASE WHEN vd.itemid IN (6641, 6678, 8842) THEN vd.value ELSE NULL END) AS sbp_min,
        MAX(CASE WHEN vd.itemid IN (6641, 6678, 8842) THEN vd.value ELSE NULL END) AS sbp_max,
        ROUND(AVG(CASE WHEN vd.itemid IN (6641, 6678, 8842) THEN vd.value ELSE NULL END)::NUMERIC, 2) AS sbp_mean,
        -- Diastolic BP
        MIN(CASE WHEN vd.itemid IN (6643, 6680, 8844) THEN vd.value ELSE NULL END) AS dbp_min,
        MAX(CASE WHEN vd.itemid IN (6643, 6680, 8844) THEN vd.value ELSE NULL END) AS dbp_max,
        ROUND(AVG(CASE WHEN vd.itemid IN (6643, 6680, 8844) THEN vd.value ELSE NULL END)::NUMERIC, 2) AS dbp_mean,
        -- Temperature
        MIN(CASE WHEN vd.itemid IN (8658, 8659, 8662, 13058, 13059, 13060, 13061, 13062, 13063, 13952, 16110) THEN vd.value ELSE NULL END) AS temp_min,
        MAX(CASE WHEN vd.itemid IN (8658, 8659, 8662, 13058, 13059, 13060, 13061, 13062, 13063, 13952, 16110) THEN vd.value ELSE NULL END) AS temp_max,
        ROUND(AVG(CASE WHEN vd.itemid IN (8658, 8659, 8662, 13058, 13059, 13060, 13061, 13062, 13063, 13952, 16110) THEN vd.value ELSE NULL END)::NUMERIC, 2) AS temp_mean
    FROM vitals_data vd
    GROUP BY vd.admissionid
),
labs_agg AS (
    -- Labs aggregation
    -- Reference: https://github.com/AmsterdamUMC/AmsterdamUMCdb/issues/5
    SELECT
        ld.admissionid,
        -- Glucose (serum/whole-blood + Glucose Astrup 9557)
        MIN(CASE WHEN ld.itemid IN (9947, 6833, 9557) THEN ld.value ELSE NULL END) AS glucose_min,
        MAX(CASE WHEN ld.itemid IN (9947, 6833, 9557) THEN ld.value ELSE NULL END) AS glucose_max,
        -- Creatinine
        MIN(CASE WHEN ld.itemid IN (9941, 6836, 14216) THEN ld.value ELSE NULL END) AS creatinine_min,
        MAX(CASE WHEN ld.itemid IN (9941, 6836, 14216) THEN ld.value ELSE NULL END) AS creatinine_max,
        -- Blood Urea Nitrogen
        MIN(CASE WHEN ld.itemid IN (9943, 6850) THEN ld.value ELSE NULL END) AS bun_min,
        MAX(CASE WHEN ld.itemid IN (9943, 6850) THEN ld.value ELSE NULL END) AS bun_max,
        -- Hemoglobin
        MIN(CASE WHEN ld.itemid IN (9960, 6778, 10286, 19703, 9553) THEN ld.value ELSE NULL END) AS hemoglobin_min,
        MAX(CASE WHEN ld.itemid IN (9960, 6778, 10286, 19703, 9553) THEN ld.value ELSE NULL END) AS hemoglobin_max,
        -- Hematocrit
        MIN(CASE WHEN ld.itemid IN (11423, 11545, 6777) THEN ld.value ELSE NULL END) AS hematocrit_min,
        MAX(CASE WHEN ld.itemid IN (11423, 11545, 6777) THEN ld.value ELSE NULL END) AS hematocrit_max,
        -- Platelets
        MIN(CASE WHEN ld.itemid IN (9964, 6797, 10409, 14252) THEN ld.value ELSE NULL END) AS platelet_min,
        MAX(CASE WHEN ld.itemid IN (9964, 6797, 10409, 14252) THEN ld.value ELSE NULL END) AS platelet_max,
        -- White blood cell count
        MIN(CASE WHEN ld.itemid IN (9965, 6779) THEN ld.value ELSE NULL END) AS wbc_min,
        MAX(CASE WHEN ld.itemid IN (9965, 6779) THEN ld.value ELSE NULL END) AS wbc_max,
        -- Sodium
        MIN(CASE WHEN ld.itemid IN (9924, 6840, 9555, 10284) THEN ld.value ELSE NULL END) AS sodium_min,
        MAX(CASE WHEN ld.itemid IN (9924, 6840, 9555, 10284) THEN ld.value ELSE NULL END) AS sodium_max,
        -- Potassium
        MIN(CASE WHEN ld.itemid IN (9927, 9556, 6835, 10285) THEN ld.value ELSE NULL END) AS potassium_min,
        MAX(CASE WHEN ld.itemid IN (9927, 9556, 6835, 10285) THEN ld.value ELSE NULL END) AS potassium_max,
        -- Albumin
        MIN(CASE WHEN ld.itemid IN (9937, 6801) THEN ld.value ELSE NULL END) AS albumin_min,
        MAX(CASE WHEN ld.itemid IN (9937, 6801) THEN ld.value ELSE NULL END) AS albumin_max,
        -- Bilirubin
        MIN(CASE WHEN ld.itemid IN (9945, 6813) THEN ld.value ELSE NULL END) AS bilirubin_min,
        MAX(CASE WHEN ld.itemid IN (9945, 6813) THEN ld.value ELSE NULL END) AS bilirubin_max,
        -- Anion gap
        MIN(CASE WHEN ld.itemid IN (9559, 8492) THEN ld.value ELSE NULL END) AS aniongap_min,
        MAX(CASE WHEN ld.itemid IN (9559, 8492) THEN ld.value ELSE NULL END) AS aniongap_max,
        -- Lactate
        MIN(CASE WHEN ld.itemid IN (10053, 6837, 9580) THEN ld.value ELSE NULL END) AS lactate_min,
        MAX(CASE WHEN ld.itemid IN (10053, 6837, 9580) THEN ld.value ELSE NULL END) AS lactate_max,
        -- pH
        MIN(CASE WHEN ld.itemid IN (12310, 6848) THEN ld.value ELSE NULL END) AS ph_min,
        MAX(CASE WHEN ld.itemid IN (12310, 6848) THEN ld.value ELSE NULL END) AS ph_max,
        -- PCO2
        MIN(CASE WHEN ld.itemid IN (6846, 9990, 21213) THEN ld.value ELSE NULL END) AS pco2_min,
        MAX(CASE WHEN ld.itemid IN (6846, 9990, 21213) THEN ld.value ELSE NULL END) AS pco2_max,
        -- PO2
        MIN(CASE WHEN ld.itemid IN (7433, 9996, 21214) THEN ld.value ELSE NULL END) AS po2_min,
        MAX(CASE WHEN ld.itemid IN (7433, 9996, 21214) THEN ld.value ELSE NULL END) AS po2_max,
        -- Base Excess
        MIN(CASE WHEN ld.itemid IN (9994, 6807) THEN ld.value ELSE NULL END) AS baseexcess_min,
        MAX(CASE WHEN ld.itemid IN (9994, 6807) THEN ld.value ELSE NULL END) AS baseexcess_max,
        -- Bicarbonate
        MIN(CASE WHEN ld.itemid IN (9992, 6810) THEN ld.value ELSE NULL END) AS bicarbonate_min,
        MAX(CASE WHEN ld.itemid IN (9992, 6810) THEN ld.value ELSE NULL END) AS bicarbonate_max,
        -- O2 saturation
        MIN(CASE WHEN ld.itemid IN (12311, 8903) THEN ld.value ELSE NULL END) AS so2_min,
        MAX(CASE WHEN ld.itemid IN (12311, 8903) THEN ld.value ELSE NULL END) AS so2_max,
        -- ---------------------------------------------------------------------
        -- -- manually selected labs based on itemid patterns from apache_ii.ipynb
        -- Calcium
        MIN(CASE WHEN ld.itemid IN (9933, 9560, 6817) THEN ld.value ELSE NULL END) AS calcium_min,
        MAX(CASE WHEN ld.itemid IN (9933, 9560, 6817) THEN ld.value ELSE NULL END) AS calcium_max,
        -- Chloride
        MIN(CASE WHEN ld.itemid IN (14413, 9930, 9558, 6819) THEN ld.value ELSE NULL END) AS chloride_min,
        MAX(CASE WHEN ld.itemid IN (14413, 9930, 9558, 6819) THEN ld.value ELSE NULL END) AS chloride_max
    FROM labs_data ld
    GROUP BY ld.admissionid
)
SELECT
    a.admissionid,
    a.admission_age,
    CASE WHEN a.gender = 'Vrouw' THEN 'Female' ELSE 'Male' END AS gender,
    CASE WHEN a.destination = 'Overleden' THEN 1 ELSE 0 END AS mortality_in_icu,
    NULL::NUMERIC AS mortality_in_hosp,
    a.admission_weight_kg,
    a.admission_height_cm,
    ROUND((a.admission_weight_kg / ((a.admission_height_cm / 100.0) ^ 2))::NUMERIC, 2) AS bmi,
    -- vitals
    v.hr_min, v.hr_max, v.hr_mean,
    v.sbp_min, v.sbp_max, v.sbp_mean,
    v.dbp_min, v.dbp_max, v.dbp_mean,
    v.temp_min, v.temp_max, v.temp_mean,
    -- complete blood count
    l.hemoglobin_min, l.hemoglobin_max,
    l.hematocrit_min, l.hematocrit_max,
    l.platelet_min, l.platelet_max,
    l.wbc_min, l.wbc_max,
    -- chemistry
    l.albumin_min, l.albumin_max,
    l.aniongap_min, l.aniongap_max,
    l.bicarbonate_min, l.bicarbonate_max,
    l.bun_min, l.bun_max,
    l.calcium_min, l.calcium_max,
    l.chloride_min, l.chloride_max,
    l.creatinine_min, l.creatinine_max,
    l.glucose_min, l.glucose_max,
    l.potassium_min, l.potassium_max,
    l.sodium_min, l.sodium_max,
    -- enzymes
    l.bilirubin_min, l.bilirubin_max,
    -- blood gases
    l.lactate_min, l.lactate_max,
    l.ph_min, l.ph_max,
    l.so2_min, l.so2_max,
    l.po2_min, l.po2_max,
    l.pco2_min, l.pco2_max,
    l.baseexcess_min, l.baseexcess_max
FROM filtered_admissions a
LEFT JOIN vitals_agg v ON a.admissionid = v.admissionid
LEFT JOIN labs_agg l ON a.admissionid = l.admissionid;
"""

print("Connecting to AmsterdamUMCdb...")
cfg = DB_CONFIG["AmsterdamUMCdb"]
engine_umcdb = create_engine(f"postgresql://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}") # fmt: skip
print("✓ Connected")

print("Extracting and processing cohort...")
cohort_umcdb = pd.read_sql_query(query_umcdb, engine_umcdb)
engine_umcdb.dispose()

output_umcdb = OUTPUT_DIR / "cohort_UMCdb.csv"
cohort_umcdb.to_csv(output_umcdb, index=False)
print(f"✓ Saved {output_umcdb} ({len(cohort_umcdb)} patients)")

################################################################################
################################################################################

# region SUMMARY
# ==============================================================================

print("\n=== SUMMARY ===")

def _load_and_normalize(path, rename_map, cast_map, source_label):
    df = (
        pl.read_csv(path)
        .drop_nulls("mortality_in_icu") # drop patients with null ICU mortality
        .rename(rename_map)
        .cast(cast_map)
        .with_columns(pl.lit(source_label).alias("source"))
    )
    print(f"{source_label}: {len(df):6.0f}")
    return df

print("\ndone")

cohort_m3 = _load_and_normalize(
    OUTPUT_DIR / "cohort_MIMIC_III.csv",
    rename_map={
        "icustay_id": "stay_id",
    },
    cast_map={
        "stay_id": str,
        "admission_age": float,
    },
    source_label="MIMIC-III",
)

cohort_m4 = _load_and_normalize(
    OUTPUT_DIR / "cohort_MIMIC_IV.csv",
    rename_map={},
    cast_map={
        "stay_id": str,
        "admission_age": float,
    },
    source_label="MIMIC-IV",
)

cohort_eicu = _load_and_normalize(
    OUTPUT_DIR / "cohort_eICU.csv",
    rename_map={
        "patientunitstayid": "stay_id",
        "admission_age_years": "admission_age",
    },
    cast_map={
        "stay_id": str,
        "admission_age": float,
        "mortality_in_icu": int,
        "mortality_in_hosp": int,
    },
    source_label="eICU-CRD",
)

cohort_umcdb = _load_and_normalize(
    OUTPUT_DIR / "cohort_UMCdb.csv",
    rename_map={
        "admissionid": "stay_id",
    },
    cast_map={
        "stay_id": str,
        "admission_age": float,
        "mortality_in_hosp": int,
        "admission_weight_kg": float,
        "admission_height_cm": float,
    },
    source_label="AmsterdamUMCdb",
)

RENAME_MAP = {
    "stay_id": "Global ICU Stay ID",
    "source": "Source Dataset",
    "admission_age": "Admission Age (years)",
    # "gender": "",
    "isfemale": "isfemale",
    "mortality_in_icu": "Mortality in ICU",
    "mortality_in_hosp": "Mortality in Hospital",
    "admission_weight_kg": "Admission Weight (kg)",
    # "admission_height_cm": "Admission Height (cm)",
    "bmi": "BMI",
    "hr_min": "Heart rate (min)",
    "hr_max": "Heart rate (max)",
    "hr_mean": "Heart rate (mean)",
    "sbp_min": "Systolic blood pressure (min)",
    "sbp_max": "Systolic blood pressure (max)",
    "sbp_mean": "Systolic blood pressure (mean)",
    "dbp_min": "Diastolic blood pressure (min)",
    "dbp_max": "Diastolic blood pressure (max)",
    "dbp_mean": "Diastolic blood pressure (mean)",
    "temp_min": "Temperature (min)",
    "temp_max": "Temperature (max)",
    "temp_mean": "Temperature (mean)",
    "albumin_max": "Albumin (max)",
    "albumin_min": "Albumin (min)",
    "aniongap_max": "Anion gap (max)",
    "aniongap_min": "Anion gap (min)",
    "baseexcess_max": "Base excess (max)",
    "baseexcess_min": "Base excess (min)",
    "bicarbonate_max": "Bicarbonate (max)",
    "bicarbonate_min": "Bicarbonate (min)",
    "bilirubin_max": "Bilirubin (max)",
    "bilirubin_min": "Bilirubin (min)",
    "bun_max": "Urea nitrogen (max)",
    "bun_min": "Urea nitrogen (min)",
    "calcium_max": "Calcium (max)",
    "calcium_min": "Calcium (min)",
    "chloride_max": "Chloride (max)",
    "chloride_min": "Chloride (min)",
    "creatinine_max": "Creatinine (max)",
    "creatinine_min": "Creatinine (min)",
    "glucose_max": "Glucose (max)",
    "glucose_min": "Glucose (min)",
    "hematocrit_max": "Hematocrit (max)",
    "hematocrit_min": "Hematocrit (min)",
    "hemoglobin_max": "Hemoglobin (max)",
    "hemoglobin_min": "Hemoglobin (min)",
    "lactate_max": "Lactate (max)",
    "lactate_min": "Lactate (min)",
    "pco2_max": "Carbon dioxide (max)",
    "pco2_min": "Carbon dioxide (min)",
    "ph_max": "pH (max)",
    "ph_min": "pH (min)",
    "platelet_max": "Platelets (max)",
    "platelet_min": "Platelets (min)",
    "po2_max": "Oxygen (max)",
    "po2_min": "Oxygen (min)",
    "potassium_max": "Potassium (max)",
    "potassium_min": "Potassium (min)",
    "so2_max": "Oxygen saturation (max)",
    "so2_min": "Oxygen saturation (min)",
    "sodium_max": "Sodium (max)",
    "sodium_min": "Sodium (min)",
    "wbc_max": "Leukocytes (max)",
    "wbc_min": "Leukocytes (min)",
}

(
    pl.concat([cohort_m3, cohort_m4, cohort_eicu, cohort_umcdb], how="diagonal")
    .with_columns(pl.col("gender").eq("Female").alias("isfemale"))
    .rename(mapping=RENAME_MAP)
    .with_columns(
        pl.when(pl.col("Source Dataset") == "MIMIC-IV")
        .then(pl.concat_str(pl.lit("mimic4-"), pl.col("Global ICU Stay ID")))
        .when(pl.col("Source Dataset") == "MIMIC-III")
        .then(pl.concat_str(pl.lit("mimic3-"), pl.col("Global ICU Stay ID")))
        .when(pl.col("Source Dataset") == "eICU-CRD")
        .then(pl.concat_str(pl.lit("eicu-"), pl.col("Global ICU Stay ID")))
        .when(pl.col("Source Dataset") == "AmsterdamUMCdb")
        .then(pl.concat_str(pl.lit("umcdb-"), pl.col("Global ICU Stay ID")))
        .otherwise(pl.col("Global ICU Stay ID"))
        .alias("Global ICU Stay ID")
    )
    .select(RENAME_MAP.values())
    .sort("Global ICU Stay ID")
    .write_parquet(OUTPUT_DIR / "cohort.parquet")
)
