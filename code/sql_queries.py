"""
sql_queries.py -- harmonised BigQuery extraction for MIMIC-IV v3.1 and eICU-CRD v2.0.

Design principle: BOTH databases are first reshaped into the SAME long format
    (stay_id, off_min, var, v)
where off_min = minutes since ICU admission, var = harmonised variable name,
v = numeric value in harmonised units. The SAME aggregation SQL (same
plausibility ranges, same windows, same hourly harmonisation) is then applied
to both. This removes "pipeline shift" by construction.

Windows:   vitals / GCS  0 <= off_min < 1440   (first 24 h of the ICU stay)
           labs         -360 <= off_min < 1440  (6 h before ICU admission to 24 h)

Every query restricts to the FIRST ICU stay per patient. All remaining cohort
filters (age, LOS >= 24 h, alive at 24 h, >= 1 HR value) are applied in
Python in NB03 so that a CONSORT flow can be reported.

If the NB00 dry-run fails on MIMIC time functions, set MIMIC_TIME = "TIMESTAMP".
"""

MIMIC_ICU = "physionet-data.mimiciv_3_1_icu"
MIMIC_HOSP = "physionet-data.mimiciv_3_1_hosp"
EICU = "physionet-data.eicu_crd"

MIMIC_TIME = "DATETIME"   # MIMIC-IV on BigQuery stores times as DATETIME

# --------------------------------------------------------------------------
# Harmonised variable definitions (single source of truth)
# --------------------------------------------------------------------------
VITAL_RANGES = {
    "hr": (20, 250), "sbp": (40, 300), "dbp": (10, 200), "mbp": (20, 250),
    "rr": (3, 70), "spo2": (50, 100), "temp": (25, 45),
}
LAB_RANGES = {
    "creat": (0.1, 30), "bun": (1, 300), "sodium": (90, 200), "potassium": (1, 12),
    "chloride": (50, 160), "bicarb": (2, 60), "glucose": (20, 2000),
    "wbc": (0.1, 300), "hgb": (2, 25), "plt": (2, 2000), "lactate": (0.1, 30),
    "bili": (0.1, 80), "albumin": (0.5, 7), "inr": (0.5, 20), "hct": (5, 75),
}

# MIMIC-IV itemids
MIMIC_VITAL_ITEMS = {
    "hr": [220045],
    "sbp": [220179, 220050],     # non-invasive + arterial
    "dbp": [220180, 220051],
    "mbp": [220181, 220052],
    "rr": [220210, 224690],
    "spo2": [220277],
    "temp": [223762, 223761],    # C, F (F converted below)
}
MIMIC_LAB_ITEMS = {
    "creat": [50912], "bun": [51006], "sodium": [50983], "potassium": [50971],
    "chloride": [50902], "bicarb": [50882], "glucose": [50931],
    "wbc": [51300, 51301], "hgb": [51222], "plt": [51265], "lactate": [50813],
    "bili": [50885], "albumin": [50862], "inr": [51237], "hct": [51221],
}
MIMIC_GCS_ITEMS = {"eye": 220739, "verbal": 223900, "motor": 223901}

# eICU lab names (lab table) -> harmonised var
EICU_LAB_NAMES = {
    "creat": ["creatinine"], "bun": ["BUN"], "sodium": ["sodium"],
    "potassium": ["potassium"], "chloride": ["chloride"], "bicarb": ["bicarbonate"],
    "glucose": ["glucose"], "wbc": ["WBC x 1000"], "hgb": ["Hgb"],
    "plt": ["platelets x 1000"], "lactate": ["lactate"], "bili": ["total bilirubin"],
    "albumin": ["albumin"], "inr": ["PT - INR"], "hct": ["Hct"],
}
# eICU nursecharting cell names -> harmonised var (from MIT-LCP eicu-code pivoted_vital)
EICU_NC_NAMES = {
    "hr": ["Heart Rate"], "rr": ["Respiratory Rate"], "spo2": ["O2 Saturation"],
    "sbp": ["Non-Invasive BP Systolic", "Invasive BP Systolic"],
    "dbp": ["Non-Invasive BP Diastolic", "Invasive BP Diastolic"],
    "mbp": ["Non-Invasive BP Mean", "Invasive BP Mean"],
    "temp": ["Temperature (C)"],
}


def _q(items):
    return ", ".join(f"'{x}'" for x in items)


def _ids(d):
    return ", ".join(str(i) for v in d.values() for i in v)


def _case_itemid(d, col):
    parts = [f"WHEN {col} IN ({', '.join(map(str, ids))}) THEN '{var}'" for var, ids in d.items()]
    return "CASE " + " ".join(parts) + " END"


def _case_name(d, col):
    parts = [f"WHEN {col} IN ({_q(names)}) THEN '{var}'" for var, names in d.items()]
    return "CASE " + " ".join(parts) + " END"


def _range_filter(ranges):
    return " OR ".join(f"(var = '{k}' AND v BETWEEN {lo} AND {hi})" for k, (lo, hi) in ranges.items())


def _tdiff(a, b):
    if MIMIC_TIME == "DATETIME":
        return f"DATETIME_DIFF({a}, {b}, MINUTE)"
    return f"TIMESTAMP_DIFF({a}, {b}, MINUTE)"


def _tadd(a, hours):
    if MIMIC_TIME == "DATETIME":
        return f"DATETIME_ADD({a}, INTERVAL {hours} HOUR)"
    return f"TIMESTAMP_ADD({a}, INTERVAL {hours} HOUR)"


# --------------------------------------------------------------------------
# Shared aggregation templates (identical for both databases)
# --------------------------------------------------------------------------
def _vital_agg(long_cte_sql):
    """Hourly harmonisation: first average within each clock hour since ICU
    admission, then min / max / mean over the hourly values. Raw-resolution
    aggregates (r*) and counts are kept for the pipeline-artefact ablation."""
    return f"""
WITH long AS (
{long_cte_sql}
),
filt AS (
  SELECT stay_id, off_min, var, v
  FROM long
  WHERE off_min >= 0 AND off_min < 1440 AND var IS NOT NULL AND v IS NOT NULL
    AND ({_range_filter(VITAL_RANGES)})
),
hourly AS (
  SELECT stay_id, var, DIV(off_min, 60) AS hb, AVG(v) AS v
  FROM filt GROUP BY stay_id, var, hb
),
agg_h AS (
  SELECT stay_id, var, MIN(v) AS vmin, MAX(v) AS vmax, AVG(v) AS vmean, COUNT(*) AS n_hours
  FROM hourly GROUP BY stay_id, var
),
agg_r AS (
  SELECT stay_id, var, MIN(v) AS rmin, MAX(v) AS rmax, AVG(v) AS rmean, COUNT(*) AS n_raw
  FROM filt GROUP BY stay_id, var
)
SELECT h.stay_id, h.var, h.vmin, h.vmax, h.vmean, h.n_hours, r.rmin, r.rmax, r.rmean, r.n_raw
FROM agg_h h JOIN agg_r r ON h.stay_id = r.stay_id AND h.var = r.var
"""


def _lab_agg(long_cte_sql):
    return f"""
WITH long AS (
{long_cte_sql}
),
filt AS (
  SELECT stay_id, var, v
  FROM long
  WHERE off_min >= -360 AND off_min < 1440 AND var IS NOT NULL AND v IS NOT NULL
    AND ({_range_filter(LAB_RANGES)})
)
SELECT stay_id, var, MIN(v) AS vmin, MAX(v) AS vmax, AVG(v) AS vmean, COUNT(*) AS n_raw
FROM filt GROUP BY stay_id, var
"""


# --------------------------------------------------------------------------
# MIMIC-IV
# --------------------------------------------------------------------------
def mimic_cohort():
    """ALL ICU stays with rank (rn = 1 is the first stay). Filters in NB03."""
    return f"""
WITH ic AS (
  SELECT i.subject_id, i.hadm_id, i.stay_id, i.intime, i.outtime, i.first_careunit,
         ROW_NUMBER() OVER (PARTITION BY i.subject_id ORDER BY i.intime, i.stay_id) AS rn
  FROM `{MIMIC_ICU}.icustays` i
)
SELECT ic.subject_id, ic.hadm_id, ic.stay_id, ic.rn, ic.first_careunit,
       {_tdiff('ic.outtime', 'ic.intime')} / 60.0 AS icu_los_h,
       p.gender,
       LEAST(p.anchor_age + (EXTRACT(YEAR FROM ic.intime) - p.anchor_year), 90) AS age,
       p.anchor_year_group,
       a.hospital_expire_flag AS mortality,
       a.race, a.admission_type,
       {_tdiff('a.deathtime', 'ic.intime')} / 60.0 AS hours_to_death
FROM ic
JOIN `{MIMIC_HOSP}.patients` p ON ic.subject_id = p.subject_id
JOIN `{MIMIC_HOSP}.admissions` a ON ic.hadm_id = a.hadm_id
"""


def _mimic_first_stays():
    return f"""
  SELECT stay_id, subject_id, intime FROM (
    SELECT i.stay_id, i.subject_id, i.intime,
           ROW_NUMBER() OVER (PARTITION BY i.subject_id ORDER BY i.intime, i.stay_id) AS rn
    FROM `{MIMIC_ICU}.icustays` i
  ) WHERE rn = 1"""


def mimic_vitals():
    long = f"""
  SELECT c.stay_id,
         {_tdiff('ce.charttime', 'c.intime')} AS off_min,
         {_case_itemid(MIMIC_VITAL_ITEMS, 'ce.itemid')} AS var,
         CASE WHEN ce.itemid = 223761 THEN (ce.valuenum - 32) * 5 / 9 ELSE ce.valuenum END AS v
  FROM ({_mimic_first_stays()}) c
  JOIN `{MIMIC_ICU}.chartevents` ce ON ce.stay_id = c.stay_id
  WHERE ce.itemid IN ({_ids(MIMIC_VITAL_ITEMS)})
    AND ce.valuenum IS NOT NULL
    AND ce.charttime >= c.intime AND ce.charttime < {_tadd('c.intime', 24)}"""
    return _vital_agg(long)


def mimic_labs():
    long = f"""
  SELECT c.stay_id,
         {_tdiff('le.charttime', 'c.intime')} AS off_min,
         {_case_itemid(MIMIC_LAB_ITEMS, 'le.itemid')} AS var,
         le.valuenum AS v
  FROM ({_mimic_first_stays()}) c
  JOIN `{MIMIC_HOSP}.labevents` le ON le.subject_id = c.subject_id
  WHERE le.itemid IN ({_ids(MIMIC_LAB_ITEMS)})
    AND le.valuenum IS NOT NULL
    AND le.charttime >= {_tadd('c.intime', -6)} AND le.charttime < {_tadd('c.intime', 24)}"""
    return _lab_agg(long)


def mimic_gcs():
    e, vb, m = MIMIC_GCS_ITEMS["eye"], MIMIC_GCS_ITEMS["verbal"], MIMIC_GCS_ITEMS["motor"]
    return f"""
WITH g AS (
  SELECT c.stay_id, ce.charttime,
         MAX(CASE WHEN ce.itemid = {e} THEN ce.valuenum END) AS eye,
         MAX(CASE WHEN ce.itemid = {vb} THEN ce.valuenum END) AS verbal,
         MAX(CASE WHEN ce.itemid = {m} THEN ce.valuenum END) AS motor
  FROM ({_mimic_first_stays()}) c
  JOIN `{MIMIC_ICU}.chartevents` ce ON ce.stay_id = c.stay_id
  WHERE ce.itemid IN ({e}, {vb}, {m}) AND ce.valuenum IS NOT NULL
    AND ce.charttime >= c.intime AND ce.charttime < {_tadd('c.intime', 24)}
  GROUP BY c.stay_id, ce.charttime
)
SELECT stay_id, MIN(eye + verbal + motor) AS gcs_min, COUNT(*) AS gcs_n
FROM g
WHERE eye + verbal + motor BETWEEN 3 AND 15
GROUP BY stay_id
"""


# --------------------------------------------------------------------------
# eICU-CRD
# --------------------------------------------------------------------------
def eicu_cohort():
    return f"""
SELECT p.patientunitstayid AS stay_id, p.patienthealthsystemstayid, p.uniquepid,
       p.hospitalid AS hospital_id, p.unitvisitnumber,
       p.hospitaldischargeyear AS year,
       CASE WHEN p.age = '> 89' THEN 90 ELSE SAFE_CAST(p.age AS INT64) END AS age,
       p.gender, p.ethnicity, p.unittype, p.unitadmitsource,
       p.unitdischargeoffset, p.hospitaldischargeoffset,
       p.hospitaldischargestatus, p.unitdischargestatus,
       ROW_NUMBER() OVER (PARTITION BY p.uniquepid
                          ORDER BY p.hospitaldischargeyear, p.patienthealthsystemstayid,
                                   p.unitvisitnumber, p.patientunitstayid) AS rn
FROM `{EICU}.patient` p
"""


def _eicu_first_stays():
    return f"""
  SELECT stay_id FROM (
    SELECT p.patientunitstayid AS stay_id,
           ROW_NUMBER() OVER (PARTITION BY p.uniquepid
                              ORDER BY p.hospitaldischargeyear, p.patienthealthsystemstayid,
                                       p.unitvisitnumber, p.patientunitstayid) AS rn
    FROM `{EICU}.patient` p
  ) WHERE rn = 1"""


def eicu_vitals():
    nc_names = [n for v in EICU_NC_NAMES.values() for n in v] + ["Temperature (F)"]
    long = f"""
  SELECT x.stay_id, x.off_min, x.var, x.v
  FROM (
    -- 1) vitalperiodic (automated 5-min medians, incl. invasive BP)
    SELECT stay_id, off_min, var, v FROM (
      SELECT patientunitstayid AS stay_id, observationoffset AS off_min,
             CAST(heartrate AS FLOAT64) AS hr, CAST(respiration AS FLOAT64) AS rr,
             CAST(sao2 AS FLOAT64) AS spo2, CAST(temperature AS FLOAT64) AS temp,
             CAST(systemicsystolic AS FLOAT64) AS sbp, CAST(systemicdiastolic AS FLOAT64) AS dbp,
             CAST(systemicmean AS FLOAT64) AS mbp
      FROM `{EICU}.vitalperiodic`
      WHERE observationoffset >= 0 AND observationoffset < 1440
    ) UNPIVOT (v FOR var IN (hr, rr, spo2, temp, sbp, dbp, mbp))
    UNION ALL
    -- 2) vitalaperiodic (non-invasive BP)
    SELECT stay_id, off_min, var, v FROM (
      SELECT patientunitstayid AS stay_id, observationoffset AS off_min,
             CAST(noninvasivesystolic AS FLOAT64) AS sbp,
             CAST(noninvasivediastolic AS FLOAT64) AS dbp,
             CAST(noninvasivemean AS FLOAT64) AS mbp
      FROM `{EICU}.vitalaperiodic`
      WHERE observationoffset >= 0 AND observationoffset < 1440
    ) UNPIVOT (v FOR var IN (sbp, dbp, mbp))
    UNION ALL
    -- 3) nursecharting (manually charted / validated values)
    SELECT patientunitstayid AS stay_id, nursingchartoffset AS off_min,
           CASE WHEN nursingchartcelltypevalname = 'Temperature (F)' THEN 'temp'
                ELSE {_case_name(EICU_NC_NAMES, 'nursingchartcelltypevalname')} END AS var,
           CASE WHEN nursingchartcelltypevalname = 'Temperature (F)'
                THEN (SAFE_CAST(nursingchartvalue AS FLOAT64) - 32) * 5 / 9
                ELSE SAFE_CAST(nursingchartvalue AS FLOAT64) END AS v
    FROM `{EICU}.nursecharting`
    WHERE nursingchartcelltypevalname IN ({_q(nc_names)})
      AND nursingchartoffset >= 0 AND nursingchartoffset < 1440
  ) x
  JOIN ({_eicu_first_stays()}) c ON c.stay_id = x.stay_id"""
    return _vital_agg(long)


def eicu_labs():
    names = [n for v in EICU_LAB_NAMES.values() for n in v]
    long = f"""
  SELECT l.patientunitstayid AS stay_id, l.labresultoffset AS off_min,
         {_case_name(EICU_LAB_NAMES, 'l.labname')} AS var,
         CAST(l.labresult AS FLOAT64) AS v
  FROM `{EICU}.lab` l
  JOIN ({_eicu_first_stays()}) c ON c.stay_id = l.patientunitstayid
  WHERE l.labname IN ({_q(names)})
    AND l.labresultoffset >= -360 AND l.labresultoffset < 1440"""
    return _lab_agg(long)


def eicu_gcs():
    # logic follows MIT-LCP eicu-code concepts/pivoted/pivoted-gcs.sql
    return f"""
WITH g AS (
  SELECT nc.patientunitstayid AS stay_id, nc.nursingchartoffset AS off_min,
    MIN(CASE
      WHEN nc.nursingchartcelltypevallabel = 'Glasgow coma score'
       AND nc.nursingchartcelltypevalname = 'GCS Total'
        THEN SAFE_CAST(nc.nursingchartvalue AS FLOAT64)
      WHEN nc.nursingchartcelltypevallabel = 'Score (Glasgow Coma Scale)'
       AND nc.nursingchartcelltypevalname = 'Value'
        THEN SAFE_CAST(nc.nursingchartvalue AS FLOAT64) END) AS gcs_total,
    MIN(CASE WHEN nc.nursingchartcelltypevallabel = 'Glasgow coma score'
              AND nc.nursingchartcelltypevalname = 'Motor'
             THEN SAFE_CAST(nc.nursingchartvalue AS FLOAT64) END) AS motor,
    MIN(CASE WHEN nc.nursingchartcelltypevallabel = 'Glasgow coma score'
              AND nc.nursingchartcelltypevalname = 'Verbal'
             THEN SAFE_CAST(nc.nursingchartvalue AS FLOAT64) END) AS verbal,
    MIN(CASE WHEN nc.nursingchartcelltypevallabel = 'Glasgow coma score'
              AND nc.nursingchartcelltypevalname = 'Eyes'
             THEN SAFE_CAST(nc.nursingchartvalue AS FLOAT64) END) AS eyes
  FROM `{EICU}.nursecharting` nc
  JOIN ({_eicu_first_stays()}) c ON c.stay_id = nc.patientunitstayid
  WHERE nc.nursingchartcelltypecat IN ('Scores', 'Other Vital Signs and Infusions')
    AND nc.nursingchartoffset >= 0 AND nc.nursingchartoffset < 1440
  GROUP BY stay_id, off_min
),
g2 AS (SELECT stay_id, COALESCE(gcs_total, eyes + verbal + motor) AS gcs FROM g)
SELECT stay_id, MIN(gcs) AS gcs_min, COUNT(*) AS gcs_n
FROM g2 WHERE gcs BETWEEN 3 AND 15
GROUP BY stay_id
"""


def eicu_hospital():
    return f"""
SELECT hospitalid AS hospital_id, numbedscategory, teachingstatus, region
FROM `{EICU}.hospital`
"""


def eicu_apache():
    return f"""
SELECT patientunitstayid AS stay_id,
       SAFE_CAST(predictedhospitalmortality AS FLOAT64) AS apache_pred,
       SAFE_CAST(apachescore AS FLOAT64) AS apache_score
FROM `{EICU}.apachepatientresult`
WHERE apacheversion = 'IVa'
"""


# --------------------------------------------------------------------------
QUERIES = {
    "mimic_cohort": mimic_cohort, "mimic_vitals": mimic_vitals,
    "mimic_labs": mimic_labs, "mimic_gcs": mimic_gcs,
    "eicu_cohort": eicu_cohort, "eicu_vitals": eicu_vitals,
    "eicu_labs": eicu_labs, "eicu_gcs": eicu_gcs,
    "eicu_hospital": eicu_hospital, "eicu_apache": eicu_apache,
}


def dry_run_all(client, names=None):
    """Validate every query on BigQuery without running it (no cost).
    Returns a list of dicts: name, ok, gb_processed, error."""
    from google.cloud import bigquery
    out = []
    for name in (names or QUERIES):
        cfg = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        try:
            job = client.query(QUERIES[name](), job_config=cfg)
            out.append(dict(name=name, ok=True,
                            gb_processed=round(job.total_bytes_processed / 1e9, 2), error=""))
        except Exception as ex:  # noqa: BLE001
            out.append(dict(name=name, ok=False, gb_processed=None, error=str(ex)[:600]))
    return out


def run_to_parquet(client, name, path):
    """Run one query and save the result as parquet. Returns the DataFrame."""
    df = client.query(QUERIES[name]()).to_dataframe()
    df.to_parquet(path, index=False)
    return df
