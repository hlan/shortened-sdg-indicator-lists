# Code and Data for:

## Reexamining Shortened Lists of Sustainable Development Indicators: Functional Redundancy for Ranking versus Diagnostic Precision

This repository contains the code used in the study titled:

**Reexamining Shortened Lists of Sustainable Development Indicators: Functional Redundancy for Ranking versus Diagnostic Precision**,  
submitted to **Geography and Sustainability**.

---

## 📁 Repository Structure

### 1. Data Sources
All analyses rely on publicly available datasets obtained from the following sources:

- **SDG Index data (tabular)**
  - Sustainable Development Report (SDR), editions 2022 to 2026: indicator values and scores, goal scores, and the SDG Index score and rank of 193 countries.  
    Reports and data: https://sdgtransformationcenter.org/sustainable-development-report  
    Data hub: https://datahub.sdgtransformationcenter.org
  - Editions 2023 to 2026 are read from the files "Sustainable Development Report 20xx (with indicators)" in CSV format, for example the 2025 edition:  
    https://sdg-transformation-center-sdsn.hub.arcgis.com/datasets/sdsn::sustainable-development-report-2025-with-indicators/about
  - Edition 2022 is read from the database workbook `SDR-2022-database.xlsx`.

- **Population (tabular)**
  - World Development Indicators, total population (SP.POP.TOTL):  
    https://data.worldbank.org/indicator/SP.POP.TOTL

- **Country boundaries (vector)**
  - Natural Earth, 1:50m Admin 0 countries (public domain), file `ne_50m_admin_0_countries.geojson`:  
    https://github.com/nvkelso/natural-earth-vector (folder `geojson`)

---

### 2. Code

The analysis has six scripts that run in order. Every script is run from the folder that contains `data/`, reads the results of the earlier scripts, and writes to `result/<number of the script>/`.

- **Step 1: Data loading, index reproduction and tolerances**  
  Reads the five editions, scores every indicator from 0 to 100 with the bounds of the report, rebuilds the SDG Index and checks it against the official scores and ranks. Sets the tolerances of the country-level criteria from 200 disturbances of the full indicator set.
  - `1_load_data_and_calibrate_tolerances.py`

- **Step 2: Audit of the headline index**  
  Computes the headline index of the report (17 indicators, one for each goal) and compares it with the full SDG Index: overall ranking, individual rank stability and weakest goal diagnosis. Also compares how often each goal is named as weakest with how often the goal is truly weakest, measures the level bias of each headline indicator, and reports the results by income group and region.
  - `2_audit_headline_index.py`

- **Step 3: Search for the smallest indicator lists across goals**  
  Searches the full indicator set of each edition for the smallest lists that meet each criterion (overall ranking, individual rank stability, weakest goal diagnosis, dual criteria), with random starts, local search and exchanges. Records the best lists found at fixed sizes.
  - `3_search_smallest_lists_across_goals.py`

- **Step 4: Indicator lists within goals and mechanisms**  
  Evaluates every subset of the indicators of each goal. Derives the fewest indicators each criterion needs, tests lists chosen by explained variance, relates the requirements to the country-specific part of the scores, tests whether dropped indicators can be predicted from the kept ones, and repeats the ranking lists under other averaging rules.
  - `4_analyze_within_goals_and_mechanisms.py`

- **Step 5: Weaknesses hidden by missing data**  
  Estimates, for every missing value, the probability that the value hides the weakest indicator of a country in a goal. Sums the probabilities by indicator and by country, compares groups of countries, and tests the estimates by hiding observed values.
  - `5_estimate_weaknesses_hidden_by_missing_data.py`

- **Step 6: Figures, tables and numbers quoted in the text**  
  Reads the results of steps 1 to 5 and produces the figures, the tables and the list of numbers quoted in the manuscript.
  - `6_make_figures_and_tables.py`

---

### 3. Computing Environment

The analysis runs on a CPU; no GPU is needed.

- **Key Software Version**
  - Python: 3.12
  - NumPy: 2.3.5
  - pandas: 2.2.3
  - SciPy: 1.15.1
  - Matplotlib: 3.9.1
  - GeoPandas: 1.0.1

---

## Citation

If you reference this work, please cite the associated manuscript:

> Reexamining Shortened Lists of Sustainable Development Indicators: Functional Redundancy for Ranking versus Diagnostic Precision.

Full citation details will be added upon publication.
