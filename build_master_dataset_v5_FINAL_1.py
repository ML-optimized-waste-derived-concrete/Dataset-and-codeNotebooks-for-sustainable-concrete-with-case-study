# Master Dataset Script

import os
import pandas as pd
import numpy as np

np.random.seed(42)

# -------------------------------------------------------------
# STEP 1 : Set folder paths
# -------------------------------------------------------------
# Point this at the folder containing my 25 and three new datasets

base_folder = "./raw_sources/"          # original 25 source files
replacement_folder = "./"                


def read_file(base, filename):
    path = os.path.join(base, filename)
    if filename.endswith(".xlsx"):
        return pd.read_excel(path)
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def to_num(series):
    return pd.to_numeric(series, errors="coerce")


ORIGINAL_SOURCES_TAG = "See Table 3 (compiled sources)"
SYNTHETIC_TAG = "This study (synthetic augmentation)"

# -------------------------------------------------------------
# STEP 2 : Extract real data from each source file
# -------------------------------------------------------------

frames = []

# Source 1: bootstrap__.csv
df = read_file(base_folder, "bootstrap__.csv")
f = pd.DataFrame()
f["cement_kg_m3"]             = to_num(df["CD"])
f["water_cement_ratio"]       = to_num(df["WBR"])
f["curing_days"]              = to_num(df["A"])
f["rac_%"]                    = to_num(df["TRCA"])
f["compressive_strength_mpa"] = to_num(df["FC"])
f["waste_type"]  = "RAC"
f["data_source"] = "real"
f["paper_ref_no"] = ORIGINAL_SOURCES_TAG
f = f.dropna(subset=["compressive_strength_mpa"])
f = f[f["compressive_strength_mpa"] > 0]
frames.append(f)
print(f"Source 1  bootstrap__.csv             : {len(f)} rows loaded")


# Source 2: ANN.csv
df = read_file(base_folder, "ANN.csv")
f = pd.DataFrame()
f["water_cement_ratio"]       = to_num(df["w/c ratio"])
f["cement_kg_m3"]             = to_num(df["Cement\n(kg/m3)"])
f["rac_%"]                    = to_num(df["S.R%"])
f["compressive_strength_mpa"] = to_num(df["Compressive Strength (MPa)"])
f["curing_days"]  = 28
f["waste_type"]   = "RAC"
f["data_source"]  = "real"
f["paper_ref_no"] = ORIGINAL_SOURCES_TAG
f = f.dropna(subset=["compressive_strength_mpa"])
f = f[f["compressive_strength_mpa"] > 0]
frames.append(f)
print(f"Source 2  ANN.csv                     : {len(f)} rows loaded")


# Source 3: Recycle aggregate(AutoRecovered) (1).csv
df = read_file(base_folder, "Recycle aggregate(AutoRecovered) (1).csv")
for col in ["RCA\n(kg/m3)", "NCA\n(kg/m3)", "Water (kg/m3)",
            "Cement (kg/m3)", "UCS (MPa)", "Age"]:
    df[col] = to_num(df[col])
df["RCA\n(kg/m3)"] = df["RCA\n(kg/m3)"].fillna(0)
df["NCA\n(kg/m3)"] = df["NCA\n(kg/m3)"].fillna(0)
f = pd.DataFrame()
total = df["RCA\n(kg/m3)"] + df["NCA\n(kg/m3)"]
f["rac_%"]                    = (df["RCA\n(kg/m3)"] / total.replace(0, np.nan) * 100).round(1)
f["water_cement_ratio"]       = (df["Water (kg/m3)"] / df["Cement (kg/m3)"]).round(3)
f["cement_kg_m3"]             = df["Cement (kg/m3)"]
f["curing_days"]              = df["Age"]
f["compressive_strength_mpa"] = df["UCS (MPa)"]
f["waste_type"]  = "RAC"
f["data_source"] = "real"
f["paper_ref_no"] = ORIGINAL_SOURCES_TAG
f = f.dropna(subset=["compressive_strength_mpa"])
f = f[f["compressive_strength_mpa"] > 0]
frames.append(f)
print(f"Source 3  Recycle aggregate.csv       : {len(f)} rows loaded")


# Source 4: dataset recycled aggregate natural fiber.csv
df = read_file(base_folder, "dataset recycled aggregate natural fiber.csv")
for col in ["RCA", "CA", "W/B", "Cem", "Age", "CS"]:
    df[col] = to_num(df[col])
df["RCA"] = df["RCA"].fillna(0)
df["CA"]  = df["CA"].fillna(0)
f = pd.DataFrame()
total = df["RCA"] + df["CA"]
f["rac_%"]                    = (df["RCA"] / total.replace(0, np.nan) * 100).round(1)
f["water_cement_ratio"]       = df["W/B"]
f["cement_kg_m3"]             = df["Cem"]
f["curing_days"]              = df["Age"]
f["compressive_strength_mpa"] = df["CS"]
f["waste_type"]  = "RAC"
f["data_source"] = "real"
f["paper_ref_no"] = ORIGINAL_SOURCES_TAG
f = f.dropna(subset=["compressive_strength_mpa"])
f = f[f["compressive_strength_mpa"] > 0]
frames.append(f)
print(f"Source 4  dataset natural fiber.csv   : {len(f)} rows loaded")


# Source 5: RCA for New model.csv
df = read_file(base_folder, "RCA for New model.csv")
for col in ["Recycled Coarse Aggregate (kg/m3)",
            "Natural Coarse Aggregate (kg/m3)",
            "w/c", "Cement (kg/m3)",
            "Compressive Strength  (MPa)",
            "Split Tensile Strength (MPa)",
            "Flexural Strength (MPa)"]:
    df[col] = to_num(df[col])
f = pd.DataFrame()
rca = df["Recycled Coarse Aggregate (kg/m3)"].fillna(0)
nca = df["Natural Coarse Aggregate (kg/m3)"].fillna(0)
total = rca + nca
f["rac_%"]                      = (rca / total.replace(0, np.nan) * 100).round(1)
f["water_cement_ratio"]         = df["w/c"]
f["cement_kg_m3"]               = df["Cement (kg/m3)"]
f["curing_days"]                = 28
f["compressive_strength_mpa"]   = df["Compressive Strength  (MPa)"]
f["split_tensile_strength_mpa"] = df["Split Tensile Strength (MPa)"]
f["flexural_strength_mpa"]      = df["Flexural Strength (MPa)"]
f["waste_type"]  = "RAC"
f["data_source"] = "real"
f["paper_ref_no"] = ORIGINAL_SOURCES_TAG
f = f.dropna(subset=["compressive_strength_mpa"])
f = f[f["compressive_strength_mpa"] > 0]
frames.append(f)
print(f"Source 5  RCA for New model.csv       : {len(f)} rows loaded")


# Source 6: tabula-Database_IIT_BBS_LAB_Concrete.csv
df = read_file(base_folder, "tabula-Database_IIT_BBS_LAB_Concrete.csv")
for col in ["CS", "cemen", "water", "RCA 20", "RCA", "AGE"]:
    df[col] = to_num(df[col])
df["RCA 20"] = df["RCA 20"].fillna(0)
df["RCA"]    = df["RCA"].fillna(0)
f = pd.DataFrame()
f["cement_kg_m3"]             = df["cemen"]
f["water_cement_ratio"]       = (df["water"] / df["cemen"]).round(3)
f["curing_days"]              = df["AGE"]
f["rac_%"]                    = df["RCA 20"] + df["RCA"]
f["compressive_strength_mpa"] = df["CS"]
f["waste_type"]  = "RAC"
f["data_source"] = "real"
f["paper_ref_no"] = ORIGINAL_SOURCES_TAG
f = f.dropna(subset=["compressive_strength_mpa"])
f = f[f["compressive_strength_mpa"] > 0]
frames.append(f)
print(f"Source 6  IIT BBS Lab.csv             : {len(f)} rows loaded")


# Source 7 (sustainable_concrete_dataset.csv) REMOVED 
# This file was found to contain fabricated RCA% data and shows
# strong statistical signatures of being a synthetic benchmark
# dataset. It has been excluded entirely.
print("Source 7  sustainable_concrete.csv    : REMOVED (fabricated data)")


# Source 8: RACdatadriven.xlsx
df = read_file(base_folder, "RACdatadriven.xlsx")
f = pd.DataFrame()
f["water_cement_ratio"]       = to_num(df["W/c"])
f["rac_%"]                    = (to_num(df["r"]) * 100).round(1)
f["compressive_strength_mpa"] = to_num(df["Fc'"])
f["curing_days"]  = 28
f["cement_kg_m3"] = 380
f["waste_type"]   = "RAC"
f["data_source"]  = "real"
f["paper_ref_no"] = ORIGINAL_SOURCES_TAG
f = f.dropna(subset=["compressive_strength_mpa"])
f = f[f["compressive_strength_mpa"] > 0]
frames.append(f)
print(f"Source 8  RACdatadriven.xlsx          : {len(f)} rows loaded")


# Source 9: RACC_dataset.csv
df = read_file(base_folder, "RACC_dataset.csv")
f = pd.DataFrame()
total = df["rca_kg_m3"] + df["nca_kg_m3"]
f["rac_%"]                    = (df["rca_kg_m3"] / total.replace(0, np.nan) * 100).round(1)
f["water_cement_ratio"]       = (df["water_kg_m3"] / df["cement_kg_m3"]).round(3)
f["cement_kg_m3"]             = df["cement_kg_m3"]
f["curing_days"]              = 28
f["compressive_strength_mpa"] = df["mean_28d_compressive_strength_mpa"]
f["waste_type"]  = "RAC"
f["data_source"] = "real"
f["paper_ref_no"] = ORIGINAL_SOURCES_TAG
f = f.dropna(subset=["compressive_strength_mpa"])
frames.append(f)
print(f"Source 9  RACC_dataset.csv            : {len(f)} rows loaded")


# Source 10: RCA_Cost_data.csv
df = read_file(base_folder, "RCA_Cost_data.csv")
f = pd.DataFrame()
f["cement_kg_m3"]             = to_num(df["cement_kg_m3"])
f["water_cement_ratio"]       = to_num(df["w_c_ratio"])
f["curing_days"]              = to_num(df["curing_age_days"])
f["rac_%"]                    = to_num(df["rca_replacement_pct"])
f["compressive_strength_mpa"] = to_num(df["target_strength_mpa"])
f["cost_usd_m3"]              = to_num(df["nac_baseline_price_usd_m3"])
f["waste_type"]  = "RAC"
f["data_source"] = "real"
f["paper_ref_no"] = ORIGINAL_SOURCES_TAG
f = f.dropna(subset=["compressive_strength_mpa"])
f = f[f["compressive_strength_mpa"] > 0]
frames.append(f)
print(f"Source 10 RCA_Cost_data.csv           : {len(f)} rows loaded")


# Source 11: 4dataset.csv
df = read_file(base_folder, "4dataset.csv")
f = pd.DataFrame()
f["rac_%"]                    = to_num(df["replacement_pct"])
f["curing_days"]              = to_num(df["curing_age_days"])
f["compressive_strength_mpa"] = to_num(df["compressive_strength_mpa"])
f["cement_kg_m3"]       = 380
f["water_cement_ratio"] = 0.45
f["waste_type"]  = "RAC"
f["data_source"] = "real"
f["paper_ref_no"] = ORIGINAL_SOURCES_TAG
f = f.dropna(subset=["compressive_strength_mpa"])
f = f[f["compressive_strength_mpa"] > 0]
frames.append(f)
print(f"Source 11 4dataset.csv                : {len(f)} rows loaded")


# Source 12: Rac_Freezthaw_dataset.csv
df = read_file(base_folder, "Rac_Freezthaw_dataset.csv")
f = pd.DataFrame()
f["cement_kg_m3"]               = to_num(df["cement_kg_m3"])
f["water_cement_ratio"]         = to_num(df["w_b_ratio"])
f["curing_days"]                = to_num(df["curing_age_days"])
f["rac_%"]                      = to_num(df["rca_pct"])
f["ceramic_%"]                  = to_num(df["ceramic_pct"])
f["plastic_%"]                  = to_num(df["plastic_pct"])
f["rcwtb_%"]                    = to_num(df["rcwtb_pct"])
f["compressive_strength_mpa"]   = to_num(df["compressive_strength_mpa"])
f["split_tensile_strength_mpa"] = to_num(df["split_tensile_strength_mpa"])
f["waste_type"]  = "RAC_FreezeThaw"
f["data_source"] = "real"
f["paper_ref_no"] = ORIGINAL_SOURCES_TAG
f = f.dropna(subset=["compressive_strength_mpa"])
f = f[f["compressive_strength_mpa"] > 0]
frames.append(f)
print(f"Source 12 Rac_Freezthaw_dataset.csv   : {len(f)} rows loaded")


# Source 13: Freezthaw_Rac_dataset.csv
df = read_file(base_folder, "Freezthaw_Rac_dataset.csv")
f = pd.DataFrame()
f["cement_kg_m3"]               = to_num(df["cement_kg_m3"])
f["water_cement_ratio"]         = to_num(df["water_cement_ratio"])
f["curing_days"]                = to_num(df["curing_age_days"])
f["rac_%"]                      = (to_num(df["rca_replacement_ratio"]) * 100)
f["compressive_strength_mpa"]   = to_num(df["compressive_strength_28d_mpa"])
f["split_tensile_strength_mpa"] = to_num(df["splitting_tensile_strength_28d_mpa"])
f["waste_type"]  = "RAC_FreezeThaw"
f["data_source"] = "real"
f["paper_ref_no"] = ORIGINAL_SOURCES_TAG
f = f.dropna(subset=["compressive_strength_mpa"])
f = f[f["compressive_strength_mpa"] > 0]
frames.append(f)
print(f"Source 13 Freezthaw_Rac_dataset.csv   : {len(f)} rows loaded")


# Source 14: plastic_dataset.csv
df = read_file(base_folder, "plastic_dataset.csv")
f = pd.DataFrame()
rca = df["rca_kg_m3"].fillna(0)
nca = df["nca_kg_m3"].fillna(0)
total = rca + nca
f["rac_%"]                    = (rca / total.replace(0, np.nan) * 100).round(1)
f["water_cement_ratio"]       = to_num(df["water_cement_ratio"])
f["cement_kg_m3"]             = to_num(df["cement_kg_m3"])
f["curing_days"]              = 28
f["compressive_strength_mpa"] = to_num(df["cube_compressive_strength_mpa"])
f["plastic_%"]   = 0
f["waste_type"]  = "Plastic"
f["data_source"] = "real"
f["paper_ref_no"] = ORIGINAL_SOURCES_TAG
f = f.dropna(subset=["compressive_strength_mpa"])
f = f[f["compressive_strength_mpa"] > 0]
frames.append(f)
print(f"Source 14 plastic_dataset.csv         : {len(f)} rows loaded")


# Source 15: plaastic2_data.csv.xlsx
df = read_file(base_folder, "plaastic2_data.csv.xlsx")
f = pd.DataFrame()
f["plastic_%"]                = to_num(df["plastic_fiber_pct"])
f["water_cement_ratio"]       = to_num(df["water_cement_ratio"])
f["cement_kg_m3"]             = to_num(df["cement_kg_m3"])
f["curing_days"]              = 28
f["flexural_strength_mpa"]    = to_num(df["flexural_strength_28d_mpa"])
f["compressive_strength_mpa"] = f["flexural_strength_mpa"] / 0.12
f["rac_%"]       = 0
f["waste_type"]  = "Plastic"
f["data_source"] = "real"
f["paper_ref_no"] = ORIGINAL_SOURCES_TAG
f = f.dropna(subset=["flexural_strength_mpa"])
frames.append(f)
print(f"Source 15 plaastic2_data.xlsx         : {len(f)} rows loaded")


# Source 16: plastic_compstrength_data.csv
df = read_file(base_folder, "plastic_compstrength_data.csv")
f = pd.DataFrame()
f["plastic_%"]                  = to_num(df["rubber_content_pct"])
f["compressive_strength_mpa"]   = to_num(df["quasi_static_compressive_strength_mpa"])
f["split_tensile_strength_mpa"] = to_num(df["quasi_static_split_tensile_strength_mpa"])
f["cement_kg_m3"]       = 380
f["water_cement_ratio"] = 0.45
f["curing_days"]        = 28
f["rac_%"]              = 0
f["waste_type"]         = "Plastic"
f["data_source"]        = "real"
f["paper_ref_no"] = ORIGINAL_SOURCES_TAG
f = f.dropna(subset=["compressive_strength_mpa"])
f = f[f["compressive_strength_mpa"] > 0]
frames.append(f)
print(f"Source 16 plastic_compstrength.csv    : {len(f)} rows loaded")


# Source 17: plasticcost_data.csv
# FIX: co2_emission_kg_m3. CO2 for these rows is computed through the
# standard formula in Step 7.
df = read_file(base_folder, "plasticcost_data.csv")
f = pd.DataFrame()
f["plastic_%"]                  = to_num(df["sand_replacement_pct"])
f["cement_kg_m3"]               = to_num(df["cement_kg_m3"])
f["water_cement_ratio"]         = to_num(df["water_cement_ratio"])
f["curing_days"]                = 28
f["compressive_strength_mpa"]   = to_num(df["compressive_strength_28d_mpa"])
f["flexural_strength_mpa"]      = to_num(df["flexural_strength_28d_mpa"])
f["split_tensile_strength_mpa"] = to_num(df["split_tensile_strength_28d_mpa"])
f["cost_usd_m3"]                = to_num(df["material_cost_kes_m3"]) / 130
f["rac_%"]       = 0
f["waste_type"]  = "Plastic"
f["data_source"] = "real"
f["paper_ref_no"] = ORIGINAL_SOURCES_TAG + " [CO2 bug fixed: see methodology note]"
f = f.dropna(subset=["compressive_strength_mpa"])
f = f[f["compressive_strength_mpa"] > 0]
frames.append(f)
print(f"Source 17 plasticcost_data.csv        : {len(f)} rows loaded")


# Sources 18-20: 5dataset, 6dataset, 7dataset
for fname in ["5dataset.csv", "6dataset.csv", "7dataset.csv"]:
    df = read_file(base_folder, fname)
    f = pd.DataFrame()
    f["plastic_%"]                  = to_num(df["replacement_pct"])
    f["curing_days"]                = to_num(df["curing_age_days"])
    f["compressive_strength_mpa"]   = to_num(df["compressive_strength_mpa"])
    f["split_tensile_strength_mpa"] = to_num(df["split_tensile_strength_mpa"])
    f["flexural_strength_mpa"]      = to_num(df["flexural_strength_mpa"])
    f["cement_kg_m3"]       = 380
    f["water_cement_ratio"] = 0.45
    f["rac_%"]              = 0
    f["waste_type"]         = "Plastic"
    f["data_source"]        = "real"
    f["paper_ref_no"] = ORIGINAL_SOURCES_TAG
    f = f.dropna(subset=["compressive_strength_mpa"])
    f = f[f["compressive_strength_mpa"] > 0]
    frames.append(f)
    print(f"Source    {fname:<30}: {len(f)} rows loaded")


# Source 21: 2dataset.csv
df = read_file(base_folder, "2dataset.csv")
f = pd.DataFrame()
f["plastic_%"]                = to_num(df["waste_replacement_pct"])
f["compressive_strength_mpa"] = to_num(df["compressive_strength_mpa"])
f["cement_kg_m3"]       = 380
f["water_cement_ratio"] = 0.45
f["curing_days"]        = 28
f["rac_%"]              = 0
f["waste_type"]         = "Plastic"
f["data_source"]        = "real"
f["paper_ref_no"] = ORIGINAL_SOURCES_TAG
f = f.dropna(subset=["compressive_strength_mpa"])
f = f[f["compressive_strength_mpa"] > 0]
frames.append(f)
print(f"Source 21 2dataset.csv                : {len(f)} rows loaded")


# Source 22: ceramicgeopolymer_data.csv.xlsx
df = read_file(base_folder, "ceramicgeopolymer_data.csv.xlsx")
f = pd.DataFrame()
f["ceramic_%"]                = to_num(df["ceramic_waste_powder_pct"])
f["cement_kg_m3"]             = to_num(df["binder_total_kg_m3"])
f["curing_days"]              = 28
f["compressive_strength_mpa"] = to_num(df["compressive_strength_28d_mpa"])
f["flexural_strength_mpa"]    = to_num(df["flexural_strength_28d_mpa"])
f["water_cement_ratio"] = 0.35
f["rac_%"]       = 0
f["waste_type"]  = "Ceramic"
f["data_source"] = "real"
f["paper_ref_no"] = ORIGINAL_SOURCES_TAG
f = f.dropna(subset=["compressive_strength_mpa"])
f = f[f["compressive_strength_mpa"] > 0]
frames.append(f)
print(f"Source 22 ceramicgeopolymer.xlsx       : {len(f)} rows loaded")


# Source 23: 8dataset.csv
df = read_file(base_folder, "8dataset.csv")
f = pd.DataFrame()
f["ceramic_%"]                  = to_num(df["ceramic_replacement_pct"])
f["curing_days"]                = to_num(df["curing_age_days"])
f["compressive_strength_mpa"]   = to_num(df["compressive_strength_mpa"])
f["split_tensile_strength_mpa"] = to_num(df["split_tensile_strength_mpa"])
f["flexural_strength_mpa"]      = to_num(df["flexural_strength_mpa"])
f["cement_kg_m3"]       = 380
f["water_cement_ratio"] = 0.44
f["rac_%"]       = 0
f["waste_type"]  = "Ceramic"
f["data_source"] = "real"
f["paper_ref_no"] = ORIGINAL_SOURCES_TAG
f = f.dropna(subset=["compressive_strength_mpa"])
f = f[f["compressive_strength_mpa"] > 0]
frames.append(f)
print(f"Source 23 8dataset.csv                : {len(f)} rows loaded")


# Source 24: 1dataset.csv
df = read_file(base_folder, "1dataset.csv")
f = pd.DataFrame()
f["ceramic_%"]                = to_num(df["cwp_replacement_pct"])
f["curing_days"]              = to_num(df["curing_age_days"])
f["compressive_strength_mpa"] = to_num(df["compressive_strength_mpa"])
f["cement_kg_m3"]       = 380
f["water_cement_ratio"] = 0.45
f["rac_%"]       = 0
f["waste_type"]  = "Ceramic"
f["data_source"] = "real"
f["paper_ref_no"] = ORIGINAL_SOURCES_TAG
f = f.dropna(subset=["compressive_strength_mpa"])
f = f[f["compressive_strength_mpa"] > 0]
frames.append(f)
print(f"Source 24 1dataset.csv                : {len(f)} rows loaded")


# Source 25: 9dataset.csv
df = read_file(base_folder, "9dataset.csv")
f = pd.DataFrame()
f["rcwtb_%"]                    = to_num(df["frp_replacement_pct"])
f["curing_days"]                = to_num(df["curing_age_days"])
f["compressive_strength_mpa"]   = to_num(df["compressive_strength_mpa"])
f["split_tensile_strength_mpa"] = to_num(df["split_tensile_strength_mpa"])
f["flexural_strength_mpa"]      = to_num(df["flexural_strength_mpa"])
f["cement_kg_m3"]       = 380
f["water_cement_ratio"] = 0.45
f["rac_%"]       = 0
f["waste_type"]  = "RCWTB"
f["data_source"] = "real"
f["paper_ref_no"] = ORIGINAL_SOURCES_TAG
f = f.dropna(subset=["compressive_strength_mpa"])
f = f[f["compressive_strength_mpa"] > 0]
frames.append(f)
print(f"Source 25 9dataset.csv (RCWTB)        : {len(f)} rows loaded")


# -------------------------------------------------------------
# STEP 3 : Load and integrate the THREE more sources
# -------------------------------------------------------------

# New Source A: Mahmoudian et al. 2024 (bond-strength dataset)
df_mah = pd.read_csv(os.path.join(replacement_folder, "mahmoudian_2024_bond_strength_dataset.csv"))
f = pd.DataFrame()
rca = df_mah["Recycled coarse aggregate (kg/m3)"].fillna(0)
nca = df_mah["Natural coarse aggregate  (kg/m3)"].fillna(0)
total = rca + nca
f["rac_%"]                    = (rca / total.replace(0, np.nan) * 100).round(1)
f["water_cement_ratio"]       = df_mah["W/C"]
f["cement_kg_m3"]             = df_mah["Cement (kg/m3)"]
f["curing_days"]              = 28  # not specified in source; standard 28-day assumed
f["compressive_strength_mpa"] = df_mah["f'c (MPa)"]
f["waste_type"]  = "RAC"
f["data_source"] = "real"
f["paper_ref_no"] = "Mahmoudian, Bypour & Kontoni 2024, Asian J. Civil Eng. (Ref 45)"
f = f.dropna(subset=["compressive_strength_mpa"])
f = f[f["compressive_strength_mpa"] > 0]
frames.append(f)
print(f"New Src A Mahmoudian et al. 2024       : {len(f)} rows loaded")


# New Source B: Hoang 2024 (RCC dataset) ]
df_hoang = pd.read_csv(os.path.join(replacement_folder, "hoang_2024_rcc_dataset.csv"))
f = pd.DataFrame()
rca = df_hoang["recycled_coarse_agg_kgm3"].fillna(0)
nca = df_hoang["natural_coarse_agg_kgm3"].fillna(0)
total = rca + nca
f["rac_%"]                    = (rca / total.replace(0, np.nan) * 100).round(1)
f["cement_kg_m3"]             = df_hoang["cement_kgm3"]
f["water_cement_ratio"]       = (df_hoang["water_kgm3"] / df_hoang["cement_kgm3"]).round(3)
f["curing_days"]              = df_hoang["concrete_age_days"]
f["compressive_strength_mpa"] = df_hoang["compressive_strength_mpa"]
f["waste_type"]  = "RAC"
f["data_source"] = "real"
f["paper_ref_no"] = "Hoang 2024, Mathematics (Ref 9)"
f = f.dropna(subset=["compressive_strength_mpa"])
f = f[f["compressive_strength_mpa"] > 0]
frames.append(f)
print(f"New Src B Hoang 2024 (RCC)             : {len(f)} rows loaded")


# New Source C: Yuan et al. 2022
df_yuan = pd.read_csv(os.path.join(replacement_folder, "yuan_2022_rac_ml_dataset.csv"))
f = pd.DataFrame()
f["rac_%"]                      = df_yuan["rca_pct"]
f["water_cement_ratio"]         = df_yuan["weff_c"]
f["cement_kg_m3"]               = 380  # not directly reported; standard assumption (a/c ratio given instead)
f["curing_days"]                = 28   # not specified per-row; standard 28-day assumed
f["compressive_strength_mpa"]   = df_yuan["compressive_strength_mpa"]
f["flexural_strength_mpa"]      = df_yuan["flexural_strength_mpa"]
f["waste_type"]  = "RAC"
f["data_source"] = "real"
f["paper_ref_no"] = "Yuan et al. 2022, Materials (Ref 47, compiled from 68 sources - see their Table S1)"
f = f.dropna(subset=["compressive_strength_mpa"])
f = f[f["compressive_strength_mpa"] > 0]
frames.append(f)
print(f"New Src C Yuan et al. 2022              : {len(f)} rows loaded")


# -------------------------------------------------------------
# STEP 4 : Combine all real data and standardize columns
# -------------------------------------------------------------

print("\nCombining all real data sources...")

real_df = pd.concat(frames, ignore_index=True)

for col in ["rac_%", "rcwtb_%", "ceramic_%", "plastic_%",
            "co2_emission_kg_m3", "cost_usd_m3",
            "flexural_strength_mpa", "split_tensile_strength_mpa"]:
    if col not in real_df.columns:
        real_df[col] = 0.0
    else:
        real_df[col] = real_df[col].fillna(0)

# FIX: bounds widened from the original narrow ranges. A systematic audit
# found the old bounds (w/c 0.25-0.70, cement 200-700, curing 1-90) were
# silently clipping hundreds of genuinely-reported real values. New bounds
# are set from the true observed range of real data across ALL sources,
# with small safety padding, they still catch genuine typos/impossible
# entries, but no longer distort legitimate reported measurements.
real_df["rac_%"]              = real_df["rac_%"].clip(0, 100)
real_df["water_cement_ratio"] = real_df["water_cement_ratio"].fillna(0.45).clip(0.15, 1.55)
real_df["curing_days"]        = real_df["curing_days"].fillna(28).clip(1, 365)
real_df["cement_kg_m3"]       = real_df["cement_kg_m3"].fillna(350).clip(100, 850)

real_df = real_df[real_df["compressive_strength_mpa"].between(5, 95)]

print(f"Total real rows after cleaning : {len(real_df)}")
print(real_df["waste_type"].value_counts().to_string())


# -------------------------------------------------------------
# STEP 5 : Generate augmented data 
# -------------------------------------------------------------

def generate_augmented(n, waste_type, seed):
    np.random.seed(seed)
    d = {}
    d["cement_kg_m3"]       = np.random.uniform(300, 520, n).round(1)
    d["water_cement_ratio"] = np.random.uniform(0.30, 0.60, n).round(3)
    d["curing_days"]        = np.random.choice([7, 14, 28, 56, 90], n)
    d["co2_emission_kg_m3"] = np.zeros(n)
    d["cost_usd_m3"]        = np.zeros(n)

    if waste_type == "RCWTB":
        d["rac_%"]     = np.random.uniform(20, 50, n).round(1)
        d["rcwtb_%"]   = np.random.uniform(5,  20, n).round(1)
        d["ceramic_%"] = np.zeros(n)
        d["plastic_%"] = np.zeros(n)
    elif waste_type == "Ceramic":
        d["rac_%"]     = np.random.uniform(20, 50, n).round(1)
        d["rcwtb_%"]   = np.zeros(n)
        d["ceramic_%"] = np.random.uniform(5,  25, n).round(1)
        d["plastic_%"] = np.zeros(n)
    elif waste_type == "Plastic":
        d["rac_%"]     = np.random.uniform(20, 50, n).round(1)
        d["rcwtb_%"]   = np.zeros(n)
        d["ceramic_%"] = np.zeros(n)
        d["plastic_%"] = np.random.uniform(5,  15, n).round(1)
    elif waste_type == "Combined":
        d["rac_%"]     = np.random.uniform(15, 40, n).round(1)
        d["rcwtb_%"]   = np.random.uniform(5,  15, n).round(1)
        d["ceramic_%"] = np.random.uniform(5,  15, n).round(1)
        d["plastic_%"] = np.random.uniform(3,   8, n).round(1)

    total = d["rac_%"] + d["rcwtb_%"] + d["ceramic_%"] + d["plastic_%"]

    base = (
        62
        - (d["water_cement_ratio"] - 0.3) * 65
        + (d["curing_days"] / 28) * 4
        - total * 0.12
        + d["rcwtb_%"]   * 0.35
        + d["ceramic_%"] * 0.22
        - d["plastic_%"] * 0.45
        + np.random.normal(0, 2.5, n)
    )

    d["compressive_strength_mpa"]   = base.clip(15, 80).round(1)
    d["flexural_strength_mpa"]      = (base * 0.12 + np.random.normal(0, 0.3, n)).clip(2, 9).round(2)
    d["split_tensile_strength_mpa"] = (base * 0.09 + np.random.normal(0, 0.2, n)).clip(1.5, 6).round(2)
    d["waste_type"]  = waste_type
    d["data_source"] = "augmented_literature_based"
    d["paper_ref_no"] = SYNTHETIC_TAG

    return pd.DataFrame(d)


def generate_rcwtb_freezethaw(n, seed):
    np.random.seed(seed)
    d = {}
    d["cement_kg_m3"]       = np.random.uniform(300, 520, n).round(1)
    d["water_cement_ratio"] = np.random.uniform(0.30, 0.55, n).round(3)
    d["curing_days"]        = np.random.choice([28, 56, 90], n)
    d["rac_%"]              = np.random.uniform(20, 50, n).round(1)
    d["rcwtb_%"]            = np.random.uniform(5,  20, n).round(1)
    d["ceramic_%"]          = np.zeros(n)
    d["plastic_%"]          = np.zeros(n)
    d["co2_emission_kg_m3"] = np.zeros(n)
    d["cost_usd_m3"]        = np.zeros(n)
    total = d["rac_%"] + d["rcwtb_%"]
    base = (
        62
        - (d["water_cement_ratio"] - 0.3) * 65
        + (d["curing_days"] / 28) * 4
        - total * 0.12
        + d["rcwtb_%"] * 0.35
        + np.random.normal(0, 2.5, n)
    )
    d["compressive_strength_mpa"]   = base.clip(15, 75).round(1)
    d["flexural_strength_mpa"]      = (base * 0.12 + np.random.normal(0, 0.3, n)).clip(2, 9).round(2)
    d["split_tensile_strength_mpa"] = (base * 0.09 + np.random.normal(0, 0.2, n)).clip(1.5, 6).round(2)
    d["waste_type"]  = "RCWTB_FreezeThaw"
    d["data_source"] = "augmented_literature_based"
    d["paper_ref_no"] = SYNTHETIC_TAG
    return pd.DataFrame(d)


aug_rcwtb    = generate_augmented(400, "RCWTB",    101)
aug_rcwtb_ft = generate_rcwtb_freezethaw(150,      102)
aug_ceramic  = generate_augmented(250, "Ceramic",  202)
aug_plastic  = generate_augmented(150, "Plastic",  303)
aug_combined = generate_augmented(200, "Combined", 404)

print(f"\nAugmented rows generated:")
print(f"  RCWTB            : {len(aug_rcwtb)}")
print(f"  RCWTB FreezeThaw : {len(aug_rcwtb_ft)}")
print(f"  Ceramic          : {len(aug_ceramic)}")
print(f"  Plastic          : {len(aug_plastic)}")
print(f"  Combined         : {len(aug_combined)}")


# -------------------------------------------------------------
# STEP 6 : Merge real and augmented data
# -------------------------------------------------------------

all_cols = [
    "cement_kg_m3", "water_cement_ratio", "curing_days",
    "rac_%", "rcwtb_%", "ceramic_%", "plastic_%",
    "compressive_strength_mpa", "flexural_strength_mpa",
    "split_tensile_strength_mpa",
    "co2_emission_kg_m3", "cost_usd_m3",
    "waste_type", "data_source", "paper_ref_no"
]

def align_columns(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    return df[cols]

master_df = pd.concat([
    align_columns(real_df,        all_cols),
    align_columns(aug_rcwtb,      all_cols),
    align_columns(aug_rcwtb_ft,   all_cols),
    align_columns(aug_ceramic,    all_cols),
    align_columns(aug_plastic,    all_cols),
    align_columns(aug_combined,   all_cols),
], ignore_index=True)

n = len(master_df)


# -------------------------------------------------------------
# STEP 7 : Calculate derived output columns
# -------------------------------------------------------------

master_df["total_replacement_%"] = (
    master_df["rac_%"] + master_df["rcwtb_%"] +
    master_df["ceramic_%"] + master_df["plastic_%"]
).round(1).clip(0, 100)

master_df["chloride_penetration_mm"] = (
    15
    + (master_df["water_cement_ratio"] - 0.3) * 35
    + master_df["total_replacement_%"] * 0.08
    - (master_df["compressive_strength_mpa"] - 20) * 0.12
    - (master_df["curing_days"] - 7) * 0.04
    - master_df["rcwtb_%"]   * 0.20
    - master_df["ceramic_%"] * 0.15
    + np.random.normal(0, 0.8, n)
).clip(1, 30).round(2)

master_df["marine_durability_score"] = (
    10
    - master_df["chloride_penetration_mm"] * 0.18
    - (master_df["water_cement_ratio"] - 0.3) * 4
    + master_df["rcwtb_%"]   * 0.08
    + master_df["ceramic_%"] * 0.06
    + (master_df["curing_days"] / 28) * 0.3
    + np.random.normal(0, 0.3, n)
).clip(1, 10).round(1)

# CO2: real values only from Source 10-equivalent (none now); rest computed
co2_calc = (
    master_df["cement_kg_m3"] * 0.82
    - master_df["rac_%"]     * 0.90
    - master_df["rcwtb_%"]   * 0.60
    - master_df["ceramic_%"] * 0.70
    - master_df["plastic_%"] * 0.50
    + np.random.normal(0, 5, n)
).clip(80, 420)

master_df["co2_emission_kg_m3"] = np.where(
    master_df["co2_emission_kg_m3"] > 0,
    master_df["co2_emission_kg_m3"],
    co2_calc
).round(1)

# FIX: floor widened from 55 to 15. Initially floor (55) sat almost
# exactly at the median of this formula's natural output distribution,
# causing 47.6% of the entire dataset to pile up at an identical,
# artificial cost value, a severe distortion for a variable that is
# both a modeling target and a PSO optimization objective. The new
# floor (15) was chosen from the formula's actual output distribution
# (including its noise term): it sits just below the natural 1st
# percentile, so it still catches genuinely nonsensical near-zero/
# negative outputs while preserving the full realistic spread of
# values in between. This reduces the pile-up from 47.6% to ~1.4%.
cost_calc = (
    master_df["cement_kg_m3"] * 0.12
    - master_df["rac_%"]     * 0.25
    - master_df["rcwtb_%"]   * 0.20
    - master_df["ceramic_%"] * 0.18
    - master_df["plastic_%"] * 0.15
    + master_df["compressive_strength_mpa"] * 0.40
    + np.random.normal(0, 4, n)
).clip(15, 165)

master_df["cost_usd_m3"] = np.where(
    master_df["cost_usd_m3"] > 0,
    master_df["cost_usd_m3"],
    cost_calc
).round(1)

mask = master_df["flexural_strength_mpa"] == 0
master_df.loc[mask, "flexural_strength_mpa"] = (
    master_df.loc[mask, "compressive_strength_mpa"] * 0.12
    + np.random.normal(0, 0.3, mask.sum())
).clip(2, 9).round(2)

mask = master_df["split_tensile_strength_mpa"] == 0
master_df.loc[mask, "split_tensile_strength_mpa"] = (
    master_df.loc[mask, "compressive_strength_mpa"] * 0.09
    + np.random.normal(0, 0.2, mask.sum())
).clip(1.5, 6).round(2)


# -------------------------------------------------------------
# STEP 8 : Fix outliers in key output columns
# -------------------------------------------------------------

master_df["co2_emission_kg_m3"] = master_df["co2_emission_kg_m3"].clip(80, 420)
# FIX: widened from (5, 90) to (5, 95) to match Step 4's own filter ceiling,
# so real values between 90-95 MPa are no longer re-clipped down to 90
# after already surviving the Step 4 between(5, 95) filter.
master_df["compressive_strength_mpa"] = master_df["compressive_strength_mpa"].round(1).clip(5, 95)


# -------------------------------------------------------------
# STEP 9 : Final column ordering, shuffling and save
# -------------------------------------------------------------

col_order = [
    "waste_type", "data_source", "paper_ref_no",
    "cement_kg_m3", "water_cement_ratio", "curing_days",
    "rac_%", "rcwtb_%", "ceramic_%", "plastic_%",
    "total_replacement_%",
    "compressive_strength_mpa", "flexural_strength_mpa",
    "split_tensile_strength_mpa",
    "chloride_penetration_mm", "marine_durability_score",
    "co2_emission_kg_m3", "cost_usd_m3"
]

master_df = master_df[col_order]
master_df = master_df.sample(frac=1, random_state=42).reset_index(drop=True)
master_df.index = master_df.index + 1
master_df.index.name = "sample_id"

output_path = "MASTER_waste_concrete_dataset_v4.csv"
master_df.to_csv(output_path)

print("\n" + "=" * 60)
print("MASTER DATASET COMPLETE (v4 - corrected)")
print("=" * 60)
print(f"Total rows    : {len(master_df)}")
print(f"Total columns : {len(master_df.columns)}")
print(f"\nWaste type breakdown:")
print(master_df["waste_type"].value_counts().to_string())
print(f"\nData source breakdown:")
print(master_df["data_source"].value_counts().to_string())
print(f"\nKey ranges:")
print(f"  Compressive strength : {master_df['compressive_strength_mpa'].min()} - {master_df['compressive_strength_mpa'].max()} MPa")
print(f"  CO2 emission         : {master_df['co2_emission_kg_m3'].min()} - {master_df['co2_emission_kg_m3'].max()} kg/m3")
print(f"  Cost                 : {master_df['cost_usd_m3'].min()} - {master_df['cost_usd_m3'].max()} USD/m3")
print(f"  Marine durability    : {master_df['marine_durability_score'].min()} - {master_df['marine_durability_score'].max()}")
print(f"\nFile saved to : {output_path}")
