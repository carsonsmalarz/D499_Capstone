'''
Main Break Risk Predictor
Created by Quentin Cox and Carson Smalarz
'''

# Required libraries

#%pip install pandas, numpy, sklearn, xgboost
import os
import sys
import shutil
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

# Paths
if getattr(sys, 'frozen', False):
    SCRIPT_DIR = os.path.dirname(sys.executable)
else:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPES_CSV       = os.path.join(SCRIPT_DIR, "Water_Pipes_Feb10.csv")
BREAKS_CSV      = os.path.join(SCRIPT_DIR, "Breaks2013toNow.csv")
BACKUP_FOLDER   = os.path.join(SCRIPT_DIR, "backups")

# Optional Break Data Update

#get a human-readable string of most recent break date
def get_last_break_date(breaks: pd.DataFrame) -> str:
    valid = breaks["break_date"].dropna()
    if valid.empty:
        return "No breaks recorded yet"
    return valid.max().strftime("%B %d, %Y")

# Load a user-supplied CSV.  Must have columns: pipe_tag, break_date: Returns a cleaned DataFrame or None on failure.
def load_new_breaks(filepath: str) -> pd.DataFrame | None:
    if not os.path.isfile(filepath):
        print(f"  [ERROR] File not found: {filepath}")
        return None
 
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"  [ERROR] Could not read file: {e}")
        return None
 
    df.columns = df.columns.str.strip().str.lower()
 
    required = {"pipe_tag", "break_date"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        print(f"  [ERROR] Missing required columns: {missing}")
        print(f"          Found columns: {list(df.columns)}")
        return None
 
    df["break_date"] = pd.to_datetime(df["break_date"], errors="coerce")
    bad_dates = df["break_date"].isna().sum()
    if bad_dates:
        print(f"  [WARNING] {bad_dates} row(s) had unparseable dates and will be skipped.")
        df = df.dropna(subset=["break_date"])
 
    if df.empty:
        print("  [ERROR] No valid rows remain after date parsing.")
        return None
 
    return df[["pipe_tag", "break_date"]]

# Remove rows that are either duplicates of existing breaks or references a pipe tag that does note exist
def validate_new_breaks(
    new_df: pd.DataFrame,
    existing_breaks: pd.DataFrame,
    pipes: pd.DataFrame,
) -> pd.DataFrame:
    valid_tags = set(pipes["tag"].unique())
 
    # Check for unknown pipe tags
    unknown_mask = ~new_df["pipe_tag"].isin(valid_tags)
    if unknown_mask.any():
        bad = new_df.loc[unknown_mask, "pipe_tag"].unique()
        print(f"\n  [WARNING] {unknown_mask.sum()} row(s) reference pipe tags not found "
              f"in Water_Pipes_Feb10.csv and will be skipped:")
        for t in bad:
            print(f"            - {t}")
        new_df = new_df[~unknown_mask].copy()
 
    if new_df.empty:
        return new_df
 
    # Check for duplicates against existing records 
    existing_pairs = set(
        zip(
            existing_breaks["pipe_tag"].astype(str),
            existing_breaks["break_date"].dt.normalize()
        )
    )
    new_df["_norm_date"] = new_df["break_date"].dt.normalize()
    dup_mask = new_df.apply(
        lambda r: (str(r["pipe_tag"]), r["_norm_date"]) in existing_pairs,
        axis=1,
    )
    if dup_mask.any():
        print(f"\n  [WARNING] {dup_mask.sum()} row(s) already exist in the break "
              f"database and will be skipped:")
        print(new_df[dup_mask][["pipe_tag", "break_date"]].to_string(index=False))
        new_df = new_df[~dup_mask].copy()
 
    new_df = new_df.drop(columns=["_norm_date"])
    return new_df

# copy the current breaks CSV into the backups folder
def backup_breaks_csv() -> None:
    os.makedirs(BACKUP_FOLDER, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_name = f"breaks_backup_{timestamp}.csv"
    backup_path = os.path.join(BACKUP_FOLDER, backup_name)
    shutil.copy2(BREAKS_CSV, backup_path)
    print(f"\n  [INFO] Backup saved → backups/{backup_name}")

# append validated new breaks to the CSV and return the updated dataframe
def append_breaks(new_df: pd.DataFrame, existing_breaks: pd.DataFrame) -> pd.DataFrame:
    backup_breaks_csv()
 
    updated = pd.concat([existing_breaks, new_df], ignore_index=True)
    updated["break_date"] = pd.to_datetime(updated["break_date"])
    updated = updated.sort_values("break_date")
 
    updated.to_csv(BREAKS_CSV, index=False)
    print(f"  [INFO] {len(new_df)} new break(s) appended to Breaks2013toNow.csv")
    return updated

# ask the user if they want to add new breaks or not
def maybe_add_breaks(breaks: pd.DataFrame, pipes: pd.DataFrame) -> pd.DataFrame:
    last = get_last_break_date(breaks)
    print("\n" + "=" * 60)
    print("  STEP 1 - Break Data Update")
    print("=" * 60)
    print(f"  Last recorded break date: {last}")
    print()
 
    while True:
        answer = input("  Do you have new break data to add? (yes/no): ").strip().lower()
        if answer in ("yes", "y"):
            break
        elif answer in ("no", "n"):
            print("  Skipping break data update.")
            return breaks
        else:
            print("  Please enter 'yes' or 'no'.")
 
    while True:
        print("\n  NOTE: Your CSV file should have two columns: pipe_tag and break_date.")
        filepath = input("  Enter the full file path to your new break CSV: ").strip().strip('"')
        new_df = load_new_breaks(filepath)
        if new_df is not None:
            break
        retry = input("  Would you like to try a different file? (yes/no): ").strip().lower()
        if retry not in ("yes", "y"):
            print("  Skipping break data update.")
            return breaks
 
    validated = validate_new_breaks(new_df, breaks, pipes)
 
    if validated.empty:
        print("\n  [INFO] No new unique breaks to add after validation. Nothing was changed.")
        return breaks
 
    print(f"\n  {len(validated)} new break(s) passed validation:")
    print(validated[["pipe_tag", "break_date"]].to_string(index=False))
    confirm = input("\n  Confirm append to database? (yes/no): ").strip().lower()
    if confirm in ("yes", "y"):
        breaks = append_breaks(validated, breaks)
    else:
        print("  Append cancelled. No changes made.")
 
    return breaks

# Get prediction window from user

def get_prediction_months() -> int:
    print("\n" + "=" * 60)
    print("  STEP 2 - Prediction Window")
    print("=" * 60)
    print("  How many months ahead should the model predict pipe breaks?")
    print("  NOTE: The model performs best for windows between 12 and 60 months.")
    print("        Values outside this range may produce less reliable results.\n")
 
    while True:
        raw = input("  Enter number of months (12-60 recommended): ").strip()
        try:
            months = int(raw)
        except ValueError:
            print("  Please enter a whole number.")
            continue
 
        if months < 1:
            print("  Please enter a positive number of months.")
            continue
 
        if not (12 <= months <= 60):
            print(f"  [WARNING] {months} months is outside the recommended 12-60 month range.")
            confirm = input("  Continue anyway? (yes/no): ").strip().lower()
            if confirm in ("yes", "y"):
                return months
        else:
            return months
        
# data prep

def clean_polywrap(x):
    if x == "n":
        return 0.0
    try:
        return float(x)
    except Exception:
        return 0.0
    
def load_and_clean_pipes() -> pd.DataFrame:
    pipes = pd.read_csv(PIPES_CSV, encoding="cp1252", dtype={10: str})
    pipes["Soil_Hydro_Group"] = pipes["Soil_Hydro_Group"].fillna("No Group")
    pipes = pipes.dropna()
    pipes["polywrap"] = pipes["polywrap"].apply(clean_polywrap)
    pipes["yr_inst"] = pd.to_numeric(pipes["yr_inst"], errors="coerce")
    return pipes

# snapshots

def build_snapshot_df(pipes: pd.DataFrame, breaks: pd.DataFrame, pred_months: int) -> pd.DataFrame:
    SNAPSHOT_START = "2013-01-01"
    SNAPSHOT_END   = breaks['break_date'].max() - pd.DateOffset(months=pred_months)
    snapshots = pd.date_range(SNAPSHOT_START, SNAPSHOT_END, freq="MS")
 
    rows = []
    for _, pipe in pipes.iterrows():
        tag = pipe["tag"]
        install_year = pipe["yr_inst"]
 
        if pd.isna(install_year):
            continue
 
        pipe_breaks = breaks.loc[
            breaks["pipe_tag"] == tag, "break_date"
        ].sort_values()
 
        for snap in snapshots:
            if snap.year < install_year:
                continue
 
            past = pipe_breaks[pipe_breaks < snap]
 
            if len(past) == 0:
                months_since_last = np.nan
                breaks_so_far     = 0
            else:
                last_break        = past.max()
                months_since_last = (snap - last_break).days / 30.44
                breaks_so_far     = len(past)
 
            rows.append({
                "tag":             tag,
                "snapshot":        snap,
                "diameter":        pipe["diameter"],
                "material":        pipe["material"],
                "polywrap":        pipe["polywrap"],
                "yr_inst":         pipe["yr_inst"],
                "length":          pipe["length"],
                "NumberOfMa":      pipe["NumberOfMa"],
                "Shape_Length":    pipe["Shape_Length"],
                "Soil_Comp":       pipe["Soil_Comp"],
                "Soil_Hydro_Group": pipe["Soil_Hydro_Group"],
                "ElevChange":      pipe["ElevChange"],
                "NumberofSLs":     pipe["NumberofSLs"],
                "age_years":       snap.year - pipe["yr_inst"],
                "Score":           pipe["Score"],
                "breaks_so_far":   breaks_so_far,
                "months_since_last": months_since_last,
            })
 
    return pd.DataFrame(rows)

# model training

FEATURE_COLS = [
    "material",
    "yr_inst",
    "length",
    "Score",
    "NumberOfMa",
    "Soil_Comp",
    "Soil_Hydro_Group",
    "ElevChange",
    "NumberofSLs",
    "breaks_so_far",
]

CAT_FEATURES = ["material", "Soil_Comp", "Soil_Hydro_Group"]
NUM_FEATURES = [c for c in FEATURE_COLS if c not in CAT_FEATURES]

def label_breaks(snap_df: pd.DataFrame, breaks: pd.DataFrame, pred_months: int) -> pd.DataFrame:
    breaks_by_tag = (
        breaks
        .dropna(subset=["pipe_tag", "break_date"])
        .sort_values("break_date")
        .groupby("pipe_tag")["break_date"]
        .apply(np.array)
    )
 
    def broke_in_next(tag, snap):
        if tag not in breaks_by_tag:
            return 0
        dates = breaks_by_tag[tag]
        start = np.datetime64(snap)
        end   = np.datetime64(snap + pd.DateOffset(months=pred_months))
        i     = np.searchsorted(dates, start, side="right")
        if i >= len(dates):
            return 0
        return int(dates[i] <= end)
 
    snap_df["Y"] = snap_df.apply(lambda r: broke_in_next(r["tag"], r["snapshot"]), axis=1)
    return snap_df

# fit imputer, encoder, and XGBoost, and return fitted objects
def train_model(snap_df: pd.DataFrame):
    Xmat = snap_df[NUM_FEATURES + CAT_FEATURES]
    y    = snap_df["Y"]
 
    X_train = Xmat
    y_train = y
 
    num_imp = SimpleImputer(strategy="median")
    X_train_num = num_imp.fit_transform(X_train[NUM_FEATURES])
 
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_cat = ohe.fit_transform(X_train[CAT_FEATURES])
 
    X_train_final = np.hstack([X_train_num, X_train_cat])
 
    xgb = XGBClassifier(
        objective        = "binary:logistic",
        eval_metric      = "aucpr",
        n_estimators     = 1500,
        tree_method      = "hist",
        max_depth        = 20,
        reg_lambda       = 30,
        learning_rate    = 1,
        alpha            = 0.001,
        eta              = 1,
        scale_pos_weight = 5,
    )
    xgb.fit(X_train_final, y_train)
 
    return xgb, num_imp, ohe
 
 # generate the risk report

def build_today_df(pipes: pd.DataFrame, breaks: pd.DataFrame) -> pd.DataFrame:
    today = pd.Timestamp.today().normalize()
    rows  = []
 
    for _, pipe in pipes.iterrows():
        tag = pipe["tag"]
        install_year = pipe["yr_inst"]
 
        if pd.isna(install_year) or today.year < install_year:
            continue
 
        pipe_breaks   = breaks.loc[breaks["pipe_tag"] == tag, "break_date"]
        past          = pipe_breaks[pipe_breaks < today]
        breaks_so_far = len(past)
 
        rows.append({
            "tag":             tag,
            "material":        pipe["material"],
            "yr_inst":         pipe["yr_inst"],
            "length":          pipe["length"],
            "Score":           pipe["Score"],
            "NumberOfMa":      pipe["NumberOfMa"],
            "Soil_Comp":       pipe["Soil_Comp"],
            "Soil_Hydro_Group": pipe["Soil_Hydro_Group"],
            "ElevChange":      pipe["ElevChange"],
            "NumberofSLs":     pipe["NumberofSLs"],
            "breaks_so_far":   breaks_so_far,
        })
 
    return pd.DataFrame(rows)

def generate_report(
    pipes:     pd.DataFrame,
    breaks:    pd.DataFrame,
    xgb,
    num_imp,
    ohe,
    pred_months: int,
) -> str:
    #Score every pipe, write CSV, return output path
    today_df = build_today_df(pipes, breaks)
 
    X_today_num   = num_imp.transform(today_df[NUM_FEATURES])
    X_today_cat   = ohe.transform(today_df[CAT_FEATURES])
    X_today_final = np.hstack([X_today_num, X_today_cat])
 
    today_df["prob_xgb"] = xgb.predict_proba(X_today_final)[:, 1]
 
    output = today_df[["tag", "prob_xgb"]].sort_values("prob_xgb", ascending=False)
 
    date_str     = datetime.now().strftime("%Y-%m-%d")
    output_name  = f"pipe_break_risk_report_{date_str}_{pred_months}mo.csv"
    output_path  = os.path.join(SCRIPT_DIR, output_name)
    output.to_csv(output_path, index=False)
 
    return output_path

# ---------------
#      MAIN
# ---------------

def main():
    print("\n" + "=" * 60)
    print("  Pipe Break Risk Predictor")
    print("=" * 60)
 
    # Load base data 
    print("\n  Loading pipe and break data …")
    try:
        pipes = load_and_clean_pipes()
    except FileNotFoundError:
        print(f"  [ERROR] Could not find pipes CSV at:\n    {PIPES_CSV}")
        sys.exit(1)
 
    try:
        breaks = pd.read_csv(BREAKS_CSV)
        breaks["break_date"] = pd.to_datetime(breaks["break_date"], errors="coerce")
    except FileNotFoundError:
        print(f"  [ERROR] Could not find breaks CSV at:\n    {BREAKS_CSV}")
        sys.exit(1)
 
    print(f"  Loaded {len(pipes):,} pipes and {len(breaks):,} break records.")
 
    # Optionally add new breaks
    breaks = maybe_add_breaks(breaks, pipes)
 
    # Get prediction window
    pred_months = get_prediction_months()
 
    # Train model
    print("\n" + "=" * 60)
    print("  STEP 3 - Training Model")
    print("=" * 60)
    print(f"  Building snapshot panel … (this may take a few minutes)…")
    snap_df = build_snapshot_df(pipes, breaks, pred_months)
    snap_df = label_breaks(snap_df, breaks, pred_months)
 
    print(f"  Training XGBoost model for a {pred_months}-month prediction window … (this may take a few minutes)…")
    xgb, num_imp, ohe = train_model(snap_df)
    print("  Model training complete.")
 
    # Generate report 
    print("\n" + "=" * 60)
    print("  STEP 4 - Generating Risk Report")
    print("=" * 60)
    output_path = generate_report(pipes, breaks, xgb, num_imp, ohe, pred_months)
    print(f"  Report saved → {os.path.basename(output_path)}")
 
    print("\n" + "=" * 60)
    print("  Done! Have a great day.")
    print("=" * 60 + "\n")
 
 
if __name__ == "__main__":
    main()