from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import shuffle

from DatasetLoader.loader import split_70_15_15, dataframe_to_tensors, build_global_eval_sets

RANDOM_STATE = 42

TARGET_COL = "ViolentCrimesPerPop"
SENSITIVE_FEATURE = "racepctblack"

BASE_DIR = Path(__file__).resolve().parent.parent
CAC_PATH = BASE_DIR / "Datasets" / "communities.csv"

# Clients are formed by a random but fixed partition of states.
STATE_COL = "state"

CAC_STATE_MAP_3 = {
    "client_1": [38, 44, 56, 32, 35, 28, 54, 41, 6, 33, 51, 10, 37, 12, 27, 8],
    "client_2": [24, 47, 22, 48, 36, 40, 34, 50, 13, 5, 49, 53, 23, 39, 9],
    "client_3": [21, 16, 29, 25, 45, 1, 18, 20, 55, 4, 46, 42, 2, 19, 11],
}

CAC_STATE_MAP_5 = {
    "client_1": [38, 44, 56, 32, 35, 28, 54, 41, 6, 33],
    "client_2": [51, 10, 37, 12, 27, 8, 24, 47, 22],
    "client_3": [48, 36, 40, 34, 50, 13, 5, 49, 53],
    "client_4": [23, 39, 9, 21, 16, 29, 25, 45, 1],
    "client_5": [18, 20, 55, 4, 46, 42, 2, 19, 11],
}

DROP_COLS = [  # these columns are dropped since they have to many missing values, no predictive value or are meta information, adapted from Komiyama et al. (2018)
    "community",
    "communityname",
    "county",

    "OtherPerCap",

    "LemasSwornFT",
    "LemasSwFTPerPop",
    "LemasSwFTFieldOps",
    "LemasSwFTFieldPerPop",
    "LemasTotalReq",
    "LemasTotReqPerPop",
    "PolicReqPerOffic",
    "PolicPerPop",
    "RacialMatchCommPol",
    "PctPolicWhite",
    "PctPolicBlack",
    "PctPolicHisp",
    "PctPolicAsian",
    "PctPolicMinor",
    "OfficAssgnDrugUnits",
    "NumKindsDrugsSeiz",
    "PolicAveOTWorked",
    "PolicCars",
    "PolicOperBudg",
    "LemasPctPolicOnPatr",
    "LemasGangUnitDeploy",
    "LemasPctOfficDrugUn",
    "PolicBudgPerPop",
]


def _binarize_by_median(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Binarizes a column into {0,1} by global median:
    1 = value > median, 0 = value <= median
    """
    median = df[col].median()
    return (df[col] > median).astype(int)


def load_cac(*, for_iid: bool) -> pd.DataFrame:
    """
    Loads communities CSVs and applies preprocessing.
    """
    df = pd.read_csv(CAC_PATH)

    # Drop columns like the paper
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number]).copy()

    # Drop rows only if essential cols missing
    df = df.dropna(subset=[STATE_COL, TARGET_COL, SENSITIVE_FEATURE]).copy()

    df[STATE_COL] = df[STATE_COL].astype(int)

    # Median-binarize target + sensitive
    df[TARGET_COL] = (df[TARGET_COL] > df[TARGET_COL].median()).astype(int)
    df[SENSITIVE_FEATURE] = (df[SENSITIVE_FEATURE] > df[SENSITIVE_FEATURE].median()).astype(int)

    feature_cols = [c for c in df.columns if c not in (STATE_COL, TARGET_COL, SENSITIVE_FEATURE)]

    df = df.dropna().copy()

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    df = shuffle(df, random_state=RANDOM_STATE)
    return df


def load_cac_states_3():
    """
    NON-IID split: assigns groups of states to 3 clients (random grouping with seed=42).
    Then each client's data is split into 70/15/15 (train/val/test).
    """
    data = load_cac(for_iid=False)

    y_encoder = LabelEncoder()
    y_encoder.fit(np.array([0, 1]))

    df1 = data[data[STATE_COL].isin(CAC_STATE_MAP_3["client_1"])].copy()
    df2 = data[data[STATE_COL].isin(CAC_STATE_MAP_3["client_2"])].copy()
    df3 = data[data[STATE_COL].isin(CAC_STATE_MAP_3["client_3"])].copy()

    dfs = [df1, df2, df3]
    splits = [split_70_15_15(d, seed=RANDOM_STATE) for d in dfs]

    data_dict = {}
    val_parts, test_parts = [], []
    for i, (tr, va, te) in enumerate(splits, start=1):
        X, y, s, ypot = dataframe_to_tensors(
            tr, target_col=TARGET_COL, sensitive_feature=SENSITIVE_FEATURE, y_encoder=y_encoder
        )
        data_dict[f"client_{i}"] = {"X": X, "y": y, "s": s, "y_pot": ypot}
        val_parts.append(va)
        test_parts.append(te)

    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    return (
        data_dict,
        *build_global_eval_sets(
            val_df,
            test_df,
            target_col=TARGET_COL,
            sensitive_feature=SENSITIVE_FEATURE,
            y_encoder=y_encoder,
        )
    )


def load_cac_states_5():
    """
    NON-IID split: assigns groups of states to 5 clients (random grouping with seed=42).
    Then each client's data is split into 70/15/15 (train/val/test).
    """
    data = load_cac(for_iid=False)

    y_encoder = LabelEncoder()
    y_encoder.fit(np.array([0, 1]))

    df1 = data[data[STATE_COL].isin(CAC_STATE_MAP_5["client_1"])].copy()
    df2 = data[data[STATE_COL].isin(CAC_STATE_MAP_5["client_2"])].copy()
    df3 = data[data[STATE_COL].isin(CAC_STATE_MAP_5["client_3"])].copy()
    df4 = data[data[STATE_COL].isin(CAC_STATE_MAP_5["client_4"])].copy()
    df5 = data[data[STATE_COL].isin(CAC_STATE_MAP_5["client_5"])].copy()

    dfs = [df1, df2, df3, df4, df5]
    splits = [split_70_15_15(d, seed=RANDOM_STATE) for d in dfs]

    data_dict = {}
    val_parts, test_parts = [], []
    for i, (tr, va, te) in enumerate(splits, start=1):
        X, y, s, ypot = dataframe_to_tensors(
            tr, target_col=TARGET_COL, sensitive_feature=SENSITIVE_FEATURE, y_encoder=y_encoder
        )
        data_dict[f"client_{i}"] = {"X": X, "y": y, "s": s, "y_pot": ypot}
        val_parts.append(va)
        test_parts.append(te)

    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    return (
        data_dict,
        *build_global_eval_sets(
            val_df,
            test_df,
            target_col=TARGET_COL,
            sensitive_feature=SENSITIVE_FEATURE,
            y_encoder=y_encoder,
        )
    )


def load_cac_random(num_clients: int = 10):
    """
    IID split: assigns states to n clients (random grouping with seed=42).
    Then each client's data is split into 70/15/15 (train/val/test).
    """
    data = load_cac(for_iid=True)

    y_encoder = LabelEncoder()
    y_encoder.fit(np.array([0, 1]))

    client_dfs = np.array_split(data, num_clients)

    data_dict = {}
    val_parts, test_parts = [], []

    for i, df_chunk in enumerate(client_dfs, start=1):
        tr, va, te = split_70_15_15(df_chunk, seed=RANDOM_STATE)
        X, y, s, ypot = dataframe_to_tensors(
            tr, target_col=TARGET_COL, sensitive_feature=SENSITIVE_FEATURE, y_encoder=y_encoder
        )
        data_dict[f"client_{i}"] = {"X": X, "y": y, "s": s, "y_pot": ypot}
        val_parts.append(va)
        test_parts.append(te)

    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    return (
        data_dict,
        *build_global_eval_sets(
            val_df,
            test_df,
            target_col=TARGET_COL,
            sensitive_feature=SENSITIVE_FEATURE,
            y_encoder=y_encoder,
        )
    )
