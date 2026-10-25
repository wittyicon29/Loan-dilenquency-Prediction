"""Feature engineering for the loan delinquency case study.

Developed and checked step by step in fe.ipynb; decision numbers (D…) refer to
docs/findings_and_decisions.md. Every step uses only each row's own values (nothing is
learned from other rows or from the target), so the same function can be applied to
dev, holdout and test without leakage.
"""
import numpy as np

# ---------------------------------------------------------------------------
# Step 1 — the -1 "not found" sentinel (D9, D12)
# ---------------------------------------------------------------------------
SENTINEL_COLS = [
    "Email Attributes.digitalAge",
    "Email Name Attributes.digitalage",
    "Phone Name Attributes.phoneNameDigitalAge",
    "Email Name Attributes.nameMatchScore",
]
NOT_FOUND_FLAGS = {                                   # flag name -> column it is built from
    "email_name_not_found": "Email Name Attributes.nameMatchScore",
    "email_not_found":      "Email Name Attributes.digitalage",
    "phone_name_not_found": "Phone Name Attributes.phoneNameDigitalAge",
}


def handle_sentinels(data, mode="flag"):
    """mode='raw': leave -1 as-is (option a). mode='flag': add not-found flags, then -1 -> NaN (option c)."""
    out = data.copy()
    if mode == "raw":
        return out
    if mode != "flag":
        raise ValueError(f"unknown mode: {mode}")

    # 1. flags first, while the -1 values still exist
    for flag, source in NOT_FOUND_FLAGS.items():
        out[flag] = out[source].eq(-1).astype("int8")

    # 2. then replace -1 with NaN (the real values stay untouched)
    out[SENTINEL_COLS] = out[SENTINEL_COLS].replace(-1, np.nan)
    return out


# ---------------------------------------------------------------------------
# Step 2 — platform presence shares (D25)
# ---------------------------------------------------------------------------
def add_platform_share(data):
    """Share of known platforms where the person is registered, separately for email and phone."""
    out = data.copy()
    for source in ["Email", "Phone"]:
        cols = [c for c in data.columns
                if c.startswith(f"{source} Social Advance.")
                and not c.endswith((".paytm", ".skype"))]
        n_known   = data[cols].notna().sum(axis=1)   # platforms with a 0 or 1
        n_present = data[cols].sum(axis=1)           # number of 1s (NaN is skipped)
        out[f"{source.lower()}_platform_known"] = n_known
        out[f"{source.lower()}_platform_share"] = n_present / n_known.replace(0, np.nan)
    return out


# ---------------------------------------------------------------------------
# Step 3 — ordinal encoding of ordered categories (D17)
# ---------------------------------------------------------------------------
ORDINAL_MAPS = {
    "Email Attributes.footprint": {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "NO_RECORDS": np.nan},
    "Phone Name Attributes.phoneFootprintStrengthOverall": {"Low": 0, "Medium": 1, "High": 2, "Very High": 3},
    "Email Name Attributes.emailFootprintStrength": {"Very Low": 0, "Low": 1, "High": 2, "Very High": 3},
}


def encode_ordinals(data, ordinal_maps=ORDINAL_MAPS):
    """Convert ordinal string values to integers, using the provided mapping."""
    out = data.copy()
    out["footprint_no_records"] = out["Email Attributes.footprint"].eq("NO_RECORDS").astype("int8")  # flag first
    for col, mapping in ordinal_maps.items():
        out[col] = out[col].map(mapping)
    return out


# ---------------------------------------------------------------------------
# Step 4 — nominal categories: fixed levels chosen from dev (D18, D32, D52)
# ---------------------------------------------------------------------------
CATEGORY_LEVELS = {
    # column: (levels to keep, keep missing as its own level?)
    "Phone Network.numberBillingType":           (["prepaid", "postpaid", "unknown"], True),
    "Phone Network.currentNetworkName":          (["Airtel", "Reliance Jio"],         False),
    "Email Attributes.emailFinalRecommendation": (["Good", "Check"],                  False),
}


def clean_categories(data, levels=CATEGORY_LEVELS):
    """Keep listed levels; everything else becomes 'other' (missing becomes 'missing' if requested)."""
    out = data.copy()
    for col, (keep, missing_separate) in levels.items():
        s = out[col]
        cleaned = s.where(s.isin(keep), "other")              # not in the list -> 'other'
        if missing_separate:
            cleaned = cleaned.where(s.notna(), "missing")     # NaN -> 'missing'
        out[col] = cleaned
    return out


# ---------------------------------------------------------------------------
# Step 5 — assemble the model feature set
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "Email Attributes.digitalAge", "Email Name Attributes.digitalage",
    "Phone Name Attributes.phoneNameDigitalAge", "Email Name Attributes.nameMatchScore",
    "Email Name Attributes.nameEmailMatch", "Email Attributes.uniqueHits",
    "email_platform_share", "phone_platform_share",
    "Email Attributes.footprint", "Phone Name Attributes.phoneFootprintStrengthOverall",
    "Email Name Attributes.emailFootprintStrength",                     # ordinal codes
]
FLAG_FEATURES = [
    "Phone Social Advance.whatsapp", "Phone Network.numberHasPortingHistory",
    "Email Name Attributes.firstNameMatch", "Email Name Attributes.lastNameMatch",
    "email_name_not_found", "email_not_found", "phone_name_not_found", "footprint_no_records",
]
CATEGORICAL_FEATURES = [
    "Phone Network.numberBillingType", "Phone Network.currentNetworkName",
    "Email Attributes.emailFinalRecommendation",
]
MODEL_FEATURES = NUMERIC_FEATURES + FLAG_FEATURES + CATEGORICAL_FEATURES


def feature_groups(sentinel_mode="flag"):
    """(numeric, flags, categorical) column lists for a sentinel mode — used to build the model pipeline."""
    flags = FLAG_FEATURES if sentinel_mode == "flag" else \
            [f for f in FLAG_FEATURES if f not in NOT_FOUND_FLAGS]       # raw mode has no not-found flags
    return list(NUMERIC_FEATURES), flags, list(CATEGORICAL_FEATURES)


def build_features(data, sentinel_mode="flag"):
    """Apply all feature steps in order and return UID + model features (+ Target if present)."""
    out = handle_sentinels(data, mode=sentinel_mode)
    out = add_platform_share(out)
    out = encode_ordinals(out)
    out = clean_categories(out)

    numeric, flags, categorical = feature_groups(sentinel_mode)
    keep = ["UID"] + numeric + flags + categorical + (["Target"] if "Target" in out.columns else [])
    return out[keep]
