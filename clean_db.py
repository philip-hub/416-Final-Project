import pandas as pd
import json

# === CONFIG ===
INPUT_FILE = "train.csv"
OUTPUT_FILE = "filtered_train_simple_models.csv"

ALLOWED_FEATURES = {"sketch", "extrusion", "extrude"}
ALLOWED_EXTRUDE_OPS = {
    
    "NewBodyFeatureOperation",
}
MAX_OPERATIONS = 2

def extract_features_and_ops(json_text):
    """Return (features, extrude_ops, total_ops) from the CAD JSON."""
    features, ops = set(), set()
    total_ops = 0
    try:
        data = json.loads(json_text)
        parts = data.get("parts", {})
        for _, part in parts.items():
            if not isinstance(part, dict):
                continue
            for k, v in part.items():
                if k in ("coordinate_system", "description"):
                    continue
                total_ops += 1
                features.add(k.lower())
                if k.lower() in ("extrusion", "extrude") and isinstance(v, dict):
                    op = v.get("operation")
                    if isinstance(op, str):
                        ops.add(op)
    except Exception:
        pass
    return features, ops, total_ops

# Load dataset
df = pd.read_csv(INPUT_FILE)

# Extract features, extrusion ops, and op count
df["features"], df["extrude_ops"], df["total_ops"] = zip(*df["json_desc"].astype(str).map(extract_features_and_ops))

# Apply filters
mask_features = df["features"].apply(lambda s: s.issubset(ALLOWED_FEATURES))
mask_ops = df["extrude_ops"].apply(lambda s: all(op in ALLOWED_EXTRUDE_OPS for op in s))
mask_count = df["total_ops"] <= MAX_OPERATIONS

filtered_df = df[mask_features & mask_ops & mask_count].drop(columns=["features", "extrude_ops", "total_ops"])

# Save filtered dataset
filtered_df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Filtered dataset saved as {OUTPUT_FILE}")
print(f"Original: {len(df)}, Remaining: {len(filtered_df)}")
