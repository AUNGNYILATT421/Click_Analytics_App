# predict_with_bundle.py
# Predict raw new data outside the app with a model downloaded from Click Analytics ("Download Model").
# The file contains the model and the recorded preparation steps, so new rows are prepared exactly like the training data.
#
#   python predict_with_bundle.py model.pkl new_data.csv predictions.csv
#
# Run it from this app's folder with the packages in requirements.txt installed.
# Only load model files you created yourself: loading a .pkl file can run code stored in it.

import sys
import joblib
from data_loading import load_data
from prediction import predict_table


def main(model_path, data_path, out_path):
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or bundle.get("format") != "click-analytics-model":
        sys.exit("This file isn't a Click Analytics model download.")
    df = load_data(data_path).reset_index(drop=True)
    missing = [c for c in bundle["needed_inputs"] if c not in df.columns]
    if missing:
        sys.exit(f"The data is missing these columns: {missing}")
    result, problem = predict_table(df, bundle["model"], bundle["needed_inputs"], bundle["input_schema"], bundle["recipe"],
                                    bundle["model_columns"], bundle["problem_type"])
    if problem:
        sys.exit(f"The rows can't be predicted: {problem}")
    result.to_csv(out_path, index=False)
    failed = int(result["Problem"].notna().sum())
    print(f"Predicted {len(result) - failed} of {len(result)} rows of '{bundle['target']}'; saved to {out_path}."
          + (f" {failed} rows can't be predicted: see the Problem column." if failed else ""))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("Usage: python predict_with_bundle.py model.pkl new_data.csv predictions.csv")
    main(*sys.argv[1:])
