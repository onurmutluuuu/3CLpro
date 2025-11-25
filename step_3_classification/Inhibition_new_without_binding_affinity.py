
from src.library import (
    pd, np, plt, sns, GridSearchCV, warnings, joblib, time, sys, os,
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    XGBClassifier, RandomForestClassifier, LGBMClassifier, CatBoostClassifier, MLPClassifier
)

from src.config import (
    DATA_RAW, RANDOM_SEED, CV_FOLDS, N_JOBS,
    XGB_CLS_GRID, RF_CLS_GRID, LIGHTGBM_CLS_GRID, CATBOOST_CLS_GRID, MLP_CLS_GRID,
    THRESHOLD_50, THRESHOLD_20  # Config'den çekilen değerler
)

from src.data_preprocessing import load_and_preprocess_data

from src.visualization import (
    plot_confusion_matrix_heatmap,
    plot_roc_curve_custom,
    plot_feature_importance
)

INPUT_FILE_NAME = "Selected_features_insilico_invitro_v3.xlsx"
TEMP_FILE_PREFIX = "temp_step3_processed"
RAW_TARGET_COLUMN = "Inhibition_new"
# Binding Affinity is NOT dropped here, enabling hybrid prediction
COLUMNS_TO_DROP = ["Pathway", "Target", "Name", "Inhibition_new"]


def run_threshold_analysis(threshold_val, output_folder_name):
    """
    Executes the classification pipeline for a specific inhibition threshold.
    Args:
        threshold_val (float): The threshold value from config (e.g., 0.5 or 50.0).
        output_folder_name (str): Name of the output directory.
    """
    start_time = time.time()
    print(f"\nExecuting Analysis for Threshold: {threshold_val} (Folder: {output_folder_name})")

    output_dir = os.path.join("outputs", output_folder_name)
    # Generate a dynamic column name based on threshold
    target_col_name = f"Target_Class_Thresh_{str(threshold_val).replace('.', '_')}"
    temp_file_path = os.path.join("data", "processed", f"{TEMP_FILE_PREFIX}_{str(threshold_val).replace('.', '_')}.csv")

    try:
        raw_path = os.path.join(DATA_RAW, INPUT_FILE_NAME)
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Source file missing: {raw_path}")

        df_raw = pd.read_excel(raw_path)

        if RAW_TARGET_COLUMN not in df_raw.columns:
            raise KeyError(f"Required column '{RAW_TARGET_COLUMN}' is missing.")

        # Create Binary Target based on current threshold
        # IMPORTANT: Ensure your config value scale matches your data (e.g., 0.5 vs 50.0)
        df_raw[target_col_name] = (df_raw[RAW_TARGET_COLUMN] >= threshold_val).astype(int)

        # Log Class Distribution
        dist = df_raw[target_col_name].value_counts(normalize=True)
        print(f"Class Distribution for Threshold {threshold_val}:\n{dist}")

        # Save Temp File for Preprocessor
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
        df_raw.to_csv(temp_file_path, index=False)

        # Preprocessing Pipeline
        data = load_and_preprocess_data(
            file_path=temp_file_path,
            y_column=target_col_name,
            columns_to_drop=COLUMNS_TO_DROP
        )

        if data is None:
            raise ValueError("Preprocessing failed.")

        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
        feature_names = data['feature_names']

        # Verify Binding Affinity Presence
        if "Binding Affinity" in feature_names:
            print("Confirmation: 'Binding Affinity' is included in the feature set.")
        else:
            print("Warning: 'Binding Affinity' was NOT found in features.")

    except Exception as e:
        print(f"Critical Error during Data Setup (Threshold {threshold_val}): {e}")
        return

    finally:
        # Cleanup temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    # Optimization Phase
    models_to_optimize = {
        "XGBoost": (XGBClassifier(random_state=RANDOM_SEED, verbosity=0, use_label_encoder=False), XGB_CLS_GRID),
        "Random Forest": (RandomForestClassifier(random_state=RANDOM_SEED), RF_CLS_GRID),
        "LightGBM": (LGBMClassifier(random_state=RANDOM_SEED, verbose=-1), LIGHTGBM_CLS_GRID),
        "CatBoost": (
        CatBoostClassifier(random_state=RANDOM_SEED, verbose=False, allow_writing_files=False), CATBOOST_CLS_GRID),
        "MLP": (MLPClassifier(random_state=RANDOM_SEED, max_iter=2000), MLP_CLS_GRID)
    }

    results = []
    best_models = {}

    print(f"Starting Hyperparameter Optimization (CV={CV_FOLDS}) for Threshold {threshold_val}...")

    for name, (model, grid) in models_to_optimize.items():
        try:
            print(f"  > Optimizing {name}...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=grid,
                    cv=CV_FOLDS,
                    scoring='accuracy',
                    n_jobs=N_JOBS,
                    verbose=0
                )
                grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)

            # AUC Calculation
            if hasattr(best_model, "predict_proba"):
                y_proba = best_model.predict_proba(X_test)[:, 1]
                auc_score = roc_auc_score(y_test, y_proba)
            else:
                auc_score = 0.5

            acc = accuracy_score(y_test, y_pred)

            best_models[name] = best_model
            results.append({
                "Model": name,
                "Test Accuracy": acc,
                "Test AUC": auc_score,
                "Best CV Accuracy": grid_search.best_score_,
                "Best Params": grid_search.best_params_
            })
            print(f"    Done. Test Acc: {acc:.4f} | AUC: {auc_score:.4f}")

        except Exception as e:
            print(f"    Error optimizing {name}: {e}")
            continue

    if not results:
        print("No models trained successfully.")
        return

    # Results Processing & Saving
    try:
        os.makedirs(output_dir, exist_ok=True)
        results_df = pd.DataFrame(results).sort_values(by="Test Accuracy", ascending=False)

        print(f"\nLeaderboard (Threshold {threshold_val}):")
        print(results_df[["Model", "Test Accuracy", "Test AUC"]].to_string(index=False))

        winner_name = results_df.iloc[0]["Model"]
        winner_model = best_models[winner_name]

        print(f"\nCHAMPION (Threshold {threshold_val}): {winner_name}")

        # Save Artifacts
        results_df.to_csv(os.path.join(output_dir, "classification_results.csv"), index=False)
        joblib.dump(winner_model, os.path.join(output_dir, f"best_model_{winner_name.replace(' ', '_')}.pkl"))

        # Visualizations
        y_pred_winner = winner_model.predict(X_test)

        plot_confusion_matrix_heatmap(
            y_test, y_pred_winner,
            title=f"Confusion Matrix ({winner_name}) - T{threshold_val}",
            save_path=os.path.join(output_dir, "fig_confusion_matrix.png")
        )

        if hasattr(winner_model, "predict_proba"):
            y_proba_winner = winner_model.predict_proba(X_test)[:, 1]
            plot_roc_curve_custom(
                y_test, y_proba_winner,
                model_name=winner_name,
                save_path=os.path.join(output_dir, "fig_roc_curve.png")
            )

        if hasattr(winner_model, "feature_importances_"):
            plot_feature_importance(
                winner_model.feature_importances_,
                feature_names=feature_names,
                title=f"Feature Importance ({winner_name}) - T{threshold_val}",
                save_path=os.path.join(output_dir, "fig_feature_importance.png")
            )

    except Exception as e:
        print(f"Error during result saving/visualization: {e}")

    print(f"Analysis for Threshold {threshold_val} completed in {(time.time() - start_time) / 60:.2f} minutes.\n")


def run_hybrid_classification_pipeline():
    """
    Main entry point for Step 3.
    Runs the classification pipeline twice using thresholds defined in src.config.
    """
    overall_start = time.time()
    print("=" * 80)
    print("STEP 3: HYBRID CLASSIFICATION (BINDING AFFINITY INCLUDED)")
    print(f"Thresholds from Config: {THRESHOLD_50} and {THRESHOLD_20}")
    print("Reference: Table 7 & Figure 7 in Manuscript")
    print("=" * 80)

    try:
        # Scenario 1: Using THRESHOLD_50 from Config (e.g., 50.0 or 0.5)
        folder_name_50 = f"step3_class_{str(THRESHOLD_50).replace('.', '_')}_hybrid"
        run_threshold_analysis(threshold_val=THRESHOLD_50, output_folder_name=folder_name_50)

        # Scenario 2: Using THRESHOLD_20 from Config (e.g., 20.0 or 0.2)
        folder_name_20 = f"step3_class_{str(THRESHOLD_20).replace('.', '_')}_hybrid"
        run_threshold_analysis(threshold_val=THRESHOLD_20, output_folder_name=folder_name_20)

    except KeyboardInterrupt:
        print("Pipeline interrupted by user.")
        sys.exit()
    except Exception as e:
        print(f"Unexpected Global Error: {e}")

    print("=" * 80)
    print(f"STEP 3 PIPELINE FINISHED. Total Time: {(time.time() - overall_start) / 60:.2f} minutes.")
    print("=" * 80)


if __name__ == "__main__":
    run_hybrid_classification_pipeline()