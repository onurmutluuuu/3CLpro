
from src.library import (
    pd, np, plt, sns, GridSearchCV, warnings, joblib, time, sys, os,
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    XGBClassifier, RandomForestClassifier, LGBMClassifier, CatBoostClassifier, MLPClassifier
)

from src.config import (
    DATA_RAW, RANDOM_SEED, CV_FOLDS, N_JOBS, THRESHOLD_50,
    XGB_CLS_GRID, RF_CLS_GRID, LIGHTGBM_CLS_GRID, CATBOOST_CLS_GRID, MLP_CLS_GRID
)

from src.data_preprocessing import load_and_preprocess_data

from src.visualization import (
    plot_confusion_matrix_heatmap,
    plot_roc_curve_custom,
    plot_feature_importance
)

INPUT_FILE_NAME = "Selected_features_insilico_invitro_v3.xlsx"
TEMP_FILE_NAME = "temp_step2_processed.csv"
TARGET_COLUMN = "Target_Class_50"
RAW_TARGET_COLUMN = "Inhibition_new"
COLUMNS_TO_DROP = ["Binding Affinity", "Pathway", "Target", "Name", "Inhibition_new"]
OUTPUT_DIR = os.path.join("outputs", "step2_classification_50")
THRESHOLD = THRESHOLD_50


def run_classification_step2():
    start_time = time.time()
    print("\nSTEP 2: CLASSIFICATION ANALYSIS (THRESHOLD 0.5)")
    print(f"Objective: Predict Active/Inactive (Threshold: {THRESHOLD}) without Binding Affinity")

    try:
        raw_path = os.path.join(DATA_RAW, INPUT_FILE_NAME)
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Input file not found: {raw_path}")

        df_raw = pd.read_excel(raw_path)

        if RAW_TARGET_COLUMN not in df_raw.columns:
            raise KeyError(f"Column '{RAW_TARGET_COLUMN}' not found in Excel file.")

        df_raw[TARGET_COLUMN] = (df_raw[RAW_TARGET_COLUMN] >= THRESHOLD).astype(int)

        balance = df_raw[TARGET_COLUMN].value_counts(normalize=True)
        print(f"Class Balance:\n{balance}")

        temp_path = os.path.join("data", "processed", TEMP_FILE_NAME)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        df_raw.to_csv(temp_path, index=False)
        print(f"Temporary processed file created at: {temp_path}")

        data = load_and_preprocess_data(
            file_path=temp_path,
            y_column=TARGET_COLUMN,
            columns_to_drop=COLUMNS_TO_DROP
        )

        if data is None:
            raise ValueError("Data preprocessing returned None.")

        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
        feature_names = data['feature_names']

        print(f"Training Samples: {X_train.shape[0]}")
        print(f"Test Samples: {X_test.shape[0]}")

        if os.path.exists(temp_path):
            os.remove(temp_path)

    except Exception as e:
        print(f"CRITICAL ERROR in Data Preparation: {e}")
        return

    models_to_optimize = {
        "XGBoost": (
            XGBClassifier(random_state=RANDOM_SEED, verbosity=0, use_label_encoder=False),
            XGB_CLS_GRID
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_SEED),
            RF_CLS_GRID
        ),
        "LightGBM": (
            LGBMClassifier(random_state=RANDOM_SEED, verbose=-1),
            LIGHTGBM_CLS_GRID
        ),
        "CatBoost": (
            CatBoostClassifier(random_state=RANDOM_SEED, verbose=False, allow_writing_files=False),
            CATBOOST_CLS_GRID
        ),
        "MLP": (
            MLPClassifier(random_state=RANDOM_SEED, max_iter=2000),
            MLP_CLS_GRID
        )
    }

    best_models = {}
    results = []

    print(f"\nStarting Deep Hyperparameter Optimization (CV={CV_FOLDS})...")

    for name, (model, grid) in models_to_optimize.items():
        print(f"Optimizing {name}...")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=grid,
                    cv=CV_FOLDS,
                    scoring='accuracy',
                    n_jobs=N_JOBS,
                    verbose=1
                )

                grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)

            if hasattr(best_model, "predict_proba"):
                y_proba = best_model.predict_proba(X_test)[:, 1]
                auc_score = roc_auc_score(y_test, y_proba)
            else:
                auc_score = 0.5

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            best_models[name] = best_model
            results.append({
                "Model": name,
                "Best CV Accuracy": grid_search.best_score_,
                "Test Accuracy": acc,
                "Test F1": f1,
                "Test AUC": auc_score,
                "Best Params": grid_search.best_params_
            })
            print(f"Done {name} | Test Acc: {acc:.4f} | AUC: {auc_score:.4f}")

        except Exception as e:
            print(f"Error optimizing {name}: {e}")

    if not results:
        print("No models optimized successfully.")
        return

    try:
        results_df = pd.DataFrame(results).sort_values(by="Test Accuracy", ascending=False)
        best_model_name = results_df.iloc[0]["Model"]
        winner_model = best_models[best_model_name]

        print(f"\nCHAMPION MODEL: {best_model_name}")
        print(f"Accuracy: {results_df.iloc[0]['Test Accuracy']:.4f}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        results_df.to_csv(os.path.join(OUTPUT_DIR, "classification_results.csv"), index=False)
        joblib.dump(winner_model, os.path.join(OUTPUT_DIR, f"best_model_{best_model_name.replace(' ', '_')}.pkl"))

        y_pred_winner = winner_model.predict(X_test)
        plot_confusion_matrix_heatmap(
            y_test,
            y_pred_winner,
            title=f"Confusion Matrix ({best_model_name})",
            save_path=os.path.join(OUTPUT_DIR, "fig_confusion_matrix.png")
        )

        if hasattr(winner_model, "predict_proba"):
            y_proba_winner = winner_model.predict_proba(X_test)[:, 1]
            plot_roc_curve_custom(
                y_test,
                y_proba_winner,
                model_name=best_model_name,
                save_path=os.path.join(OUTPUT_DIR, "fig_roc_curve.png")
            )

        if hasattr(winner_model, "feature_importances_"):
            plot_feature_importance(
                winner_model.feature_importances_,
                feature_names=feature_names,
                title=f"Feature Importance ({best_model_name})",
                save_path=os.path.join(OUTPUT_DIR, "fig_feature_importance.png")
            )

        print(f"Outputs saved to: {OUTPUT_DIR}")

    except Exception as e:
        print(f"Error in Saving/Visualization phase: {e}")

    print(f"Pipeline completed in {(time.time() - start_time) / 60:.2f} minutes.")


if __name__ == "__main__":
    run_classification_step2()