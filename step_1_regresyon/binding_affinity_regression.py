# IMPORTS

from src.library import (
    pd, np, plt, sns, GridSearchCV, os, sys, joblib, time, warnings,
    r2_score, mean_squared_error, sqrt,
    XGBRegressor, RandomForestRegressor, LGBMRegressor, CatBoostRegressor, MLPRegressor
)

from src.config import (
    DATA_RAW, RANDOM_SEED, CV_FOLDS, N_JOBS,
    XGB_REG_GRID, RF_REG_GRID, LIGHTGBM_REG_GRID, CATBOOST_REG_GRID, MLP_REG_GRID
)


from src.data_preprocessing import load_and_preprocess_data
from src.visualization import (
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_residual_error
)


# 2. CONFIGURATION & CONSTANTS

INPUT_FILE_NAME = "Selected_features_insilico_invitro_v3.xlsx"
TARGET_COLUMN = "Binding Affinity"
COLUMNS_TO_DROP = ["Pathway", "Target", "Name", "Inhibition_new"]
OUTPUT_DIR = os.path.join("outputs", "step1_regression")



# 3. MAIN PIPELINE

def run_regression_pipeline():
    """
    Executes the complete regression pipeline:
    1. Data Loading & Cleaning
    2. Deep Hyperparameter Optimization (5 Algorithms)
    3. Evaluation & Selection
    4. Visualization & Model Saving
    """
    start_time = time.time()
    print("\n" + "=" * 80)
    print("STEP 1: REGRESSION ANALYSIS PIPELINE")
    print(f"Target Variable: '{TARGET_COLUMN}'")
    print("=" * 80)


    # PHASE A: DATA PREPARATION

    print("\n[Phase A] Loading and Preprocessing Data...")
    try:
        file_path = os.path.join(DATA_RAW, INPUT_FILE_NAME)

        # Check file existence explicitly before calling loader
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found at: {file_path}")

        data = load_and_preprocess_data(
            file_path=file_path,
            y_column=TARGET_COLUMN,
            columns_to_drop=COLUMNS_TO_DROP
        )

        if data is None:
            raise ValueError("Data preprocessing failed (returned None). Check logs.")

        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
        feature_names = data['feature_names']

        # CRITICAL CHECK: Ensure Target is Numeric (Regression Requirement)
        if not np.issubdtype(y_train.dtype, np.number):
            raise TypeError(f"Target column '{TARGET_COLUMN}' is not numeric! Cannot perform regression.")

        print(f"   -> Training Samples: {X_train.shape[0]}")
        print(f"   -> Test Samples:     {X_test.shape[0]}")
        print(f"   -> Features:         {X_train.shape[1]}")

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Phase A Failed: {e}")
        return


    # PHASE B: MODEL DEFINITIONS

    # Mapping Algorithms to their Deep Grids
    models_to_optimize = {
        "XGBoost": (
            XGBRegressor(random_state=RANDOM_SEED, verbosity=0),
            XGB_REG_GRID
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=RANDOM_SEED),
            RF_REG_GRID
        ),
        "LightGBM": (
            LGBMRegressor(random_state=RANDOM_SEED, verbose=-1),
            LIGHTGBM_REG_GRID
        ),
        "CatBoost": (
            CatBoostRegressor(random_state=RANDOM_SEED, verbose=False, allow_writing_files=False),
            CATBOOST_REG_GRID
        ),
        "MLP": (
            MLPRegressor(random_state=RANDOM_SEED, max_iter=2000),
            MLP_REG_GRID
        )
    }

    best_models = {}
    results = []

    print("\n" + "-" * 80)
    print(f"[Phase B] Starting Deep Hyperparameter Optimization (CV={CV_FOLDS})")
    print("Note: This process may take a significant amount of time depending on the grid size.")
    print("-" * 80)


    # PHASE C: OPTIMIZATION LOOP

    for name, (model, grid) in models_to_optimize.items():
        print(f"\n> Optimizing Algorithm: {name}...")
        model_start = time.time()

        try:
            # Suppress Warnings (Convergence warnings, FutureWarnings etc.) for cleaner output
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Setup Grid Search
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=grid,
                    cv=CV_FOLDS,
                    scoring='r2',
                    n_jobs=N_JOBS,
                    verbose=1
                )

                # Execute Training
                grid_search.fit(X_train, y_train)

            # Extract Best Results
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_cv_score = grid_search.best_score_

            # Final Evaluation on Held-out Test Set
            y_pred = best_model.predict(X_test)
            test_r2 = r2_score(y_test, y_pred)
            test_rmse = sqrt(mean_squared_error(y_test, y_pred))

            # Store
            best_models[name] = best_model
            results.append({
                "Model": name,
                "Best CV R2": best_cv_score,
                "Test R2": test_r2,
                "Test RMSE": test_rmse,
                "Best Params": best_params
            })

            elapsed = time.time() - model_start
            print(f"  [DONE] {name} | Test R2: {test_r2:.4f} | Time: {elapsed:.2f}s")

        except KeyboardInterrupt:
            print(f"  [STOPPED] User interrupted optimization for {name}.")
            sys.exit()
        except Exception as e:
            print(f"  [ERROR] Optimization failed for {name}. Details: {e}")
            # Continue to next model instead of crashing


    # PHASE D: REPORTING & SELECTION

    if not results:
        print("\n[FAILURE] No models were successfully optimized. Exiting.")
        return

    try:
        results_df = pd.DataFrame(results).sort_values(by="Test R2", ascending=False)

        print("\n" + "=" * 80)
        print("FINAL PERFORMANCE LEADERBOARD (Sorted by Test R2)")
        print("=" * 80)
        print(results_df[["Model", "Test R2", "Test RMSE", "Best CV R2"]].to_string(index=False))

        # Select Winner
        best_model_name = results_df.iloc[0]["Model"]
        winner_model = best_models[best_model_name]

        print(f"\n>>> CHAMPION MODEL: {best_model_name}")
        print(f">>> R2 Score: {results_df.iloc[0]['Test R2']:.4f}")

    except Exception as e:
        print(f"[Phase D] Error in reporting: {e}")
        return


    # PHASE E: SAVING & VISUALIZATION

    print(f"\n[Phase E] Saving outputs to: {OUTPUT_DIR}")

    try:
        # Create directory if not exists
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR, exist_ok=True)

        csv_path = os.path.join(OUTPUT_DIR, "regression_results.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"   -> Saved Results Table: {csv_path}")

        model_path = os.path.join(OUTPUT_DIR, f"best_model_{best_model_name.replace(' ', '_')}.pkl")
        joblib.dump(winner_model, model_path)
        print(f"   -> Saved Model File: {model_path}")

        print("   -> Generating Plots...")
        y_pred_winner = winner_model.predict(X_test)

        plot_actual_vs_predicted(
            y_test,
            y_pred_winner,
            title=f"Actual vs Predicted ({best_model_name})",
            save_path=os.path.join(OUTPUT_DIR, "fig_actual_vs_predicted.png")
        )

        print("   -> Generating Residual Plot...")
        plot_residual_error(
            y_test,
            y_pred_winner,
            save_path=os.path.join(OUTPUT_DIR, "fig_residual_plot.png")
        )

        if hasattr(winner_model, "feature_importances_"):
            plot_feature_importance(
                winner_model.feature_importances_,
                feature_names=feature_names,
                title=f"Feature Importance ({best_model_name})",
                save_path=os.path.join(OUTPUT_DIR, "fig_feature_importance.png")
            )
        else:
            print(f"   -> Note: {best_model_name} does not support feature importance plotting.")

    except OSError as e:
        print(f"   [ERROR] File I/O Error: {e}")
    except Exception as e:
        print(f"   [ERROR] Visualization/Saving Error: {e}")

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED SUCCESSFULLY in {total_time / 60:.2f} minutes.")
    print("=" * 80)


if __name__ == "__main__":
    run_regression_pipeline()