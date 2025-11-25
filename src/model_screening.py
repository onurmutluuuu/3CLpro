from src.library import (
    pd, warnings,
    r2_score,
    LinearRegression, Ridge, Lasso,
    DecisionTreeRegressor, GradientBoostingRegressor, RandomForestRegressor,
    XGBRegressor, LGBMRegressor, CatBoostRegressor, MLPRegressor
)
from src.config import RANDOM_SEED


def screen_regression_models(data_bundle):
    """
    Evaluates 10 candidate regression algorithms (7 non-linear, 3 linear)
    using the pre-processed dataset to identify the top performers.

    Methodology:
    It trains each model using default or standard parameters on the Training set
    and evaluates performance (R2 Score) on the Test set.

    Args:
        data_bundle (dict): Dictionary containing 'X_train', 'y_train', 'X_test', 'y_test'.

    Returns:
        top_5_models (list): List of names of the 5 highest-performing models.
        results_df (DataFrame): A DataFrame containing Train and Test R2 scores for all models.
    """

    print("\n" + "=" * 60)
    print("STEP: Initial Screening of 10 Regression Models")
    print("Objective: Select top 5 candidates based on Test R2 Score")
    print("=" * 60)

    # 1. Validate and Unpack Data
    if data_bundle is None:
        print("Error: Data bundle is None. Cannot proceed with screening.")
        return [], None

    try:
        X_train = data_bundle['X_train']
        y_train = data_bundle['y_train']
        X_test = data_bundle['X_test']
        y_test = data_bundle['y_test']
    except KeyError as e:
        print(f"Critical Error: Missing required data keys in bundle. Details: {e}")
        return [], None

    models = {
        # --- Linear Models ---
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(random_state=RANDOM_SEED),
        "Lasso Regression": Lasso(random_state=RANDOM_SEED),

        # --- Non-Linear Models ---
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_SEED),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_SEED),
        "Random Forest": RandomForestRegressor(random_state=RANDOM_SEED),
        "XGBoost": XGBRegressor(random_state=RANDOM_SEED, verbosity=0),
        "LightGBM": LGBMRegressor(random_state=RANDOM_SEED, verbose=-1),
        "CatBoost": CatBoostRegressor(random_state=RANDOM_SEED, verbose=False, allow_writing_files=False),
        "MLP": MLPRegressor(random_state=RANDOM_SEED, max_iter=1000)
    }

    results = []

    print("-" * 75)
    print(f"{'Model Name':<20} | {'Train R2':<10} | {'Test R2':<10} | {'Status':<10}")
    print("-" * 75)

    for name, model in models.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                model.fit(X_train, y_train)

                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)

                r2_train = r2_score(y_train, y_pred_train)
                r2_test = r2_score(y_test, y_pred_test)

                results.append({
                    "Model": name,
                    "Train R2": r2_train,
                    "Test R2": r2_test
                })

                print(f"{name:<20} | {r2_train:.4f}     | {r2_test:.4f}     | OK")

        except Exception as e:
            print(f"{name:<20} | {'N/A':<10} | {'N/A':<10} | ERROR")
            # Optional: Print the specific error for debugging
            # print(f"  -> {e}")

    if not results:
        print("\nCritical Warning: No models were trained successfully.")
        return [], None

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(by="Test R2", ascending=False).reset_index(drop=True)

    top_5_models = results_df.head(5)["Model"].tolist()

    print("-" * 75)
    print("Selection Complete. The top 5 models have been identified for optimization:")
    for idx, model_name in enumerate(top_5_models, 1):
        score = results_df[results_df['Model'] == model_name]['Test R2'].values[0]
        print(f"   {idx}. {model_name:<20} (Test R2: {score:.4f})")
    print("=" * 60 + "\n")

    return top_5_models, results_df