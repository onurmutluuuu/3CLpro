from src.library import pd, np, os, sys, train_test_split, StandardScaler, os

# Import necessary tools from the centralized library
from src.library import train_test_split, StandardScaler

# Import configuration constants for reproducibility
from src.config import RANDOM_SEED, TRAIN_SIZE, TEST_SIZE, VAL_SIZE


def load_and_preprocess_data(file_path, y_column, columns_to_drop=None):
    """
    Loads the dataset, performs cleaning, splits into Train/Test/Validation sets,
    and applies Standard Scaling to features.

    Methodology:
    1. Load data from CSV or Excel.
    2. Drop missing values (NaNs).
    3. Separate Target (y) and Features (X).
    4. Remove specified metadata columns (e.g., Names, SMILES) to prevent data leakage.
    5. Split data into 70% Train, 20% Test, and 10% Validation.
    6. Apply StandardScaler (Fit on Train, Transform on Test/Val).

    Args:
        file_path (str): Path to the source file (.csv or .xlsx).
        y_column (str): The name of the target column to predict.
        columns_to_drop (list): Additional columns to remove from X (e.g., identifiers).

    Returns:
        dict: A dictionary containing processed arrays and the scaler object, 
              or None if an error occurs.
    """
    print(f"\n[Data Preprocessing] Initiating pipeline for file: {os.path.basename(file_path)}")

    data_bundle = None
    df = None

    try:
        # PHASE 1: Data Loading
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file at '{file_path}' does not exist.")

        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")

        initial_shape = df.shape
        df = df.dropna()
        dropped_rows = initial_shape[0] - df.shape[0]

        if dropped_rows > 0:
            print(f"[Data Preprocessing] Warning: Dropped {dropped_rows} rows containing missing values (NaNs).")

        # PHASE 2: Feature & Target Separation
        if y_column not in df.columns:
            raise KeyError(f"Target column '{y_column}' not found in the dataset.")

        # Extract Target Variable
        y = df[y_column].values

        # Prepare Feature Matrix (X)
        # We must exclude the target itself from the features
        cols_to_remove = [y_column]

        if columns_to_drop:
            # Validate which columns actually exist before trying to drop them
            existing_cols_to_drop = [c for c in columns_to_drop if c in df.columns]
            ignored_cols = set(columns_to_drop) - set(existing_cols_to_drop)

            if ignored_cols:
                print(f"[Data Preprocessing] Note: The following columns were not found and ignored: {ignored_cols}")

            cols_to_remove.extend(existing_cols_to_drop)

        X = df.drop(columns=cols_to_remove)
        feature_names = X.columns.tolist()

        print(f"[Data Preprocessing] Target Variable: '{y_column}'")
        print(f"[Data Preprocessing] Feature Count: {len(feature_names)}")

        # PHASE 3: Data Splitting (70% Train, 20% Test, 10% Validation)

        # Step A: Split 70% Train and 30% Temporary (Test + Val)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(TEST_SIZE + VAL_SIZE), random_state=RANDOM_SEED
        )

        # Step B: Split the 30% Temporary set into Test (20% total) and Validation (10% total)
        split_ratio = VAL_SIZE / (TEST_SIZE + VAL_SIZE)

        X_test, X_val, y_test, y_val = train_test_split(
            X_temp, y_temp, test_size=split_ratio, random_state=RANDOM_SEED
        )

        # ---------------------------------------------------------------------
        # PHASE 4: Feature Scaling (Standardization)
        # ---------------------------------------------------------------------
        # Critical: Fit only on TRAIN data to prevent data leakage.
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_val_scaled = scaler.transform(X_val)

        # Store results in a dictionary
        data_bundle = {
            'X_train': X_train_scaled,
            'y_train': y_train,
            'X_test': X_test_scaled,
            'y_test': y_test,
            'X_val': X_val_scaled,
            'y_val': y_val,
            'feature_names': feature_names,
            'scaler': scaler
        }

    except FileNotFoundError as fnf_error:
        print(f"[Data Preprocessing] Error: {fnf_error}")
        return None
    except KeyError as key_error:
        print(f"[Data Preprocessing] Column Error: {key_error}")
        return None
    except Exception as e:
        print(f"[Data Preprocessing] Unexpected Critical Error: {e}")
        return None

    else:
        # Executed only if no exceptions occurred
        print(
            f"[Data Preprocessing] Success. Split Sizes -> Train: {len(X_train)}, Test: {len(X_test)}, Val: {len(X_val)}")
        return data_bundle

    finally:
        # Always executed (Clean up or status report)
        print("[Data Preprocessing] Pipeline finished.")