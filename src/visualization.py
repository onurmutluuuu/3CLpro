# src/visualization.py

from src.library import pd, np, sns, plt, os, confusion_matrix, roc_curve, auc

# Global style settings
sns.set(style="whitegrid", font_scale=1.2)
plt.rcParams.update({'figure.max_open_warning': 0})


def save_plot(fig, output_path):
    """
    Helper to save figures with high resolution safely.
    """
    try:
        if output_path:
            directory = os.path.dirname(output_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"[Visualization] Saved: {output_path}")
        plt.close(fig)
    except Exception as e:
        print(f"[Visualization] Error saving plot to '{output_path}': {e}")


def plot_binding_affinity_distribution(df, col_affinity, save_path=None):
    """
    Generates Figure 2: Distribution of binding affinity scores.
    Also covers basic histograms from the notebook.
    """
    try:
        if col_affinity not in df.columns:
            raise KeyError(f"Column '{col_affinity}' not found in DataFrame.")

        plt.figure(figsize=(12, 6))
        sns.histplot(data=df, x=col_affinity, bins=30, kde=False, color='#4c72b0', edgecolor='white')

        # Add vertical lines for controls (Specific to manuscript context)
        controls = {'Lopinavir': -8.73, 'Remdesivir': -7.88, 'GC-376': -7.87}
        colors = ['cyan', 'pink', 'brown']

        for (name, val), color in zip(controls.items(), colors):
            plt.axvline(val, color=color, linestyle='--', linewidth=2, label=f'{name}: {val}')

        plt.title('Distribution of Binding Affinity Values', fontsize=16, fontweight='bold')
        plt.xlabel('Binding Affinity (kcal/mol)', fontsize=14)
        plt.ylabel('Molecule Count', fontsize=14)
        plt.legend()

        if save_path:
            save_plot(plt.gcf(), save_path)
        else:
            plt.show()

    except Exception as e:
        print(f"[Visualization] Error in plot_binding_affinity_distribution: {e}")


def plot_inhibition_scatter(df, col_inhibition, threshold=50, save_path=None):
    """
    Generates Figure 3: Normalized in-vitro inhibition percentages.
    """
    try:
        if col_inhibition not in df.columns:
            raise KeyError(f"Column '{col_inhibition}' not found in DataFrame.")

        plt.figure(figsize=(14, 6))

        # Scatter plot
        plt.scatter(range(len(df)), df[col_inhibition], alpha=0.7, color='purple', s=30)

        # Threshold line
        plt.axhline(threshold, color='darkred', linestyle='-', linewidth=2)
        plt.text(len(df) * 0.9, threshold + 2, f'{threshold}% Threshold', color='darkred', fontweight='bold')

        plt.title('Normalized 3CLpro Inhibition (%) Across Molecule Library', fontsize=16, fontweight='bold')
        plt.xlabel('Molecule Number', fontsize=14)
        plt.ylabel('Normalized 3CLpro Inhibition %', fontsize=14)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        if save_path:
            save_plot(plt.gcf(), save_path)
        else:
            plt.show()

    except Exception as e:
        print(f"[Visualization] Error in plot_inhibition_scatter: {e}")


def plot_correlation_affinity_vs_inhibition(df, x_col, y_col, save_path=None):
    """
    Generates Figure 4: Correlation between In-Silico Binding Affinity and In-Vitro Inhibition.
    """
    try:
        if x_col not in df.columns or y_col not in df.columns:
            raise KeyError(f"Columns '{x_col}' or '{y_col}' not found.")

        plt.figure(figsize=(10, 8))
        sns.regplot(data=df, x=x_col, y=y_col, scatter_kws={'alpha': 0.6, 'color': 'royalblue'},
                    line_kws={'color': 'red', 'linestyle': '--'})

        # Calculate Correlation safely
        if len(df) > 1:
            corr = df[[x_col, y_col]].corr().iloc[0, 1]
            plt.text(0.05, 0.95, f'r = {corr:.2f}', transform=plt.gca().transAxes,
                     fontsize=14, bbox=dict(facecolor='white', alpha=0.8))

        plt.title('Binding Affinity vs Inhibition - Correlation', fontsize=16)
        plt.xlabel(x_col, fontsize=14)
        plt.ylabel(y_col, fontsize=14)

        if save_path:
            save_plot(plt.gcf(), save_path)
        else:
            plt.show()

    except Exception as e:
        print(f"[Visualization] Error in plot_correlation_affinity_vs_inhibition: {e}")


def plot_complex_joint_figure(df, col_affinity, col_logp, col_mw, save_path=None):
    """
    Generates Figure 5: Complex dual-joint plot layout for Molecular Descriptors.
    """
    try:
        required_cols = [col_affinity, col_logp, col_mw]
        if not all(col in df.columns for col in required_cols):
            raise KeyError(f"One of the required columns {required_cols} is missing.")

        fig = plt.figure(figsize=(15, 6))

        y_min = df[col_affinity].min() - 0.5
        y_max = df[col_affinity].max() + 0.5

        # Left Panel (LogP)
        gs1 = fig.add_gridspec(nrows=2, ncols=2, left=0.05, right=0.48, bottom=0.1, top=0.9,
                               width_ratios=[4, 1], height_ratios=[1, 4],
                               wspace=0.05, hspace=0.05)

        ax1_main = fig.add_subplot(gs1[1, 0])
        ax1_top = fig.add_subplot(gs1[0, 0], sharex=ax1_main)
        ax1_right = fig.add_subplot(gs1[1, 1], sharey=ax1_main)

        sns.regplot(data=df, x=col_logp, y=col_affinity, ax=ax1_main, color='red', truncate=False,
                    scatter_kws={'color': '#4682B4', 's': 30, 'alpha': 0.7},
                    line_kws={'color': 'red'})

        sns.histplot(data=df, x=col_logp, ax=ax1_top, color='red', kde=True, element="step")
        sns.histplot(data=df, y=col_affinity, ax=ax1_right, color='red', kde=True, element="step")

        ax1_top.axis('off')
        ax1_right.axis('off')
        ax1_main.set_ylim(y_min, y_max)
        ax1_main.set_xlabel('\n' + col_logp, fontsize=14, fontweight='bold')
        ax1_main.set_ylabel(col_affinity, fontsize=14, fontweight='bold')
        ax1_main.text(-0.1, 1.05, 'A', transform=ax1_main.transAxes, fontsize=20, fontweight='bold')

        # Right Panel (MW)
        gs2 = fig.add_gridspec(nrows=2, ncols=2, left=0.55, right=0.98, bottom=0.1, top=0.9,
                               width_ratios=[4, 1], height_ratios=[1, 4],
                               wspace=0.05, hspace=0.05)

        ax2_main = fig.add_subplot(gs2[1, 0])
        ax2_top = fig.add_subplot(gs2[0, 0], sharex=ax2_main)
        ax2_right = fig.add_subplot(gs2[1, 1], sharey=ax2_main)

        sns.regplot(data=df, x=col_mw, y=col_affinity, ax=ax2_main, color='blue', truncate=False,
                    scatter_kws={'color': 'purple', 's': 30, 'alpha': 0.7},
                    line_kws={'color': 'blue'})

        sns.histplot(data=df, x=col_mw, ax=ax2_top, color='blue', kde=True, element="step")
        sns.histplot(data=df, y=col_affinity, ax=ax2_right, color='blue', kde=True, element="step")

        ax2_top.axis('off')
        ax2_right.axis('off')
        ax2_main.set_ylim(y_min, y_max)
        ax2_main.set_xlabel('\n' + col_mw, fontsize=14, fontweight='bold')
        ax2_main.set_ylabel(col_affinity, fontsize=14, fontweight='bold')
        ax2_main.text(-0.1, 1.05, 'B', transform=ax2_main.transAxes, fontsize=20, fontweight='bold')

        if save_path:
            save_plot(fig, save_path)
        else:
            plt.show()

    except Exception as e:
        print(f"[Visualization] Error in plot_complex_joint_figure: {e}")


def plot_boxplot_distribution(df, x_col, y_col=None, title="Boxplot Distribution", save_path=None):
    """
    Generates Boxplots as seen in visualization_.ipynb (Cell 12, 14).
    If y_col is None, it plots a single variable distribution.
    If both x_col and y_col are provided, it plots x vs y (e.g., Pathway vs Binding Affinity).
    """
    try:
        plt.figure(figsize=(12, 8))
        if y_col:
            sns.boxplot(x=x_col, y=y_col, data=df, palette="Set3")
            plt.xticks(rotation=45, ha='right')
        else:
            sns.boxplot(x=df[x_col], palette="Set3")

        plt.title(title, fontsize=16)

        if save_path:
            save_plot(plt.gcf(), save_path)
        else:
            plt.show()

    except Exception as e:
        print(f"[Visualization] Error in plot_boxplot_distribution: {e}")


def plot_correlation_bar_target(df, target_col, top_n=20, save_path=None):
    """
    Generates a bar plot showing correlation of all features with the target variable.
    Based on visualization_.ipynb (Cell 15).
    """
    try:
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found.")

        # Calculate correlations
        correlation_matrix = df.corr()
        corr_with_target = correlation_matrix[target_col].drop(target_col)

        # Sort by absolute value to show most important relationships
        corr_with_target = corr_with_target.iloc[corr_with_target.abs().argsort()[::-1]].head(top_n)

        plt.figure(figsize=(12, 6))
        corr_with_target.plot(kind='bar', color='teal', alpha=0.8)
        plt.title(f'Top Correlations with {target_col}', fontsize=16)
        plt.xlabel('Features', fontsize=14)
        plt.ylabel('Pearson Correlation Coefficient', fontsize=14)
        plt.axhline(0, color='black', linewidth=0.8)
        plt.xticks(rotation=45, ha='right')

        if save_path:
            save_plot(plt.gcf(), save_path)
        else:
            plt.show()

    except Exception as e:
        print(f"[Visualization] Error in plot_correlation_bar_target: {e}")


def plot_pairplot(df, columns=None, kind="reg", save_path=None):
    """
    Generates a Pairplot for selected columns.
    Based on visualization_.ipynb (Cell 16).
    """
    try:
        if columns:
            df_subset = df[columns]
        else:
            # Limit to numerical columns and a reasonable number to prevent crash
            df_subset = df.select_dtypes(include=[np.number]).iloc[:, :10]

        g = sns.pairplot(df_subset, kind=kind, diag_kind="kde", plot_kws={'alpha': 0.6})
        g.fig.suptitle("Pairwise Relationships", y=1.02, fontsize=16)

        if save_path:
            save_plot(g.fig, save_path)
        else:
            plt.show()

    except Exception as e:
        print(f"[Visualization] Error in plot_pairplot: {e}")


def plot_joint_distribution(df, x_col, y_col, kind="reg", save_path=None):
    """
    Generates Joint Plots (Scatter + Histogram/KDE).
    Based on visualization_.ipynb (Cell 17-19).
    """
    try:
        g = sns.jointplot(x=x_col, y=y_col, data=df, kind=kind, color="darkblue", height=8)
        g.fig.suptitle(f"Joint Plot: {x_col} vs {y_col}", y=1.02, fontsize=16)

        if save_path:
            save_plot(g.fig, save_path)
        else:
            plt.show()

    except Exception as e:
        print(f"[Visualization] Error in plot_joint_distribution: {e}")


def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted", save_path=None):
    """
    Generates Figure S2: Regression performance plot.
    """
    try:
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.6, color='blue')

        # Perfect fit line
        if len(y_true) > 0 and len(y_pred) > 0:
            min_val = min(min(y_true), min(y_pred))
            max_val = max(max(y_true), max(y_pred))
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)

        plt.title(title, fontsize=16)
        plt.xlabel("Actual Values", fontsize=14)
        plt.ylabel("Predicted Values", fontsize=14)
        plt.axis('equal')

        if save_path:
            save_plot(plt.gcf(), save_path)
        else:
            plt.show()

    except Exception as e:
        print(f"[Visualization] Error in plot_actual_vs_predicted: {e}")


def plot_residual_error(y_true, y_pred, save_path=None):
    """
    Generates Residual Plot (Standard for regression analysis).
    """
    try:
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.6, color='crimson')
        plt.axhline(0, color='black', linestyle='--', lw=2)

        plt.title("Residual Plot", fontsize=16)
        plt.xlabel("Predicted Values", fontsize=14)
        plt.ylabel("Residuals (Actual - Predicted)", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.5)

        if save_path:
            save_plot(plt.gcf(), save_path)
        else:
            plt.show()

    except Exception as e:
        print(f"[Visualization] Error in plot_residual_error: {e}")


def plot_confusion_matrix_heatmap(y_true, y_pred, title="Confusion Matrix", save_path=None):
    """
    Generates Figure S1/S6: Confusion Matrix Heatmap.
    """
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 16})

        plt.title(title, fontsize=16)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        # Assuming binary classification for standard view
        plt.xticks([0.5, 1.5], ['Inactive (0)', 'Active (1)'])
        plt.yticks([0.5, 1.5], ['Inactive (0)', 'Active (1)'])

        if save_path:
            save_plot(plt.gcf(), save_path)
        else:
            plt.show()

    except Exception as e:
        print(f"[Visualization] Error in plot_confusion_matrix_heatmap: {e}")


def plot_roc_curve_custom(y_true, y_proba, model_name="Model", save_path=None):
    """
    Generates Figure 7: ROC Curve.
    """
    try:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title(f'ROC Curve - {model_name}', fontsize=16)
        plt.legend(loc="lower right")

        if save_path:
            save_plot(plt.gcf(), save_path)
        else:
            plt.show()

    except Exception as e:
        print(f"[Visualization] Error in plot_roc_curve_custom: {e}")


def plot_feature_importance(importance_data, feature_names=None, title="Feature Importance", top_n=20, save_path=None):
    """
    Generates Figure 8: Feature Importance Bar Plot.
    Accepts importance_data as a dictionary OR as a list/array (if feature_names provided).
    """
    try:
        # Normalize input to a Series
        if isinstance(importance_data, dict):
            imp_series = pd.Series(importance_data)
        elif feature_names is not None:
            imp_series = pd.Series(importance_data, index=feature_names)
        else:
            # Fallback for array without names
            imp_series = pd.Series(importance_data)

        # Sort and select top N
        imp_series = imp_series.sort_values(ascending=False).head(top_n)

        plt.figure(figsize=(12, 8))
        sns.barplot(x=imp_series.index, y=imp_series.values, palette="viridis")

        plt.title(title, fontsize=16)
        plt.ylabel("Importance Score", fontsize=14)
        plt.xlabel("Features", fontsize=14)
        plt.xticks(rotation=45, ha='right')

        if not imp_series.empty:
            mean_imp = imp_series.mean()
            plt.axhline(mean_imp, color='red', linestyle='--', label='Mean Importance')
            plt.legend()

        if save_path:
            save_plot(plt.gcf(), save_path)
        else:
            plt.show()

    except Exception as e:
        print(f"[Visualization] Error in plot_feature_importance: {e}")


def plot_correlation_heatmap(df, title="Correlation Matrix", save_path=None):
    """
    Generates Figure S4: Correlation Matrix Heatmap.
    Also covers notebook Cell 20.
    """
    try:
        plt.figure(figsize=(20, 15))
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))

        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
                    cbar_kws={"shrink": .8}, vmin=-1, vmax=1)

        plt.title(title, fontsize=18)

        if save_path:
            save_plot(plt.gcf(), save_path)
        else:
            plt.show()

    except Exception as e:
        print(f"[Visualization] Error in plot_correlation_heatmap: {e}")


def plot_optimization_heatmap(results_df, param_x, param_y, metric_col, title="Hyperparameter Optimization",
                              save_path=None):
    """
    Generates a heatmap for hyperparameter search results (Figure 6 context).
    Useful for visualizing Grid Search results.
    """
    try:
        pivot_table = results_df.pivot(index=param_y, columns=param_x, values=metric_col)

        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt='.4f')
        plt.title(title, fontsize=16)
        plt.xlabel(param_x, fontsize=14)
        plt.ylabel(param_y, fontsize=14)

        if save_path:
            save_plot(plt.gcf(), save_path)
        else:
            plt.show()

    except Exception as e:
        print(f"[Visualization] Error in plot_optimization_heatmap: {e}")


def plot_complex_joint_figure(df, col_affinity, col_logp, col_mw, save_path=None):
    """
    Generates Figure 5: Complex dual-joint plot layout for Molecular Descriptors.
    Replicates the specific GridSpec layout and styling from 'figure 5.py'.

    Args:
        df: DataFrame containing the data.
        col_affinity: Column name for Binding Affinity (Y-axis).
        col_logp: Column name for CrippenLogP (X-axis Left).
        col_mw: Column name for MW (X-axis Right).
        save_path: Path to save the output image.
    """
    try:
        # Check required columns
        required_cols = [col_affinity, col_logp, col_mw]
        if not all(col in df.columns for col in required_cols):
            raise KeyError(f"Missing one of the required columns: {required_cols}")

        fig = plt.figure(figsize=(15, 6))

        # Determine global Y-axis limits for consistency
        y_min = df[col_affinity].min() - 0.5
        y_max = df[col_affinity].max() + 0.5

        # LEFT PANEL (CrippenLogP vs Binding Affinity) - RED THEME

        # Specific GridSpec layout from figure 5.py
        gs1 = fig.add_gridspec(nrows=2, ncols=2, left=0.05, right=0.48, bottom=0.1, top=0.9,
                               width_ratios=[4, 1], height_ratios=[1, 4],
                               wspace=0.05, hspace=0.05)

        ax1_main = fig.add_subplot(gs1[1, 0])
        ax1_top = fig.add_subplot(gs1[0, 0], sharex=ax1_main)
        ax1_right = fig.add_subplot(gs1[1, 1], sharey=ax1_main)

        # Scatter + Regplot (Red Line, SteelBlue Scatter)
        sns.regplot(data=df, x=col_logp, y=col_affinity, ax=ax1_main, color='red', truncate=False,
                    scatter_kws={'color': '#4682B4', 's': 30, 'alpha': 0.7},
                    line_kws={'color': 'red'})

        # Histograms
        sns.histplot(data=df, x=col_logp, ax=ax1_top, color='red', kde=True, element="step")
        sns.histplot(data=df, y=col_affinity, ax=ax1_right, color='red', kde=True, element="step")

        # Styling
        ax1_top.axis('off')
        ax1_right.axis('off')
        ax1_main.set_ylim(y_min, y_max)
        ax1_main.tick_params(axis='both', labelsize=12)
        ax1_main.set_xlabel('\n' + col_logp, fontsize=14, fontweight='bold')
        ax1_main.set_ylabel(col_affinity, fontsize=14, fontweight='bold')

        # Annotation 'A'
        ax1_main.text(-0.15, 1.15, 'A', transform=ax1_main.transAxes,
                      fontsize=24, fontweight='bold', va='top', ha='right')

        # RIGHT PANEL (MW vs Binding Affinity) - BLUE THEME
        gs2 = fig.add_gridspec(nrows=2, ncols=2, left=0.55, right=0.98, bottom=0.1, top=0.9,
                               width_ratios=[4, 1], height_ratios=[1, 4],
                               wspace=0.05, hspace=0.05)

        ax2_main = fig.add_subplot(gs2[1, 0])
        ax2_top = fig.add_subplot(gs2[0, 0], sharex=ax2_main)
        ax2_right = fig.add_subplot(gs2[1, 1], sharey=ax2_main)

        # Scatter + Regplot (Blue Line, Purple Scatter)
        sns.regplot(data=df, x=col_mw, y=col_affinity, ax=ax2_main, color='blue', truncate=False,
                    scatter_kws={'color': 'purple', 's': 30, 'alpha': 0.7},
                    line_kws={'color': 'blue'})

        # Histograms
        sns.histplot(data=df, x=col_mw, ax=ax2_top, color='blue', kde=True, element="step")
        sns.histplot(data=df, y=col_affinity, ax=ax2_right, color='blue', kde=True, element="step")

        # Styling
        ax2_top.axis('off')
        ax2_right.axis('off')
        ax2_main.set_ylim(y_min, y_max)
        ax2_main.tick_params(axis='both', labelsize=12)
        ax2_main.set_xlabel('\n' + col_mw, fontsize=14, fontweight='bold')
        ax2_main.set_ylabel(col_affinity, fontsize=14, fontweight='bold')

        # Annotation 'B'
        ax2_main.text(-0.15, 1.15, 'B', transform=ax2_main.transAxes,
                      fontsize=24, fontweight='bold', va='top', ha='right')

        if save_path:
            save_plot(fig, save_path)
        else:
            plt.show()

    except Exception as e:
        print(f"[Visualization] Error in plot_complex_joint_figure: {e}")