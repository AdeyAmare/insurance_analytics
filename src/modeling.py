import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Optional XGBoost support (fails gracefully)
try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False



class ModelHelper:
    """
    Helper class for training, evaluating, and analyzing machine learning models
    for regression and classification tasks.

    Attributes
    ----------
    models : dict
        Dictionary to store trained models by name.
    """

    def __init__(self):
        """Initialize the ModelHelper with an empty models dictionary."""
        self.models = {}

    # ----------------------------
    # Training functions
    # ----------------------------

    def train_linear_regression(self, X_train, y_train, name="LinearRegression"):
        """
        Train a Linear Regression model.

        Parameters
        ----------
        X_train : pd.DataFrame or np.ndarray
            Training features.
        y_train : pd.Series or np.ndarray
            Target variable.
        name : str
            Name to store the trained model in self.models.

        Returns
        -------
        model : sklearn.linear_model.LinearRegression
            Trained Linear Regression model.
        """
        model = LinearRegression()
        model.fit(X_train, y_train)
        self.models[name] = model
        return model

    def train_random_forest_regressor(self, X_train, y_train, n_estimators=100, random_state=42, name="RandomForestRegressor"):
        """
        Train a Random Forest Regressor.

        Parameters
        ----------
        X_train : pd.DataFrame or np.ndarray
            Training features.
        y_train : pd.Series or np.ndarray
            Target variable.
        n_estimators : int
            Number of trees in the forest.
        random_state : int
            Random seed for reproducibility.
        name : str
            Name to store the trained model in self.models.

        Returns
        -------
        model : sklearn.ensemble.RandomForestRegressor
            Trained Random Forest Regressor.
        """
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        self.models[name] = model
        return model

    def train_xgboost_regressor(self, X_train, y_train, n_estimators=100, random_state=42, name="XGBRegressor"):
        """
        Train an XGBoost Regressor.

        Parameters
        ----------
        X_train : pd.DataFrame or np.ndarray
            Training features.
        y_train : pd.Series or np.ndarray
            Target variable.
        n_estimators : int
            Number of boosting rounds.
        random_state : int
            Random seed for reproducibility.
        name : str
            Name to store the trained model in self.models.

        Returns
        -------
        model : xgboost.XGBRegressor
            Trained XGBoost Regressor.

        Raises
        ------
        ImportError
            If XGBoost is not installed.
        """
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
        model = XGBRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_train, y_train)
        self.models[name] = model
        return model

    def train_random_forest_classifier(self, X_train, y_train, n_estimators=100, random_state=42, name="RandomForestClassifier"):
        """
        Train a Random Forest Classifier.

        Parameters
        ----------
        X_train : pd.DataFrame or np.ndarray
            Training features.
        y_train : pd.Series or np.ndarray
            Target variable.
        n_estimators : int
            Number of trees in the forest.
        random_state : int
            Random seed for reproducibility.
        name : str
            Name to store the trained model in self.models.

        Returns
        -------
        model : sklearn.ensemble.RandomForestClassifier
            Trained Random Forest Classifier.
        """
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        self.models[name] = model
        return model

    def train_xgboost_classifier(self, X_train, y_train, n_estimators=100, random_state=42, name="XGBClassifier"):
        """
        Train an XGBoost Classifier.

        Parameters
        ----------
        X_train : pd.DataFrame or np.ndarray
            Training features.
        y_train : pd.Series or np.ndarray
            Target variable.
        n_estimators : int
            Number of boosting rounds.
        random_state : int
            Random seed for reproducibility.
        name : str
            Name to store the trained model in self.models.

        Returns
        -------
        model : xgboost.XGBClassifier
            Trained XGBoost Classifier.

        Raises
        ------
        ImportError
            If XGBoost is not installed.
        """
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
        model = XGBClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train)
        self.models[name] = model
        return model
    
    def train_logistic_regression(self, X_train, y_train, random_state=42, name="LogisticRegression"):
        """
        Train a Logistic Regression model for classification tasks.

        Parameters
        ----------
        X_train : pd.DataFrame or np.ndarray
            Training features.
        y_train : pd.Series or np.ndarray
            Target variable (class labels).
        random_state : int
            Random seed for reproducibility.
        name : str
            Name to store the trained model in self.models.

        Returns
        -------
        model : sklearn.linear_model.LogisticRegression
            Trained Logistic Regression model.
        """
        model = LogisticRegression(max_iter=1000, random_state=random_state)
        model.fit(X_train, y_train)
        self.models[name] = model
        return model


    # ----------------------------
    # Evaluation functions
    # ----------------------------

    @staticmethod
    def evaluate_regression(y_true, y_pred):
        """
        Evaluate regression predictions using RMSE and R².

        Parameters
        ----------
        y_true : array-like
            True target values.
        y_pred : array-like
            Predicted values.

        Returns
        -------
        dict
            Dictionary with "rmse" and "r2".
        """
        return {
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "r2": r2_score(y_true, y_pred),
        }

    @staticmethod
    def evaluate_classification(y_true, y_pred):
        """
        Evaluate classification predictions using common metrics.

        Parameters
        ----------
        y_true : array-like
            True class labels.
        y_pred : array-like
            Predicted class labels.

        Returns
        -------
        dict
            Dictionary with "accuracy", "precision", "recall", and "f1".
        """
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }

    # ----------------------------
    # Feature importance
    # ----------------------------

    @staticmethod
    def get_feature_importance(model, feature_names):
        """
        Extract feature importance from tree-based models.

        Parameters
        ----------
        model : estimator
            Trained model with `feature_importances_` attribute.
        feature_names : list of str
            Names of features used in training.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns "feature" and "importance", sorted descending.

        Raises
        ------
        ValueError
            If the model does not provide feature_importances_.
        """
        if not hasattr(model, "feature_importances_"):
            raise ValueError("Model does not provide feature_importances_")
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_,
        })
        return importance_df.sort_values("importance", ascending=False)

    def explain_model_shap(self, model, X, feature_names=None, max_display=20, plot_type="summary"):
        """
        Compute and visualize SHAP values for a model.

        Parameters
        ----------
        model : estimator
            Trained model.
        X : pd.DataFrame or np.ndarray
            Features to explain.
        feature_names : list of str, optional
            Feature names.
        max_display : int
            Max number of features to show.
        plot_type : str
            "summary" for summary plot, "bar" for bar plot, "force" for force plot (local).
        
        Returns
        -------
        shap_values : np.ndarray
            SHAP values computed.
        """
        if not HAS_SHAP:
            raise ImportError("SHAP not installed. Run: pip install shap")

        if isinstance(X, pd.DataFrame):
            feature_names = feature_names or X.columns.tolist()
        else:
            feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]

        explainer = shap.Explainer(model, X)
        shap_values = explainer(X, check_additivity=False)

        if plot_type == "summary":
            shap.summary_plot(shap_values, X, feature_names=feature_names, max_display=max_display)
        elif plot_type == "bar":
            shap.plots.bar(shap_values, max_display=max_display)
        else:
            raise ValueError("plot_type must be 'summary', 'bar', or 'force'")

        return shap_values

    def shap_dependence_plot(self, model, X, feature, feature_names=None, shap_values=None):
        """
        Generate a SHAP dependence plot for a single feature.

        Parameters
        ----------
        model : estimator
            Trained model.
        X : pd.DataFrame or np.ndarray
            Dataset to explain.
        feature : str
            Feature name to plot.
        feature_names : list of str, optional
            List of feature names if X is ndarray.
        shap_values : np.ndarray, optional
            Precomputed SHAP values (optional, can reuse if already computed).

        Returns
        -------
        None (displays the plot)
        """
        if not HAS_SHAP:
            raise ImportError("SHAP not installed. Run: pip install shap")

        if isinstance(X, pd.DataFrame):
            feature_names = feature_names or X.columns.tolist()
        else:
            feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]

        # Compute SHAP values if not provided
        if shap_values is None:
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)

        shap.dependence_plot(feature, shap_values.values, X, feature_names=feature_names)

    
    def compare_models(self, X_test, y_test, task="regression"):
        """
        Evaluate all trained models and rank them by key metrics.

        Parameters
        ----------
        X_test : pd.DataFrame or np.ndarray
            Test features.
        y_test : pd.Series or np.ndarray
            True target values.
        task : str
            "regression" or "classification".

        Returns
        -------
        pd.DataFrame
            DataFrame of models with evaluation metrics sorted by performance.
        """
        results = []

        for name, model in self.models.items():
            y_pred = model.predict(X_test)

            if task == "classification":
                if hasattr(model, "predict_proba"):
                    y_pred_prob = model.predict_proba(X_test)
                    if y_pred_prob.shape[1] > 1:  # multiclass
                        y_pred = y_pred_prob.argmax(axis=1)
                    else:  # binary
                        y_pred = (y_pred_prob[:, 1] >= 0.5).astype(int)
                # Handle floats for binary outputs
                elif y_pred.dtype.kind in "fc":
                    y_pred = (y_pred >= 0.5).astype(int)

            if task == "regression":
                metrics = self.evaluate_regression(y_test, y_pred)
            else:
                metrics = self.evaluate_classification(y_test, y_pred)

            metrics["model"] = name
            results.append(metrics)

        # Sort by first metric: RMSE for regression (ascending), first metric for classification (descending)
        sort_col = list(results[0].keys())[0]
        ascending = task == "regression"
        return pd.DataFrame(results).sort_values(by=sort_col, ascending=ascending)
