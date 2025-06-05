
import pandas as pd
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.compose import ColumnTransformer
from tabulate import tabulate
from imblearn.under_sampling import RandomUnderSampler
warnings.filterwarnings("ignore")


def process_data(X, fit=False, preprocessor=None):
    """
        Applies preprocessing to the dataset. If `fit=True`, it fits the transformations on the data.
        
        Parameters:
            X (pd.DataFrame): The dataset to be processed.
            fit (bool): If True, fits the preprocessing pipeline. If False, applies existing transformations.
            preprocessor (ColumnTransformer or None): The preprocessor to be used. If None, a new one is created.
        
        Returns:
            Transformed X as DataFrame and the fitted preprocessor if fit=True.
    """
    if fit:
        cat_ft = X.select_dtypes(include=['object']).columns.tolist()
        num_ft = X.select_dtypes(exclude=['object']).columns.tolist()

        preprocessor = ColumnTransformer([
            ('encoder', OrdinalEncoder(
                handle_unknown='use_encoded_value', unknown_value=-1), cat_ft),
            ('scaler', StandardScaler(), num_ft)
        ])

        X_transformed = preprocessor.fit_transform(X)
    else:
        X_transformed = preprocessor.transform(X)

    encoded_columns = preprocessor.get_feature_names_out()
    X_transformed = pd.DataFrame(X_transformed, columns=encoded_columns)

    return X_transformed, preprocessor


def preprocess_and_resample(df, target_col, resampling_method=None):
    """
    Preprocesses the dataset and applies undersampling or SMOTE-NC if specified.

    Parameters:
        df (pd.DataFrame): The dataset containing features and target variable.
        target_col (str): The name of the target column.
        resampling_method (str, optional): 'undersample' for undersampling or 'smote' for SMOTE-NC.

    Returns:
        X_train (pd.DataFrame): Preprocessed and resampled training features.
        X_test (pd.DataFrame): Preprocessed test features.
        y_train (pd.Series): Training labels (after resampling).
        y_test (pd.Series): Test labels.
    """

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.30, random_state=42
    )

    X_train, preprocessor = process_data(X_train, fit=True)
    X_test, _ = process_data(X_test, fit=False, preprocessor=preprocessor)

    if resampling_method == 'undersample':
        undersampler = RandomUnderSampler(random_state=42)
        X_train, y_train = undersampler.fit_resample(X_train, y_train)
    elif resampling_method == 'smote':
        categorical_indices = [i for i, col in enumerate(
            X_train.columns) if col.startswith('encoder')]

        smote_nc = SMOTENC(
            categorical_features=categorical_indices, random_state=42)
        X_train, y_train = smote_nc.fit_resample(X_train, y_train)

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    return X_train, X_test, y_train, y_test


def hyperparameter_tuning(X_train, y_train, cv, random_seed=42):
    random_seed = 42
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

    models = {

        "LogisticRegression": (LogisticRegression(random_state=random_seed, max_iter=1000), {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'solver': ['liblinear', 'saga']
        }),


        "RandomForestClassifier": (RandomForestClassifier(random_state=random_seed), {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20],
            'min_samples_split': [2, 5, 10]
        }),


        "DecisionTreeClassifier": (DecisionTreeClassifier(random_state=random_seed), {
            'max_depth': [3, 5, 10],
            'criterion': ['gini', 'entropy']
        }),


        "KNeighborsClassifier": (KNeighborsClassifier(), {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        }),

        "SVC": (SVC(random_state=random_seed), {
            'C': [0.1, 1, 5, 10, 100, 1000],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto'],
            'degree': [2, 3]
        }),

        "GradientBoostingClassifier": (GradientBoostingClassifier(random_state=random_seed), {
            'n_estimators': [50, 100, 200, 300, 400],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 10]
        }),

        "XGBoost": (xgb.XGBClassifier(random_state=random_seed, use_label_encoder=False, eval_metric='logloss'), {
            'n_estimators': [50, 100, 200, 300, 400],
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'max_depth': [3, 5, 7, 9, 10],
            'min_split_loss':  [0, 25, 50, 100, 500, 100],
            'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
            'gamma':  [0, 2, 5, 20, 50, 100]
        }),

        "CatBoost": (cb.CatBoostClassifier(verbose=0, random_seed=random_seed), {
            'iterations': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'depth': [3, 5, 10]
        }),

        "LightGBM": (lgb.LGBMClassifier(random_state=random_seed), {
            'max_depth': [3, 5, 7, 9, 10],
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'min_data_in_leaf': [10, 20, 50],
            'num_leaves': [10, 20, 30, 50, 100]
            # "verbose": -1
        })

    }

    best_models = []

    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'average_precision']

    for model_name, (model, param_grid) in models.items():
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring=scoring_metrics, refit='recall', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        results_grid_search = grid_search.cv_results_

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        best_index = grid_search.best_index_
        results = grid_search.cv_results_

        best_metrics = {
            'accuracy': results[f'mean_test_accuracy'][best_index],
            'precision': results[f'mean_test_precision'][best_index],
            'recall': results[f'mean_test_recall'][best_index],
            'f1': results[f'mean_test_f1'][best_index],
            'auc_pr': results_grid_search[f'mean_test_average_precision'][best_index]
        }

        best_models.append({
            "Best Model": [model_name],
            "Best Score": [best_score],
            "Best Index": [best_index],
            "Best Parameters": [str(best_params)],
            **best_metrics
        })

    best_models_df = pd.DataFrame(best_models)
    best_models_df = best_models_df.sort_values(by="recall", ascending=False)
    print(tabulate(best_models_df, headers="keys", tablefmt="pretty"))
    return None


def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1_metric = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    auc_pr = average_precision_score(y_true, y_pred)
    
    validation_df = pd.DataFrame([{
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_metric,
        "ROC AUC": roc_auc,
        "AUC PR": auc_pr
    }])

def highlight_scores(val):
    if val >= 0.8:
        color = '#084594' 
    elif val >= 0.5:
        color = '#4292c6' 
    else:
        color = '#deebf7' 
    return f'background-color: {color}'
    
    styled_df = validation_df.style.applymap(highlight_scores)

    return styled_df