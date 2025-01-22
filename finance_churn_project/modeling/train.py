import os
import datetime
import logging
import pickle

import mlflow
import pandas as pd
from hyperopt import (
    fmin,
    space_eval,
    hp,
    tpe,
    Trials,
    STATUS_OK
)
from processing.data_processing import processing_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])

logger = logging.getLogger(__name__)


def generate_train_test_valid_data(customer_churn_df: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Series]:
    x = customer_churn_df.copy()
    x = x.drop(columns='Exited')
    y = customer_churn_df.Exited

    logger.info("Start training data split")
    x_train, x_not_train, y_train, y_not_train = train_test_split(x, y, test_size=0.3, stratify=y, random_state=0)
    x_valid, x_test, y_valid, y_test = train_test_split(x_not_train, y_not_train, test_size=0.5, stratify=y_not_train, random_state=0)

    logger.info("Start training data processing")
    x_train = processing_pipeline(x_train)
    x_valid = processing_pipeline(x_valid)
    x_test = processing_pipeline(x_test)

    training_data = {
        'x_train': x_train,
        'x_valid': x_valid,
        'x_test': x_test,
        'y_train': y_train,
        'y_valid': y_valid,
        'y_test': y_test,
    }
    return training_data


def train_model(
    training_data: dict[str, pd.DataFrame | pd.Series],
    space: dict
) -> XGBClassifier:
    x_train = training_data.get('x_train')
    y_train = training_data.get('y_train')
    x_valid = training_data.get('x_valid')
    y_valid = training_data.get('y_valid')

    x_train_sample = x_train.sample(frac=0.5, random_state=0)
    y_train_sample = y_train.loc[x_train_sample.index]
    x_valid_sample = x_valid.sample(frac=0.5, random_state=0)
    y_valid_sample = y_valid.loc[x_valid_sample.index]

    def objective(space):
        clf = XGBClassifier(
            objective=space['objective'],
            n_estimators=int(space['n_estimators']),
            max_depth=int(space['max_depth']),
            tree_method=space['tree_method'],
            gamma=space['gamma'],
            reg_alpha=space['reg_alpha'],
            min_child_weight=int(space['min_child_weight']),
            colsample_bytree=space['colsample_bytree'],
            learning_rate=space['learning_rate'],
            reg_lambda=space['reg_lambda'],
            eval_metric=space['eval_metric'],
            early_stopping_rounds=150)

        evaluation = [(x_valid_sample, y_valid_sample)]
        clf.fit(x_train_sample, y_train_sample,
                eval_set=evaluation, verbose=50)

        pred = clf.predict(x_valid)

        score = f1_score(y_valid, pred)
        # print("F1 Score:", score)
        return {'loss': 1 - score, 'status': STATUS_OK}
    trials = Trials()

    logger.info("Start hyperparameter tuning")
    hyperparams = fmin(fn=objective,
                       space=space,
                       algo=tpe.suggest,
                       max_evals=25,
                       trials=trials)

    best_hyperparams = space_eval(space, hyperparams)

    logger.info("Done. Logging parameters")
    mlflow.log_params(best_hyperparams)
    logger.info("Model fit and prediction.")
    model = XGBClassifier(**best_hyperparams)
    model.fit(x_train, y_train)
    pred = model.predict(x_valid)

    score = f1_score(y_valid, pred)
    print("F1 Score:", score)

    return model


def evaluate_model(
    model: XGBClassifier,
    training_data: dict[str, pd.DataFrame | pd.Series]
) -> float:
    logger.info("Start model evaluation")

    x_valid = training_data.get('x_valid')
    y_valid = training_data.get('y_valid')
    x_test = training_data.get('x_test')
    y_test = training_data.get('y_test')

    validation_preds = model.predict(x_valid)
    test_preds = model.predict(x_test)

    val_f1_score = f1_score(y_valid, validation_preds)
    test_f1_score = f1_score(y_test, test_preds)

    val_accuracy = accuracy_score(y_valid, validation_preds)
    test_accuracy = accuracy_score(y_test, test_preds)

    mlflow.log_metrics(
        metrics={
            'Validation F1 Score': val_f1_score,
            'Test F1 Score': test_f1_score,
            'Validation Accuracy': val_accuracy,
            'Test Accuracy': test_accuracy,
        }
    )
    return test_f1_score


def train_and_validate(
    customer_churn_df: pd.DataFrame
) -> None:

    mlflow.set_tracking_uri(os.environ.get('URL_MLFLOW', '../mlruns'))

    mlflow.set_experiment('BANK CHURN PROJECT')
    with mlflow.start_run(run_name='XGBoost Model Training'):

        training_data = generate_train_test_valid_data(customer_churn_df)

        space = {
            'objective': 'binary:logistic',
            'n_estimators': 1500,
            'colsample_bytree': hp.uniform('colsample_bytree', 0.2, 0.7),
            'gamma': hp.uniform('gamma', 0, 0.5),
            'learning_rate': hp.quniform('learning_rate', 0.001, 0.05, 0.01),
            'max_depth': hp.choice('max_depth', range(5, 15, 1)),
            'min_child_weight': hp.quniform('min_child_weight', 1, 5, 1),
            'reg_alpha': hp.uniform('reg_alpha', 0, 10),
            'reg_lambda': hp.uniform('reg_lambda', 0.1, 5),
            'eval_metric': 'error',
            'tree_method': 'hist',
            'subsample': 0.6,
            'seed': 42
        }

        tuned_model = train_model(training_data, space)
        model_f1_score = evaluate_model(tuned_model, training_data)
    model_name = f"xgb_{datetime.datetime.today().strftime(format='%Y-%m-%d')}_score_{model_f1_score:.4f}.pkl"
    model_path = os.path.join("models", model_name)
    logger.info(f"Model training pipeline finished. Storing model in {model_path}.")

    os.makedirs('models', exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(tuned_model, f)

    logger.info("Finished pipeline")
