import os
import datetime
import glob
import pickle

import pandas as pd
from xgboost import XGBClassifier
from processing.data_processing import processing_pipeline
from utils.general_utils import get_best_model_path


def generate_predictions(
    model: XGBClassifier,
    prediction_data: pd.DataFrame,
    row_number_feature: pd.Series
) -> pd.DataFrame:
    model_preds = model.predict(prediction_data)
    predictions = pd.DataFrame(row_number_feature)
    predictions['predictedValues'] = model_preds

    return predictions


def make_inference(customer_data: pd.DataFrame) -> None:

    file_list = glob.glob(os.path.join('models', '*'))
    best_model_path = get_best_model_path(file_list)
    with open(best_model_path, 'rb') as f:
        xgb_model = pickle.load(f)

    customer_data_processed = processing_pipeline(customer_data)

    predictions = generate_predictions(xgb_model, customer_data_processed, customer_data['RowNumber'])

    os.makedirs('predictions', exist_ok=True)
    predictions_name = f"preds_{datetime.datetime.today().strftime(format='%Y-%m-%d')}.csv"
    predictions_path = os.path.join('predictions', predictions_name)
    predictions.to_csv(predictions_path, index=False)
