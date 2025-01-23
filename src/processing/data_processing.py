import pandas as pd


def processing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy = df_copy.iloc[:, 3:]

    df_copy['Gender'].replace({'Male': 0, 'Female': 1}, inplace=True)
    geography_dummies = pd.get_dummies(df_copy['Geography']).astype(int)
    df_copy = pd.concat([df_copy, geography_dummies], axis=1).drop('Geography', axis=1)
    df_copy['Products_Tenure_relation'] = df_copy['NumOfProducts'] * df_copy['Tenure']

    return df_copy
