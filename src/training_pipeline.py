import os
from modeling.train import train_and_validate
from processing.data_loader import data_load_pipeline


def main():
    ROOT_PATH = os.environ['ROOT_PATH']
    training_path = os.path.join(ROOT_PATH, 'data/training/raw/')

    customer_churn_df = data_load_pipeline(folder_path=training_path)
    train_and_validate(customer_churn_df)  # type: ignore


if __name__ == "__main__":
    main()
