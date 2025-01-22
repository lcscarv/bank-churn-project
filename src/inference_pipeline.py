import os
from inference.predict import make_inference
from processing.data_loader import data_load_pipeline


def main():
    ROOT_PATH = os.environ['ROOT_PATH']
    inference_path = os.path.join(ROOT_PATH, 'data/inference/raw/')

    customer_data = data_load_pipeline(folder_path=inference_path)

    make_inference(customer_data)  # type: ignore


if __name__ == "__main__":
    main()
