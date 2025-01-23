SRC_DIR = src
TRAIN_DIR= data/training/raw 
INFERENCE_DIR = data/inference/raw

create_data_folders:
	@echo "Creating train folder"
	mkdir -p $(TRAIN_DIR)

	@echo "Creating inference folder"
	mkdir -p $(INFERENCE_DIR)
full_pipeline:
	@echo "Running Train Pipeline..."
	cd $(SRC_DIR) && python training_pipeline.py
	@echo "Running Inference Pipeline..."
	cd $(SRC_DIR) && python inference_pipeline.py