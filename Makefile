SRC_DIR = src

full_pipeline:
	@echo "Running Train Pipeline..."
	cd $(SRC_DIR) && python training_pipeline.py
	@echo "Running Inference Pipeline..."
	cd $(SRC_DIR) && python inference_pipeline.py