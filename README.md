# Adverse Drug Event (ADE) Detection: NLP Pipeline

This repository contains the engineering pipeline and midterm project for detecting Adverse Drug Events (ADEs) in unstructured patient reviews. It compares a highly-tuned traditional machine learning approach (TF-IDF + Logistic Regression) against a state-of-the-art deep learning architecture (DistilBERT Transformer).

## Repository Structure
* `/src/`: Contains the modularized Python scripts for the engineering pipeline.
* `/notebooks/`: Contains the heavily documented Jupyter Notebook (`Midterm_Capstone.ipynb`) featuring Exploratory Data Analysis, model error analysis, and visualization generation.
* `/presentation/`: Contains the midterm presentation slide deck.
* `/data/`: Directory for the raw UCI dataset (ignored via `.gitignore` to prevent large file uploads).

## Dataset Requirement
This project requires the **UCI Machine Learning Repository Drug Review Dataset**. 
Before running the pipeline, download the dataset and place the unzipped `.tsv` files (`drugsComTrain_raw.tsv` and `drugsComTest_raw.tsv`) directly into the `/data/` folder.

## How to Execute the Code (Containerized)
This project is fully containerized using Docker to ensure reproducibility and prevent dependency conflicts.

**Step 1: Build the Docker Image**
From the root directory of this repository, run:
\`\`\`bash
docker build -t ade-nlp-pipeline .
\`\`\`

**Step 2: Run the Pipeline**
Run the container, mounting your local `/data/` directory so the container can access the TSV files:
\`\`\`bash
docker run -v $(pwd)/data:/app/data ade-nlp-pipeline
\`\`\`

## Exception and Bug Handling
The codebase features robust `try-except` wrappers around data ingestion, matrix transformations, and model training loops. Missing data files or missing columns will cleanly terminate the pipeline with instructional terminal outputs rather than throwing raw stack traces. 

## Code Documentation
All functions within the `/src/` directory utilize Google-style docstrings denoting inputs, outputs, and logical flow. Inline comments are provided to explain *why* specific hyperparameters (such as class weights or n-gram ranges) were chosen, prioritizing readability and architectural clarity.