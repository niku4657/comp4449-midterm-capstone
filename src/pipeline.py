"""
Main Engineering Pipeline for ADE Detection.
Executes data loading, model training (TF-IDF & DistilBERT), and evaluation.
"""
import sys
from data_prep import load_and_prepare_data
from models import train_tfidf_improved, train_transformer_weighted

def main():
    print("Starting ADE Detection Pipeline...")
    
    # Paths to data (expecting user to have downloaded them to the /data/ folder)
    train_path = 'data/drugsComTrain_raw.tsv'
    test_path = 'data/drugsComTest_raw.tsv'
    
    # 1. Data Preparation (With Exception Handling)
    try:
        train_df, test_df = load_and_prepare_data(train_path, test_path)
        if train_df is None or test_df is None:
            raise ValueError("Dataframes returned None. Check file paths and data integrity.")
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Data files not found at {train_path} or {test_path}.")
        print("Please download the UCI Drug Review dataset and place the TSV files in the /data/ directory.")
        sys.exit(1)
    except Exception as e:
        print(f"CRITICAL ERROR during data pipeline: {e}")
        sys.exit(1)

    # 2. Train Traditional Baseline
    try:
        tfidf_results, tfidf_preds = train_tfidf_improved(train_df, test_df)
    except Exception as e:
        print(f"ERROR training TF-IDF model: {e}")

    # 3. Train Transformer
    try:
        trans_results, trans_preds, trans_history = train_transformer_weighted(train_df, test_df)
    except Exception as e:
        print(f"ERROR training DistilBERT model: {e}")
        print("Ensure you have a GPU enabled or PyTorch configured correctly.")

    print("\nPipeline execution complete.")

if __name__ == "__main__":
    main()