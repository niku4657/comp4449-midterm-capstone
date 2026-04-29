def load_and_prepare_data(train_df, test_df):
    """
    Loads pre-split data, concatenates separated review columns,
    and creates a binary sentiment label.
    """
    try:
        # 1. Drop rows missing the target variable
        train_df.dropna(subset=['rating'], inplace=True)
        test_df.dropna(subset=['rating'], inplace=True)

        # 2. Concatenate the three review columns into a single text feature
        # We use fillna('') to prevent NaNs from turning the entire concatenated string into NaN
        train_df['review'] = "BENEFITS: " + train_df['benefitsReview'].fillna('None') + \
               " SIDE_EFFECTS: " + train_df['sideEffectsReview'].fillna('None') + \
               " COMMENTS: " + train_df['commentsReview'].fillna('None')
        test_df['review'] = "BENEFITS: " + test_df['benefitsReview'].fillna('None') + \
               " SIDE_EFFECTS: " + test_df['sideEffectsReview'].fillna('None') + \
               " COMMENTS: " + test_df['commentsReview'].fillna('None')

        # Clean up extra spaces created by empty columns
        train_df['review'] = train_df['review'].str.replace(r'\s+', ' ', regex=True).str.strip()
        test_df['review'] = test_df['review'].str.replace(r'\s+', ' ', regex=True).str.strip()

        # 3. Remove rows where the combined review is entirely empty
        train_df = train_df[train_df['review'] != '']
        test_df = test_df[test_df['review'] != '']

        # 4. Create binary label: 1 if rating > 5 (Positive), 0 otherwise (Negative/ADE)
        train_df['label'] = (train_df['rating'] > 5).astype(int)
        test_df['label'] = (test_df['rating'] > 5).astype(int)

        train_df.drop(columns=['benefitsReview', 'sideEffectsReview', 'commentsReview', 'Unnamed: 0', 'urlDrugName', 'rating'], inplace=True)
        test_df.drop(columns=['benefitsReview', 'sideEffectsReview', 'commentsReview', 'Unnamed: 0', 'urlDrugName', 'rating'], inplace=True)

        print(f"Data loaded and cleaned! Train size: {len(train_df)} | Test size: {len(test_df)}")
        return train_df, test_df
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the files are uploaded to the Colab environment.")
        return None, None
    except Exception as e:
        print(f"Unexpected error in data preparation: {e}")
        return None, None