import numpy as np
import pandas as pd
import pickle
import matrix_factorization_utilities

# Load user ratings
raw_dataset_df = pd.read_csv('test.txt', encoding='ISO-8859-1')

# Convert the running list of user ratings into a matrix
ratings_df = pd.pivot_table(raw_dataset_df, index='UserID', columns='MovieID', aggfunc=np.max)

# Apply matrix factorization to find the latent features
U, M = matrix_factorization_utilities.low_rank_matrix_factorization(ratings_df.as_matrix(),
                                                                    num_features=15,
                                                                    regularization_amount=0.1)

# Find all predicted ratings by multiplying U and M
predicted_ratings = np.matmul(U, M)

# Save features and predicted ratings to files for later use
pickle.dump(U, open("user_features.dat", "wb"))
pickle.dump(M, open("product_features.dat", "wb"))
pickle.dump(predicted_ratings, open("predicted_ratings.dat", "wb" ))