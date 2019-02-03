import numpy as np
import pandas as pd
import matrix_factorization_utilities

# Load user ratings
raw_dataset_df = pd.read_csv('test.txt')

# Load movie titles
movies_df = pd.read_csv('movies.txt', encoding="ISO-8859-1", index_col='MovieID')

# Convert the running list of user ratings into a matrix
ratings_df = pd.pivot_table(raw_dataset_df, index='UserID',
                            columns='MovieID',
                            aggfunc=np.max)

# Apply matrix factorization to find the latent features
U, M = matrix_factorization_utilities.low_rank_matrix_factorization(ratings_df.as_matrix(),
                                                                    num_features=15,
                                                                    regularization_amount=0.1)

# Find all predicted ratings by multiplying U and M matrices
predicted_ratings = np.matmul(U, M)

print("Enter a user_id to get recommendations (Between 1 and 100):")
user_id_to_search = int(input())

print("Movies previously reviewed by user_id {}:".format(user_id_to_search))

reviewed_movies_df = raw_dataset_df[raw_dataset_df['UserID'] == user_id_to_search]
reviewed_movies_df = reviewed_movies_df.join(movies_df, on='MovieID')

print(reviewed_movies_df[['Title', 'Genres', 'Rating']])

input("Press enter to continue.")

print("Movies we will recommend:")

user_ratings = predicted_ratings[user_id_to_search - 1]
sLength = len(user_ratings)
movies_df['Rating'] = pd.Series(np.random.randn(sLength))

already_reviewed = reviewed_movies_df['MovieID']
recommended_df = movies_df[movies_df.index.isin(already_reviewed) == False]
recommended_df = recommended_df.sort_values(by=['Rating'], ascending=False)

print(recommended_df[['Title', 'Genres', 'Rating']].head(15))

