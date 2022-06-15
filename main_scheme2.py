
# %%
import numpy as np
import pandas as pd
from time import time 


def get_feature_index(x, genre_index, is_x_one=False):
    num_x1_feats = 11
    num_x2_feats = 9
    feat_block_start = genre_index * (num_x1_feats + num_x2_feats)
    if is_x_one:
        x = x * 10
        index = round(x) + feat_block_start
    else:
        index = round(x * 2) - 2
        index = index + feat_block_start + num_x1_feats
    return index
    
genre_idx_dict = {
    'Action': 0, 
    'Adventure': 1, 
    'Animation': 2, 
    'Children': 3, 
    'Comedy': 4, 
    'Crime': 5, 
    'Documentary': 6, 
    'Drama': 7, 
    'Fantasy': 8, 
    'Film-Noir': 9, 
    'Horror': 10, 
    'Musical': 11, 
    'Mystery': 12, 
    'Romance': 13, 
    'Sci-Fi': 14, 
    'Thriller': 15, 
    'War': 16, 
    'Western': 17, 
    '(no genres listed)': 18 
    }
    

# ds_name = 'ml-latest-small/movies.csv'
ds_name = 'ml-25m'
# link_df = pd.read_csv('ml-latest-small/links.csv')
movie_df = pd.read_csv(ds_name + '/movies.csv')
rating_df = pd.read_csv(ds_name + '/ratings.csv')

rating_df['timestamp'] = pd.to_datetime(rating_df['timestamp'], unit='s')
genres_unique = pd.DataFrame(movie_df.genres.str.split('|').tolist()).stack().unique()
genres_unique = pd.DataFrame(genres_unique, columns=['genre'])
genre_list = genres_unique.genre

ds_X = None
ds_y = None
start = time()
for i in list(range(1,10000)):    
    ratings = rating_df.loc[rating_df['userId'] == i].sort_values(by=['timestamp'])
    num_rated = len(ratings.index)
    temp_X = np.zeros([num_rated,380], dtype=int)
    temp_y = np.zeros([num_rated,], dtype=int)

    genre_dict = {genre_list[i]: {'total_rating': 0, 'num_rated': 0} for i in range(len(genre_list))}
    num_movies_rated = 0

    user_df = ratings.merge(movie_df, on='movieId', how='left')

    for index, row in user_df.iterrows():
        # feature_vec = [0] * 20 * 19
        genres = row['genres'].split('|')
        rating = row['rating']
        for genre in genres:
            if genre != 'IMAX':
                g_num_rated = genre_dict[genre]['num_rated']

                if num_movies_rated != 0:
                    x_one_val = g_num_rated / num_movies_rated
                else:
                    x_one_val = 0
                
                if g_num_rated != 0:
                    x_two_val = genre_dict[genre]['total_rating'] / g_num_rated
                else:
                    x_two_val = 0
                
                # bin_x_one = binarize_x(x_one_val, [.5,.6,.7,.8,.9,1.0])
                # bin_x_two = binarize_x(x_two_val,[1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0])
                x_one_idx = get_feature_index(x_one_val, genre_idx_dict[genre], is_x_one=True)
                x_two_idx = get_feature_index(x_two_val, genre_idx_dict[genre])
                # feature_vec[x_one_idx] = 1
                # feature_vec[x_two_idx] = 1
                temp_X[index][x_one_idx] = 1
                temp_X[index][x_two_idx] = 1

                # print(f'genre: {genre}, genre_idx: {genre_idx_dict[genre]}, x1_val: {x_one_val}, x1_idx: {x_one_idx}, x2_val: {x_two_val}, x2_idx: {x_two_idx}')

                genre_dict[genre]['num_rated'] += 1
                genre_dict[genre]['total_rating'] += rating
        if rating >= 5:
            # ds_y.append(1)
            temp_y[index] = 1
        # else:
        #     # ds_y.append(0)
        # ds_X.append(feature_vec)
        num_movies_rated += 1
        # print(row)
        # print(feature_vec)
    if ds_X is None:
        ds_X = temp_X
    else:
        ds_X = np.vstack((ds_X, temp_X))
    if ds_y is None:
        ds_y = temp_y
    else:
        ds_y = np.concatenate((ds_y, temp_y))
    
    if i % 100 == 0:
        stop = time()
        print(f'time to process 100 users: {stop - start}')
        start = stop
        


# %%
X = ds_X
y = ds_y

# %%
from time import time 
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
from sklearn import metrics
from sklearn.model_selection import train_test_split

# X = np.asarray(ds_X)
# y = np.asarray(ds_y)

# np.random.shuffle(X)
# np.random.shuffle(y)
num_clauses = 400
threshold = 50
s_val = 5
number_of_features = 380


X_train, X_test, ytrain, ytest = train_test_split(X, y, test_size=.3, random_state=42)
print(f'parameters => number of clauses: {num_clauses}, T: {threshold}, s: {s_val}')
tm1 = MultiClassTsetlinMachine(num_clauses, threshold, s_val)
max = 0
for i in range(500):
	start_training = time()
	tm1.fit(X_train, ytrain, epochs=1, incremental=True)
	stop_training = time()

	start_testing = time()
	result2 = 100*(tm1.predict(X_train) == ytrain).mean()
	result1 = 100*(tm1.predict(X_test) == ytest).mean()
	y_pred = tm1.predict(X_test)
	f1= metrics.f1_score(ytest, y_pred, average='macro')
	if result1>max:
		max = result1
		pred = tm1.predict(X_test)
		ta_state = tm1.get_state()
	stop_testing = time()
	print("#%d AccuracyTrain: %.2f%% AccuracyTest: %.2f%% F1-Score: %.2f%%  Training: %.2fs Testing: %.2fs" % (i+1, result2, result1, f1*100, stop_training-start_training, stop_testing-start_testing))

# %%
# num_clauses = 10
# number_of_features = 15
print("\nClass 0 Positive Clauses:\n")
for j in range(0, num_clauses, 2):
	print("Clause #%d: " % (j), end=' ')
	l = []
	for k in range(number_of_features*2):
		if tm1.ta_action(0, j, k) == 1:
			if k < number_of_features:
				l.append(" x%d" % (k))
			else:
				l.append("¬x%d" % (k-number_of_features))
	print(" ∧ ".join(l))
# %%
# number_of_features = 15
print("\nClass 0 Negative Clauses:\n")
for j in range(1, num_clauses, 2):
	print("Clause #%d: " % (j), end=' ')
	l = []
	for k in range(number_of_features*2):
		if tm1.ta_action(0, j, k) == 1:
			if k < number_of_features:
				l.append(" x%d" % (k))
			else:
				l.append("¬x%d" % (k-number_of_features))
	print(" ∧ ".join(l))
# %%
# number_of_features = 15
print("\nClass 1 Positive Clauses:\n")
for j in range(0, num_clauses, 2):
	print("Clause #%d: " % (j), end=' ')
	l = []
	for k in range(number_of_features*2):
		if tm1.ta_action(1, j, k) == 1:
			if k < number_of_features:
				l.append(" x%d" % (k))
			else:
				l.append("¬x%d" % (k-number_of_features))
	print(" ∧ ".join(l))
# %%
print("\nClass 1 Negative Clauses:\n")
for j in range(1, num_clauses, 2):
	print("Clause #%d: " % (j), end=' ')
	l = []
	for k in range(number_of_features*2):
		if tm1.ta_action(1, j, k) == 1:
			if k < number_of_features:
				l.append(" x%d" % (k))
			else:
				l.append("¬x%d" % (k-number_of_features))
	print(" ∧ ".join(l))
# %%
