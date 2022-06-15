# %%
rand_user_list = random.sample(range(1,162541), 40)
# %%
import numpy as np
import pandas as pd
from time import time 
import random

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

def get_feature_indexes(x, genre_index, is_x_one=False):
    num_x1_feats = 11
    num_x2_feats = 9
    feat_block_start = genre_index * (num_x1_feats + num_x2_feats)
    if is_x_one:
        x = x * 10
        start_index = round(x) + feat_block_start
        end_index = feat_block_start + num_x1_feats
    else:
        start_index = round(x * 2) - 2
        start_index = start_index + feat_block_start + num_x1_feats
        end_index = feat_block_start + num_x1_feats + num_x2_feats
    return start_index, end_index

def binarize_features(x_arr, y_arr, g_dict, movie_genre_sets, idx_dict, total_movies, binary_assignment):
    for idx, g_set in enumerate(movie_genre_sets):
        for genre in g_set:
            g_num_rated = g_dict[genre]['num_rated']
            x_one_val = g_num_rated / total_movies

            if g_num_rated != 0:
                x_two_val = g_dict[genre]['total_rating'] / g_num_rated
            else:
                x_two_val = 0
            
            x1_start, x1_end = get_feature_indexes(x_one_val, idx_dict[genre], is_x_one=True)
            x2_start, x2_end = get_feature_indexes(x_two_val, idx_dict[genre])
            x_arr[idx, x1_start:x1_end] = 1
            x_arr[idx, x2_start:x2_end] = 1
        y_arr[idx] = binary_assignment


def preprocess(is_test_data=False):
    if is_test_data:
        user_idx_list = list(range(200,220))
    else:
        user_idx_list = list(range(1,200))

    genre_names = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western', '(no genres listed)']
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
    genre_x_one_vals = [] #pd.DataFrame(columns=genre_names)
    genre_x_two_vals = [] #pd.DataFrame(columns=genre_names)
    start = time()
    # for i in rand_user_list:   
    for i in user_idx_list:     
        ratings = rating_df.loc[rating_df['userId'] == i].sort_values(by=['timestamp'])
        num_rated = len(ratings.index)
        # temp_X = np.zeros([num_rated,380], dtype=int)
        # temp_y = np.zeros([num_rated,], dtype=int)

        genre_dict = {genre_list[i]: {'total_rating': 0, 'num_rated': 0} for i in range(len(genre_list))}

        user_df = ratings.merge(movie_df, on='movieId', how='left')

        hi_movie_genre_sets = []
        lo_movie_genre_sets = []
        for index, row in user_df.iterrows():
            genres = row['genres'].split('|')
            rating = row['rating']
            g_list = []
            for genre in genres:
                if genre != 'IMAX':
                    genre_dict[genre]['num_rated'] += 1
                    genre_dict[genre]['total_rating'] += rating
                    g_list.append(genre)

            if rating >= 5:
                hi_movie_genre_sets.append(g_list)
            elif rating <= 2:
                lo_movie_genre_sets.append(g_list)

        hi_X = np.zeros([len(hi_movie_genre_sets), 380])
        hi_y = np.zeros([len(hi_movie_genre_sets),])
        binarize_features(hi_X,hi_y,genre_dict,hi_movie_genre_sets,genre_idx_dict,num_rated,1)

        lo_X = np.zeros([len(lo_movie_genre_sets), 380])
        lo_y = np.zeros([len(lo_movie_genre_sets),])
        binarize_features(lo_X,lo_y,genre_dict,lo_movie_genre_sets,genre_idx_dict,num_rated,0)

        temp_X = np.vstack((hi_X, lo_X))
        temp_y = np.concatenate((hi_y, lo_y))
        if ds_X is None:
            ds_X = temp_X
        else:
            ds_X = np.vstack((ds_X, temp_X))
        if ds_y is None:
            ds_y = temp_y
        else:
            ds_y = np.concatenate((ds_y, temp_y))
    return ds_X, ds_y


from sklearn.utils import shuffle

X_train, ytrain = preprocess()
X_train, ytrain = shuffle(X_train, ytrain, random_state=42)

X_test, ytest = preprocess(is_test_data=True)
X_test, ytest = shuffle(X_test, ytest, random_state=42)

# %%
from time import time 
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
from sklearn import metrics
from sklearn.model_selection import train_test_split

# X = np.asarray(ds_X)
# y = np.asarray(ds_y)

# np.random.shuffle(X)
# np.random.shuffle(y)
num_clauses = 200
threshold = 50
s_val = 5
number_of_features = 380


# X_train, X_test, ytrain, ytest = train_test_split(X, y, test_size=.3, random_state=42)
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
