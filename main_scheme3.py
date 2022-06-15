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
for i in rand_user_list:    
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
        rec_x1 = {}
        rec_x2 = {}
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
                
                rec_x1[genre], rec_x2[genre] = round(x_one_val,2), round(x_two_val,1)

                # print(f'genre: {genre}, genre_idx: {genre_idx_dict[genre]}, x1_val: {x_one_val}, x1_idx: {x_one_idx}, x2_val: {x_two_val}, x2_idx: {x_two_idx}')

                genre_dict[genre]['num_rated'] += 1
                genre_dict[genre]['total_rating'] += rating
        genre_x_one_vals.append(rec_x1)
        genre_x_two_vals.append(rec_x2)
        num_movies_rated +=1

# %%
temp1_df = pd.DataFrame(genre_x_one_vals,columns=genre_names)
temp2_df = pd.DataFrame(genre_x_two_vals,columns=genre_names)

thresholds1 = {}
thresholds2 = {}
for col in temp1_df:
    t1 = temp1_df[col].dropna().unique()
    t2 = temp2_df[col].dropna().unique()
    t1.sort()
    t2.sort()
    thresholds1[col] = t1
    thresholds2[col] = t2

# %%
curr_idx_x1, curr_idx_x2 = 0,0
genre_feat_idx = {}
for key in thresholds1.keys():
    g_num_feat1 = len(thresholds1[key])
    g_num_feat2 = len(thresholds2[key])
    genre_feat_idx[key] = {
        'x1': {
            'start': curr_idx_x1,
            'end': curr_idx_x1 + g_num_feat1
            },
        'x2': {
            'start': curr_idx_x2,
            'end': curr_idx_x2 + g_num_feat2
        }
        }
    curr_idx_x1 += g_num_feat1
    curr_idx_x2 += g_num_feat2
    

for key in genre_feat_idx.keys():
    genre_feat_idx[key]['x2']['start'] += curr_idx_x1
    genre_feat_idx[key]['x2']['end'] += curr_idx_x1

# %%
def get_closest_idx(arr, x):
    idx = (np.abs(arr - x)).argmin()
    return idx


# %%
ds_X = None
ds_y = None
start = time()
for i in list(range(1,10000)):    
    ratings = rating_df.loc[rating_df['userId'] == i].sort_values(by=['timestamp'])
    num_rated = len(ratings.index)
    temp_X = np.zeros([num_rated,genre_feat_idx['(no genres listed)']['x2']['end']], dtype=int)
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
                x_one_idx = get_closest_idx(thresholds1[genre], x_one_val) + genre_feat_idx[genre]['x1']['start']
                x_two_idx = get_closest_idx(thresholds2[genre], x_two_val) + genre_feat_idx[genre]['x2']['start']
                # feature_vec[x_one_idx] = 1
                # feature_vec[x_two_idx] = 1
                temp_X[index][x_one_idx:genre_feat_idx[genre]['x1']['end']] = 1
                temp_X[index][x_two_idx:genre_feat_idx[genre]['x2']['end']] = 1

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
num_clauses = 1000
threshold = 50
s_val = 15
number_of_features = 1410


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
