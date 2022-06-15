#%%
# import numpy as np
# import pandas as pd


# link_df = pd.read_csv('ml-latest-small/links.csv')
# movie_df = pd.read_csv('ml-latest-small/movies.csv')
# rating_df = pd.read_csv('ml-latest-small/ratings.csv')
# # %%
# time_sorted_rating_df = rating_df.loc[rating_df['userId'] == 1].sort_values(by=['timestamp'])
# time_sorted_rating_df = time_sorted_rating_df.reset_index(drop=True)

# column_names = ['userId','totalMoviesRated', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western', '(no genres listed)']
# user_movie_genre_totals = pd.DataFrame(columns=column_names)
# empty_rec = [0] * len(column_names)
# empty_rec[0] = 1
# user_movie_genre_totals.loc[len(user_movie_genre_totals.index)] = empty_rec
# x_one = 
# for index, row in time_sorted_rating_df.iterrows():
#     genres = 
#     print(index)
#     print(row)
#     break
# %%
# genres = ['Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western', '(no genres listed)']

# import numpy as np
# import pandas as pd


# link_df = pd.read_csv('ml-latest-small/links.csv')
# movie_df = pd.read_csv('ml-latest-small/movies.csv')
# rating_df = pd.read_csv('ml-latest-small/ratings.csv')

# rating_df['timestamp'] = pd.to_datetime(rating_df['timestamp'], unit='s')

# ratings = rating_df.loc[rating_df['userId'] == 1]
# genres_unique = pd.DataFrame(movie_df.genres.str.split('|').tolist()).stack().unique()
# genres_unique = pd.DataFrame(genres_unique, columns=['genre'])

# movie_df = movie_df.join(movie_df.genres.str.get_dummies().astype(bool).astype(int))
# # movie_df.drop('genres', inplace=True, axis=1)

# user_df = (ratings.set_index('movieId').join(movie_df.set_index('movieId'), how='left'))

# num_recs = len(user_df)
# for genre in genres_unique.genre:
#     user_df[genre]['num_rated'] = user_df.groupby('userId')[genre].rolling(min_periods=1,window=num_recs).sum().values
# user_df['total_movies_watched'] = user_df.groupby('userId')['userId'].rolling(min_periods=1, window=num_recs).count().values






# %%
import numpy as np
import pandas as pd

def binarize_x(x, increments):
    binarized = []
    for i in increments:
        if x <= i:
            binarized.append(0)
        else:
            binarized.append(1)

    return binarized

def binarize_alt(x, is_x_one=False):
    if is_x_one:
        x = x * 10
        binarized = [0] * 11
        index = round(x)
    else:
        binarized = [0] * 9
        index = round(x * 2) - 2
    binarized[index] = 1
    return binarized
    
    

# ds_name = 'ml-latest-small/movies.csv'
ds_name = 'ml-25m'
# link_df = pd.read_csv('ml-latest-small/links.csv')
movie_df = pd.read_csv(ds_name + '/movies.csv')
rating_df = pd.read_csv(ds_name + '/ratings.csv')

rating_df['timestamp'] = pd.to_datetime(rating_df['timestamp'], unit='s')


ds_X = []
ds_y = []
for i in list(range(1,10000)):
    ratings = rating_df.loc[rating_df['userId'] == i].sort_values(by=['timestamp'])
    genres_unique = pd.DataFrame(movie_df.genres.str.split('|').tolist()).stack().unique()
    genres_unique = pd.DataFrame(genres_unique, columns=['genre'])

    genre_list = genres_unique.genre
    genre_dict = {genre_list[i]: {'total_rating': 0, 'num_rated': 0} for i in range(len(genre_list))}
    num_movies_rated = 0

    user_df = ratings.merge(movie_df, on='movieId', how='left')

    for index, row in user_df.iterrows():
        
        genres = row['genres'].split('|')
        rating = row['rating']
        for genre in genres:
            
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
            bin_x_one = binarize_alt(x_one_val, is_x_one=True)
            bin_x_two = binarize_alt(x_two_val)
            ds_X.append(bin_x_one + bin_x_two)
            if rating >= 5:
                ds_y.append(1)
            # elif rating >= 4:
            #     ds_y.append(4)
            # elif rating >= 3:
            #     ds_y.append(3)
            # elif rating >= 2:
            #     ds_y.append(2)
            # elif rating >= 1:
            #     ds_y.append(1)
            else:
                ds_y.append(0)

            genre_dict[genre]['num_rated'] += 1
            genre_dict[genre]['total_rating'] += rating
        num_movies_rated += 1



# %%
X = np.asarray(ds_X)
y = np.asarray(ds_y)

# %%
from time import time 
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
from sklearn import metrics
from sklearn.model_selection import train_test_split

# X = np.asarray(ds_X)
# y = np.asarray(ds_y)

# np.random.shuffle(X)
# np.random.shuffle(y)
num_clauses = 20
threshold = 20
s_val = 5
number_of_features = 20


X_train, X_test, ytrain, ytest = train_test_split(X, y, test_size=.3, random_state=42)

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
