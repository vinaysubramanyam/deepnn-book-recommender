import pandas as pd
from sklearn import dummy, metrics, cross_validation, ensemble
import numpy as np
import keras.models as kmodels
import keras.layers as klayers
import keras.backend as K
import keras
from keras.utils.visualize_util import plot

    

df = pd.read_csv('./BX-CSV-Dump/BX-Book-Ratings.csv', dtype={'user':np.int32, 'isbn':str, 'rating':str})
df['rating'].replace(['NA'],'0')
df['rating'].fillna(0,inplace=True)
# df['rating'].astype(int)


user = np.vstack(df['user'])
isbn = np.vstack(df['isbn'])
rating = np.vstack(df['rating'])
rating = np.asarray(rating,dtype=int)
# print rating[1][0]

df.user=df.user.astype('category')
df.isbn=df.isbn.astype('category')
userid=df.user.cat.codes.values
bookid=df.isbn.cat.codes.values

y = np.zeros((rating.shape[0], 11))

n_movies=isbn.shape[0]
n_users=user.shape[0]
# for _ in range(0,rating.shape[0]):
# 	y[_,rating[_][0]]=1
i=0
for row in y:
	y[i,rating[i][0]]=1
	# print y[i,rating[i][0]]
	i=i+1
# print y[1,5]
# y[np.arange(rating.shape[0]), rating] = 1

# First, we take the movie and vectorize it.
# The embedding layer is normally used for sequences (think, sequences of words)
# so we need to flatten it out.
# The dropout layer is also important in preventing overfitting
movie_input = keras.layers.Input(shape=[1])
movie_vec = keras.layers.Flatten()(keras.layers.Embedding(n_movies + 1, 32)(movie_input))
movie_vec = keras.layers.Dropout(0.5)(movie_vec)

# Same thing for the users
user_input = keras.layers.Input(shape=[1])
user_vec = keras.layers.Flatten()(keras.layers.Embedding(n_users + 1, 32)(user_input))
user_vec = keras.layers.Dropout(0.5)(user_vec)

# Next, we join them all together and put them
# through a pretty standard deep learning architecture
input_vecs = keras.layers.merge([movie_vec, user_vec], mode='concat')
nn = keras.layers.Dropout(0.5)(keras.layers.Dense(128, activation='relu')(input_vecs))
nn = keras.layers.normalization.BatchNormalization()(nn)
nn = keras.layers.Dropout(0.5)(keras.layers.Dense(128, activation='relu')(nn))
nn = keras.layers.normalization.BatchNormalization()(nn)
nn = keras.layers.Dense(128, activation='relu')(nn)

# Finally, we pull out the result!
result = keras.layers.Dense(11, activation='softmax')(nn)

# And make a model from it that we can actually run.
model = kmodels.Model([movie_input, user_input], result)
model.compile('adam', 'categorical_crossentropy')

plot(model,to_file='out.png')
# If we wanted to inspect part of the model, for example, to look
# at the movie vectors, here's how to do it. You don't need to 
# compile these models unless you're going to train them.
final_layer = kmodels.Model([movie_input, user_input], nn)
movie_vec = kmodels.Model(movie_input, movie_vec)


# Split the data into train and test sets...
a_movieid, b_movieid, a_userid, b_userid, a_y, b_y = cross_validation.train_test_split(bookid, userid, y)


# And of _course_ we need to make sure we're improving, so we find the MAE before
# training at all.
metrics.mean_absolute_error(np.argmax(b_y, 1)+1, np.argmax(model.predict([b_movieid, b_userid]), 1)+1)


try:
    history = model.fit([a_movieid, a_userid], a_y, 
                         nb_epoch=20, 
                         validation_data=([b_movieid, b_userid], b_y))
    plot(history.history['loss'])
    plot(history.history['val_loss'])
except KeyboardInterrupt:
    pass




# This is the number that matters. It's the held out 
# test set score. Note the + 1, because np.argmax will
# go from 0 to 4, while our ratings go 1 to 5.
ans=metrics.mean_absolute_error(
    np.argmax(b_y, 1)+1, 
    np.argmax(model.predict([b_movieid, b_userid]), 1)+1)

print "Testing accuracy=",ans 



