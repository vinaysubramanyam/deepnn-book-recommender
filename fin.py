import pandas as pd
from sklearn import dummy, metrics, cross_validation, ensemble
import numpy as np
import keras.models as kmodels
import keras.layers as klayers
import keras.backend as K
import keras
from keras.utils.visualize_util import plot
from keras.models import load_model
    

df = pd.read_csv('BX-Book-Ratings.csv', dtype={'user':np.int32, 'isbn':str, 'rating':str})
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

n_books=isbn.shape[0]
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

book_input = keras.layers.Input(shape=[1])
book_vec = keras.layers.Flatten()(keras.layers.Embedding(n_books + 1, 32)(book_input))
book_vec = keras.layers.Dropout(0.5)(book_vec)
user_input = keras.layers.Input(shape=[1])
user_vec = keras.layers.Flatten()(keras.layers.Embedding(n_users + 1, 32)(user_input))
user_vec = keras.layers.Dropout(0.5)(user_vec)
input_vecs = keras.layers.merge([book_vec, user_vec], mode='concat')
nn = keras.layers.Dropout(0.5)(keras.layers.Dense(128, activation='relu')(input_vecs))
nn = keras.layers.normalization.BatchNormalization()(nn)
nn = keras.layers.Dropout(0.5)(keras.layers.Dense(128, activation='relu')(nn))
nn = keras.layers.normalization.BatchNormalization()(nn)
nn = keras.layers.Dense(128, activation='relu')(nn)


result = keras.layers.Dense(11, activation='softmax')(nn)


model = kmodels.Model([book_input, user_input], result)
model.compile('adam', 'categorical_crossentropy')

model.save('book_model.h5')

plot(model,to_file='out.png')

final_layer = kmodels.Model([book_input, user_input], nn)
book_vec = kmodels.Model(book_input, book_vec)

a_bookid, b_bookid, a_userid, b_userid, a_y, b_y = cross_validation.train_test_split(bookid, userid, y)


metrics.mean_absolute_error(np.argmax(b_y, 1)+1, np.argmax(model.predict([b_bookid, b_userid]), 1)+1)

try:
    history = model.fit([a_bookid, a_userid], a_y, 
                         nb_epoch=20, 
                         validation_data=([b_bookid, b_userid], b_y))
    plot(history.history['loss'])
    plot(history.history['val_loss'])
except KeyboardInterrupt:
    pass

ans=metrics.mean_absolute_error(
    np.argmax(b_y, 1)+1, 
    np.argmax(model.predict([b_bookid, b_userid]), 1)+1)

print "Testing accuracy=",ans 



