import math
import gc
import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from bokeh.plotting import figure,output_file,show, output_notebook
from bokeh.layouts import column
from bokeh.models import Slider
from bokeh.io import output_file, show
from bokeh.models.widgets import FileInput



n_sample = 1

file_input = FileInput()
# Load the data  
frame = pd.read_csv('dataframe1.csv')
# formated data
data = np.hstack((frame.values.astype('float32')[:, :-2], frame.values.astype('float32')[:, [-2]]))

# normalize data 
scaler = MinMaxScaler(feature_range=(0, 1))

scaled = scaler.fit_transform(data)

iput, oput = scaled[:,:-1], scaled[:,[-1]]

# data split
train_x, test_x, train_y, test_y = train_test_split(iput, oput, test_size=0.5, shuffle=True)

# reshape data for model
train_x = train_x.reshape(n_sample, len(train_x), iput.shape[1])
test_x = test_x.reshape(n_sample, len(test_x), iput.shape[1])

train_y = train_y.reshape(n_sample, 31912)
test_y = test_y.reshape(n_sample, 31912)

gc.collect()

n_in, n_out, n_feature = train_x.shape[1], train_y.shape[1], train_x.shape[2]

# training model
model = Sequential()
model.add(Conv1D(
        filters=2,
        kernel_size=7, 
        activation='relu', 
        input_shape=(n_in, n_feature)
    )
)
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2)) #0.2 return better
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(n_out))
model.compile(optimizer='adam', loss='mae', metrics=['mse', 'mae'])


#model.save_weights(str('depth_weight1.212')+'.h5')
#model.load_weights('depth_weight1.253.h5')

mdl = model.fit(
        train_x, train_y, batch_size=32, validation_data=(test_x, test_y),
        epochs=10, shuffle=False, verbose=2)

output_notebook()

output_file("stats.html")

TOOLTIPS = [
    ("index", "$index"),
    ("(x,y)", "($x, $y)"),
    ("desc", "@desc"),
]

p = figure(tooltips=TOOLTIPS)

p.line([1,2,3,4,5,6,7,8,9,10], mdl.history['loss'], line_color="blue")
p.line([1,2,3,4,5,6,7,8,9,10], mdl.history['val_loss'], line_color="red")

slider = Slider(start=0.1, end=2, step=0.01, value=0.2)



show(column(p, slider, file_input))
