"""" this program is to train Bi- LSTM model """
                                   
import numpy as np
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize)


print("\033[H\033[J")
plt.close('all')

max_features = 15
n_points = 640
nu = np.linspace(0,1,n_points)
wavenumber_640 = np.linspace(0,1,640)

def random_chi3():
    """
    generates a random spectrum, without NRB.
    output:
        params =  matrix of parameters. each row corresponds to the [amplitude, resonance, linewidth] of each generated feature (n_lor,3)
    """
    n_lor = np.random.randint(1,max_features)
    a = np.random.uniform(0,1,n_lor)
    w = np.random.uniform(0,1,n_lor)
    g = np.random.uniform(0.001,0.008, n_lor)
    params = np.c_[a,w,g]
    return params


def build_chi3(params):
    """
    buiilds the normalized chi3 complex vector
    inputs: params: (n_lor, 3)
    outputs chi3: complex, (n_points, )
    """
    chi3 = np.sum(params[:,0]/(-nu[:,np.newaxis]+params[:,1]-1j*params[:,2]),axis = 1)

    return chi3/np.max(np.abs(chi3))

def sigmoid(x,c,b):
    return 1/(1+np.exp(-(x-c)*b))


def generate_nrb():
    """
    Produces a normalized shape for the NRB
    outputs
        NRB: (n_points,)
    """
    [r2, r4, r5]= np.random.randint(-10, 10,size=3)
    [r1,r3]=np.random.uniform(-1, 1,size=2)
    nrb=np.polyval([r1,r2,r3,r4,r5], nu)
    nrb=nrb-min(nrb)
    nrb=nrb/max(nrb)
    return nrb



def get_spectrum():
    """
    Produces a cars spectrum.
    It outputs the normalized cars and the corresponding imaginary part.
    Outputs
        cars: (n_points,)
        chi3.imag: (n_points,)
    """
    chi3 = build_chi3(random_chi3())*np.random.uniform(0.3,1)
    nrb = generate_nrb()
    noise = np.random.randn(n_points)*np.random.uniform(0.0005,0.003)
    cars = ((np.abs(chi3+nrb)**2)/2+noise)
    return cars, chi3.imag

def generate_batch(size = 1):
    X = np.empty((size, n_points,1))
    y = np.empty((size,n_points))

    for i in range(size):
        X[i,:,0], y[i,:] = get_spectrum()
    return X, y

# create and fit the LSTM network
import tensorflow as tf
from tensorflow import keras
import tensorflow
tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Bidirectional


paradict = {'n_units': 30, 'batch_size': 10, 'epochs': 10, 
                    'optmethod': 'Adam', 'learning_rate': 0.005, 
                    'drop_rate': 0.14, 'rdrop_rate': 0.16, 'rreg_l2':0*1e-6, 
                    'Bayesian': True}
n_units, batch_size, epochs = paradict['n_units'], paradict['batch_size'], paradict['epochs']
optmethod, learning_rate = paradict['optmethod'], paradict['learning_rate']
drop_rate, rdrop_rate, rreg_l2 = paradict['drop_rate'], paradict['rdrop_rate'], paradict['rreg_l2']
Bayesian = paradict['Bayesian']

n_steps = 640
n_features = 1

recurrent_regularizer = None
if rreg_l2 is not None: 
    recurrent_regularizer = l2(rreg_l2)  # typical values are 1e-6, 1e-5, 1e-4 ...

opt = keras.optimizers.Adam(learning_rate=0.005)
tf.keras.backend.clear_session()

model_2 = Sequential()
model_2.add(Bidirectional(LSTM(n_units, activation='tanh', dropout=0.0, kernel_regularizer=None, \
                recurrent_dropout=rdrop_rate, recurrent_regularizer=recurrent_regularizer, recurrent_activation='sigmoid', \
                return_sequences=True, implementation=1), input_shape=(n_steps, n_features)))   # No dropout in first layer

model_2.add(Bidirectional(LSTM(n_units, activation='tanh', dropout=drop_rate, kernel_regularizer=None, \
                recurrent_dropout=rdrop_rate, recurrent_regularizer=recurrent_regularizer, recurrent_activation='sigmoid', \
                return_sequences=True, implementation=1)))

model_2.add(Bidirectional(LSTM(n_units, activation='tanh', dropout=drop_rate, kernel_regularizer=None, \
                recurrent_dropout=rdrop_rate, recurrent_regularizer=recurrent_regularizer, recurrent_activation='sigmoid', \
                return_sequences=True, implementation=1)))


model_2.add(TimeDistributed(Dense(1, activation='tanh')))  # No dropout before last layer.

model_2.compile(loss='mse', optimizer=opt, metrics=['mean_absolute_error','mse','accuracy'])
model_2.summary()


xnew2, ynew2 = generate_batch(50000)
history = model_2.fit(xnew2, ynew2, validation_split=0.20, epochs=30, batch_size=10, verbose=1)

'plotting Accuracy and loss'

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.figure()
plt.plot(epochs, acc, 'bo', label='Training')
plt.plot(epochs, val_acc, 'r', label='Validation')
plt.ylabel('Accuracy',fontweight='bold',size=20,family='Times New Roman')
plt.xlabel('Epoch',fontweight='bold',size=20,family='Times New Roman')
plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
plt.title('Training and validation accuracy',fontweight='bold',size=17,family='Times New Roman')
plt.legend(loc=0)
legend_properties = {'weight':'bold','size': 17,'family': 'Times New Roman'}
plt.legend(prop=legend_properties)
plt.figure()
plt.plot(epochs, loss, 'bo-', label='Training')
plt.plot(epochs, val_loss, 'ro-', label='Validation')
plt.ylabel('Loss',fontweight='bold',size=20,family='Times New Roman')
plt.xlabel('Epoch',fontweight='bold',size=20,family='Times New Roman')
plt.title('Training and validation loss',fontweight='bold',size=17,family='Times New Roman')
plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
plt.legend(loc=0)
legend_properties = {'weight':'bold','size': 17,'family': 'Times New Roman'}
plt.legend(prop=legend_properties)
plt.show()

# model_2.save('bi-LSTM_model_weights.h5')
   