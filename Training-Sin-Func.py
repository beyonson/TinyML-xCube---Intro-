#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow.keras import layers


# In[2]:


nsamples = 1000
val_ratio = 0.2
test_ratio = 0.2
tflite_model_name = 'sine_model'
c_model_name = 'sine_model'


# In[3]:


# generating random samples
np.random.seed(1234)
x_values = np.random.uniform(low=0, high=(2*math.pi), size=nsamples)
plt.plot(x_values)


# In[4]:


# create noisy sine wave
y_values= np.sin(x_values) + (0.1 * np.random.randn(x_values.shape[0]))
plt.plot(x_values, y_values, '.')


# In[5]:


# splitting dataset
val_split = int(val_ratio * nsamples)
test_split = int(val_split + (test_ratio * nsamples))
x_val, x_test, x_train = np.split(x_values, [val_split, test_split])
y_val, y_test, y_train = np.split(y_values, [val_split, test_split])

# make sure it adds up
assert(x_train.size + x_val.size + x_test.size) == nsamples

# plot the data
plt.plot(x_train, y_train, 'b.', label="train")
plt.plot(x_test, y_test, 'r.', label="test")
plt.plot(x_val, y_val, 'g.', label="val")
plt.legend()
plt.show()


# In[6]:


# create model
model = tf.keras.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(1,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1))


# In[7]:


model.summary()


# In[8]:


# add optimizer and loss functions 
model.compile(optimizer='rmsprop', loss='mae', metrics=['mae'])


# In[9]:


# train using fit function
history = model.fit(x_train, y_train, epochs=500, batch_size=64, validation_data=(x_val, y_val))


# In[10]:


# plot dis johnny
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label="training loss")
plt.plot(epochs, val_loss, 'r', label="validation loss")
plt.title("training and validation loss")
plt.legend()
plt.show()


# In[11]:


# plot predictions against actual values
predictions = model.predict(x_test)

plt.clf()
plt.title("comparison of predictions to actual values")
plt.plot(x_test, y_test, 'b.', label="actual")
plt.plot(x_test, predictions, 'g.', label="predictions")
plt.legend()
plt.show()


# In[12]:


# convert to tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = converter.convert()

open(tflite_model_name + '.tflite', 'wb').write(tflite_model)


# In[13]:


# function converting to HEX for C
def hex_to_c_array(hex_data, var_name):
    
    c_str = ''
    
    # create header guard
    c_str += '#ifndef' + var_name.upper() + '_H\n'
    c_str += '#define' + var_name.upper() + '_H\n\n'
    
    # add array length at top of file
    c_str += '\nunsigned int ' + var_name + '_len = ' + str(len(hex_data)) + ';\n'
    
    # declare c variable
    c_str += 'unsigned char' + var_name + '[] = {'
    hex_array = []
    for i, val in enumerate(hex_data) : 
        
        # construct string from hex
        hex_str = format(val, '#04x')
        
        # add formatting so each line stays within 80 chars
        if (i + 1) < len(hex_data):
            hex_str += ','
        if (i + 1) %12 == 0:
            hex_str += '\n'
        hex_array.append(hex_str)
        
    # add closing brace
    c_str += '\n' + format(''.join(hex_array)) + '\n};\n\n'
    
    # close out header guard
    c_str += '#endif //' + var_name.upper() + '_H'
    
    return c_str
        
        


# In[14]:


# write tflite model to a c source file
with open(c_model_name + '.h', 'w') as file:
    file.write(hex_to_c_array(tflite_model, c_model_name))


# In[15]:


# save keras model
model.save(tflite_model_name + '.h5')


# In[ ]:




