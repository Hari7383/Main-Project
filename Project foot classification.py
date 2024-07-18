#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tensorflow tensorflow-gpu opencv-python matplotlib')


# In[2]:


get_ipython().system('pip install tensorflow tensorflow-gpu opencv-python matplotlib')


# In[3]:


get_ipython().system('pip list')


# In[4]:


import tensorflow as tf
import os


# In[5]:


pip install tensorflow


# In[6]:


pip install tensorflow


# In[7]:


import tensorflow as tf
import os
from matplotlib import pyplot as plt


# In[8]:


gpus=tf.config.experimental.list_physical_devices('CPU')


# In[9]:


gpus


# In[10]:


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)


# In[11]:


import cv2
import imghdr


# In[12]:


data_dir='foot datasets'


# In[13]:


os.listdir(os.path.join(data_dir,'foot datasets'))


# In[14]:


image_exts=['jpeg','jpg','bmp','png']


# In[15]:


image_exts


# In[16]:


img=cv2.imread(os.path.join('foot datasets','foot datasets','Screenshot 2023-09-08 190649.png'))


# In[17]:


img.shape


# In[18]:


plt.imshow(img)


# In[19]:


for image_class in os.listdir(data_dir): 
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try: 
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts: 
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e: 
            print('Issue with image {}'.format(image_path))
            # os.remove(image_path)


# In[20]:


import numpy as np
from matplotlib import pyplot as plt


# In[21]:


data = tf.keras.utils.image_dataset_from_directory('foot datasets')


# In[22]:


data_iterator = data.as_numpy_iterator()


# In[23]:


batch = data_iterator.next()


# In[24]:


fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])


# In[25]:


data = data.map(lambda x,y: (x/255, y))


# In[26]:


data.as_numpy_iterator().next()


# In[27]:


train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)


# In[28]:


train_size


# In[29]:


train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)


# In[30]:


train


# In[31]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


# In[32]:


model = Sequential()


# In[33]:


model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[34]:


model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])


# In[35]:


model.summary()


# In[36]:


logdir='logs'


# In[37]:


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


# In[38]:


hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])


# In[39]:


hist.history


# In[40]:


fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()


# In[41]:


fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()


# In[42]:


from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy


# In[43]:


pre = Precision()
re = Recall()
acc = BinaryAccuracy()


# In[44]:


len(test)


# In[45]:


for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)


# In[46]:


print(f'percision:{pre.result().numpy()},Recall:{ re.result().numpy()},Accuracy:{acc.result().numpy}')


# In[47]:


import cv2


# In[48]:


img = cv2.imread('foottest3.jpg')
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show()


# In[49]:


resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()


# In[50]:


resize.shape


# In[51]:


np.expand_dims(resize,0).shape


# In[52]:


yhat=model.predict(np.expand_dims(resize/255,0))


# In[53]:


yhat


# In[68]:


if yhat<5.1:
    print(f'predicted class is not human')
else:
    print(f'predicted class is human')


# In[55]:


from tensorflow.keras.models import load_model


# In[56]:


model.save(os.path.join('models','footprint.h5'))


# In[57]:


new_model = load_model(os.path.join('models','footprint.h5'))


# In[58]:


yhatnew=new_model.predict(np.expand_dims(resize/255, 0))


# In[59]:


import cv2


# In[60]:


img = cv2.imread('Screenshot 2023-09-08 190532.png')
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show()


# In[61]:


resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()


# In[62]:


resize.shape


# In[63]:


np.expand_dims(resize,0).shape


# In[64]:


yhat=model.predict(np.expand_dims(resize/255,0))


# In[65]:


yhat


# In[69]:


if yhat>2:
    print(f'predicted class is not human')
else:
    print(f'predicted class is human')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




