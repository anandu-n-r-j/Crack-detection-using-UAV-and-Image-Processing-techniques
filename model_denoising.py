import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import shutil

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.callbacks import EarlyStopping

# Source directory paths
source_train = r'C:\Users\anand\Downloads\Denoising_Images_model\train'
source_cleaned_train = r'C:\Users\anand\Downloads\Denoising_Images_model\train_cleaned'

# Destination directory (working directory)
destination = os.getcwd()

# Copying files from source directories to working directory
shutil.copytree(source_train, os.path.join(destination, 'train_data'))
shutil.copytree(source_cleaned_train, os.path.join(destination, 'train_cleaned_data'))

# Stoinge image names in list for later use
train_img = sorted(os.listdir(os.path.join(destination, 'train_data')))
train_cleaned_img = sorted(os.listdir(os.path.join(destination, 'train_cleaned_data')))

# Preparing function
def process_image(path):
    img = cv2.imread(path)
    img = np.asarray(img, dtype="float32")
    img = cv2.resize(img, (540, 420))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    img = np.reshape(img, (420, 540, 1))
    
    return img

# Preprocessing images
train = []
train_cleaned = []

for f in sorted(os.listdir(os.path.join(destination, 'train_data'))):
    train.append(process_image(os.path.join(destination, 'train_data', f)))

for f in sorted(os.listdir(os.path.join(destination, 'train_cleaned_data'))):
    train_cleaned.append(process_image(os.path.join(destination, 'train_cleaned_data', f)))

plt.figure(figsize=(15, 25))
for i in range(0, 8, 2):
    plt.subplot(4, 2, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train[i][:, :, 0], cmap='gray')
    plt.title('Noise image: {}'.format(os.path.basename(train_img[i])))

    plt.subplot(4, 2, i+2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_cleaned[i][:, :, 0], cmap='gray')
    plt.title('Denoised image: {}'.format(os.path.basename(train_cleaned_img[i])))

plt.show()

# Converting list to numpy array
X_train = np.asarray(train)
y_train = np.asarray(train_cleaned)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15)

conv_autoencoder = Sequential()

# Encoder
conv_autoencoder.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(420, 540, 1), activation='relu', padding='same'))
conv_autoencoder.add(MaxPooling2D((2, 2), padding='same'))

conv_autoencoder.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'))
conv_autoencoder.add(MaxPooling2D((2, 2), padding='same'))

# Decoder
conv_autoencoder.add(Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'))
conv_autoencoder.add(Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'))

# Output
conv_autoencoder.add(Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same'))

conv_autoencoder.summary()

conv_autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
history = conv_autoencoder.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200, batch_size=16, callbacks=[early_stop])

# Checking how loss & mae went down
epoch_loss = history.history['loss']
epoch_val_loss = history.history['val_loss']
epoch_mae = history.history['mae']
epoch_val_mae = history.history['val_mae']

plt.figure(figsize=(20, 6))
plt.subplot(1, 2, 1)
plt.plot(range(0, len(epoch_loss)), epoch_loss, 'b-', linewidth=2, label='Train Loss')
plt.plot(range(0, len(epoch_val_loss)), epoch_val_loss, 'r-', linewidth=2, label='Val Loss')
plt.title('Evolution of loss on train & validation datasets over epochs')
plt.legend(loc='best')

plt.subplot(1, 2, 2)
plt.plot(range(0, len(epoch_mae)), epoch_mae, 'b-', linewidth=2, label='Train MAE')
plt.plot(range(0, len(epoch_val_mae)), epoch_val_mae, 'r-', linewidth=2, label='Val MAE')
plt.title('Evolution of MAE on train & validation datasets over epochs')
plt.legend(loc='best')

plt.show()

conv_autoencoder.save('model1.h5')
