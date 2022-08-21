import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split

from model import get_custom_model
from utilities import read_data_file, visualize_keypoint, plot_training_process


images, keypoints = read_data_file(img_path="face_images.npz",
                                   kp_path="facial_keypoints.csv")

X_train, X_test, y_train, y_test = train_test_split(images, keypoints, test_size=0.25, shuffle=True, random_state=42)

model = get_custom_model()
model.summary()

# plot_model(model, 'architecture.png', show_shapes=True)

check_point = tf.keras.callbacks.ModelCheckpoint('best.h5', monitor='val_loss', save_best_only=True)

model.compile(optimizer='Adam',
              loss='mse',
              metrics=[tf.keras.metrics.RootMeanSquaredError(), 'mae', 'mape'])

history = model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), callbacks=[check_point])

model.save('model_v1.h5')

plot_training_process(history)
