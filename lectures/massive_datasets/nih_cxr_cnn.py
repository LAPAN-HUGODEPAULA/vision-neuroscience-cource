# file: nih_cxr_cnn.py
import tensorflow as tf, numpy as np, pandas as pd, os, cv2
IMG_PATH = 'nih-cxr/images'
CSV_PATH = 'nih-cxr/Data_Entry_2017.csv'
df = pd.read_csv(CSV_PATH).sample(20000, random_state=0)  # quick subset
class_names = ['Atelectasis','Effusion','Infiltration']

def load(path):
    img = cv2.imread(os.path.join(IMG_PATH,path),0)
    return cv2.resize(img,(224,224))[...,None]/255.

images = np.array([load(p) for p in df['Image Index']])
labels = df['Finding Labels'].str.get_dummies().reindex(columns=class_names,fill_value=0)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,3,activation='relu',input_shape=(224,224,1)),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(64,3,activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(3,activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['AUC'])
model.fit(images, labels, validation_split=0.2, epochs=5, batch_size=64)