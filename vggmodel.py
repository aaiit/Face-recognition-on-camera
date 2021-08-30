import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
import warnings



import os
# source_dir=os.path.join('/kaggle','input','pins-face-recognition','105_classes_pins_dataset')


class IdentityMetadata():
    def __init__(self, base, name, file):
        self.base = base
        # identity name
        self.name = name
        # image file name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file) 
    



import cv2
def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout, Flatten, Activation

def vgg_face():	
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    return model


model = vgg_face()

model.load_weights('vgg_face_weights.h5')



from tensorflow.keras.models import Model
vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)


# Get embedding vector for first image in the metadata using the pre-trained model
img_path = "Image path to an image.png"
img = load_image(img_path)

# Normalising pixel values from [0-255] to [0-1]: scale RGB values to interval [0,1]
img = (img / 255.).astype(np.float32)
img = cv2.resize(img, dsize = (224,224))
print(img.shape)

# Obtain embedding vector for an image
# Get the embedding vector for the above image using vgg_face_descriptor model and print the shape 
embedding_vector = vgg_face_descriptor.predict(np.expand_dims(img, axis=0))[0]
print(embedding_vector.shape)




from tqdm import tqdm
metadata = os.listdir("cropped_dataset_faces/")
metadata = np.array([ "cropped_dataset_faces/"+i for i in metadata])
total_images = len(metadata)

print('total_images :', total_images)


print(metadata[0])

import pickle

"""
py vggmodel.py run
"""
if len(sys.argv)==2 and sys.argv[1] == "run":
    embeddings = np.zeros((metadata.shape[0], 2622))
    for i, m in tqdm(list(enumerate(metadata))):
        img_path = metadata[i]#.image_path()
        img = load_image(img_path)
        img = (img / 255.).astype(np.float32)
        img = cv2.resize(img, dsize = (224,224))
        embedding_vector = vgg_face_descriptor.predict(np.expand_dims(img, axis=0))[0]
        embeddings[i]=embedding_vector # embedding_vector_vector

    with open('embeddings.pickle', 'wb') as f:
        pickle.dump([embeddings], f)
else:
    with open('embeddings.pickle', 'rb') as f:
        embeddings = pickle.load(f)[0]


def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))



def show_pair(idx1, idx2):
    plt.figure(figsize=(8,3))
    plt.suptitle(f'Distance between {idx1} & {idx2}= {distance(embeddings[idx1], embeddings[idx2]):.2f}')
    plt.subplot(121)
    plt.imshow(load_image(metadata[idx1]))
    plt.subplot(122)
    plt.imshow(load_image(metadata[idx2]));    

show_pair(900, 901)
show_pair(900, 101)




def predict(img_path):
    img = load_image(img_path)
    img = (img / 255.).astype(np.float32)
    img = cv2.resize(img, dsize = (224,224))
    embedding_vector = vgg_face_descriptor.predict(np.expand_dims(img, axis=0))[0]
    k = sorted([[distance(embeddings[i],embedding_vector),i]  for i in tqdm(range(len(metadata)))])[0][1]
    print("This face is most similar to this",metadata[k])

# testing on new face
predict("998.jpg")

# sequence = []

def predict_frame(frame):
    img = (frame / 255.).astype(np.float32)
    img = cv2.resize(img, dsize = (224,224))
    embedding_vector = vgg_face_descriptor.predict(np.expand_dims(img, axis=0))[0]
    k = sorted([[distance(embeddings[i],embedding_vector),i]  for i in tqdm(range(len(metadata)))])[0][1]
    print("This face is most similar to this",metadata[k])

    # sequence.append(k)
    # sequence = sequence[:10]
    # x = sum(sequence)/len(sequence)
    # k = sorted([[abs(x-i),i] for i in sequence])[0][1]

    return metadata[k]#.split("/")[-1].split(".")[0]
