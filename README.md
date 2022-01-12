# Image Classifier using Tensorflow 2.x
#### _A project using Tensorflow for image classification._

Sup guys. In this project I will be presenting an image classifier using Tensorflow 2.x.
<br>
So far I've used it to classify insects, being: 
> * bee
> * cockroach
> * butterfly
> * ant
> * spider

but the algorithm is generalist, and can be used to classify any other object, just insert a dataset in the 'dataset' directory following the logical structure.
<br>
>I'm just at the beginning of my learning of Tensorflow, so there's a lot of things to improve on the algorithm.

### Instaling and Loading the Packages

```
# Python version
from platform import python_version
print('Python version used in this notebook:', python_version())
```
```
# Imports
# Here are all the packages that will be used.
import sklearn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib2 import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation
from tensorflow.keras.layers.experimental.preprocessing import RandomZoom
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision
from tensorflow.keras.metrics import Recall
```
Let's check if there is GPU available to perform Tensorflow operations
```
print('Number of GPUs available: {}'.format(len(tf.config.list_physical_devices('GPU'))))
```
```
#Let's display the operations logs
tf.debugging.set_log_device_placement(True)
```
```
# Path to training data
path_training = Path("dataset/train")
# Path to test data
path_test = Path("dataset/test")
# Path to calculate classes lenght
path_len = Path("dataset/train")
```
```
# Listing folder contents
train_imgs = list(path_training.glob("*/*"))
# Lambda expression that extracts just the value with the path of each image
train_imgs = list(map(lambda x: str(x), train_imgs))
# Total training images
print('Total training images: {}'.format(len(train_imgs)))
# Classes lenght
len_class = len(list(path_len.glob("*/")))
print('Classes lenght: {}'.format(len_class))
```

### Data Pre-Processing

The first step of the algorithm is to perform the pre-processing of data, such as:
> * Get the image labels
> * Data encoder using sklearn.preprocessing.LabelEncoder
> * Split data for training and testing using sklearn.model_selection.train_test_split

```
# Function that gets the label of each image
def extract_label(path_img):
    return path_img.split("\\")[-2]
```
```
# Applying the function
train_img_labels = list(map(lambda x: extract_label(x), train_imgs))
```
```
# Create a object
encoder = LabelEncoder()
```
```
# Apply the fit_transform
train_img_labels = encoder.fit_transform(train_img_labels)
```
```
# Apply the One-Hot-Encoding on labels
train_img_labels = tf.keras.utils.to_categorical(train_img_labels)
```
```
# We split the training data into two samples, training and validation
X_train, X_valid, y_train, y_valid = train_test_split(train_imgs, train_img_labels)
```

### Dataset Augmentation

This step consists of increasing the dataset by creating copies of the images by applying
  some techniques like:
> * Rotate the image horizontally;
> * Rotate the image
> * Apply random zoom

```
# Resizing all images to 224 x 224 so that all input data have the same parameters
img_size = 224
resize = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.Resizing(img_size, img_size)])
```
```
# Creates the augmentation dataset object
data_augmentation = tf.keras.Sequential([RandomFlip("horizontal"),
                                         RandomRotation(0.2),
                                         RandomZoom(height_factor = (-0.3,-0.2)) ])
```

### Preparing the Data
```
# Hyperparameters
batch_size = 32
autotune = tf.data.experimental.AUTOTUNE
```
```
# Function to load and transform images
def load_transform(image, label):
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image, channels = 3)
    return image, label
```
```
# Function for preparing data in TensorFlow format
def prepare_dataset(path, labels, train = True):

    # Prepare the data
    image_paths = tf.convert_to_tensor(path)
    labels = tf.convert_to_tensor(labels)
    image_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    label_dataset = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
    dataset = dataset.map(lambda image, label: load_transform(image, label)) 
    dataset = dataset.map(lambda image, label: (resize(image), label), num_parallel_calls = autotune)
    dataset = dataset.shuffle(100)
    dataset = dataset.batch(batch_size)

    # If train = True apply dataset augmentation
    if train:
        dataset = dataset.map(lambda image, label: (data_augmentation(image), label), num_parallel_calls = autotune)
  
    # If train = False repeats over the dataset and returns
    dataset = dataset.repeat()

    return dataset
```
```
# Create training dataset
train_dataset = prepare_dataset(X_train, y_train)
```
```
# Shape
img, label = next(iter(train_dataset))
print(img.shape)
print(label.shape)
```
```
# Create the validation dataset
valid_dataset = prepare_dataset(X_valid, y_valid, train = False)
```

### Model Building

Now let's start the model building, compilation and execution step.

* The first step is to transfer learning using a network pre-trained by Tensorflow developers
> The EfficientNetB3 function returns a Keras image classification model, optionally loaded with weights pre-trained on ImageNet.

```
# Loading a pre-trained model
pre_model = EfficientNetB3(input_shape = (224,224,3), include_top = False)
```
```
# Adding our own layers to pre_model
model = tf.keras.Sequential([pre_model,
                              tf.keras.layers.GlobalAveragePooling2D(),
                              tf.keras.layers.Dense(len_class, activation = 'softmax')])
```
```
# Hyperparameters
lr = 0.001
beta1 = 0.9
beta2 = 0.999
ep = 1e-07
```
> An overview of ADAM (Adaptive Moment Estimation)
>> Link: https://ruder.io/optimizing-gradient-descent/

> An overview of Cross Entropy
> > Link: https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/categorical-crossentropy

```
# Model compilation
model.compile(optimizer = Adam(learning_rate = lr, 
                                beta_1 = beta1, 
                                beta_2 = beta2, 
                                epsilon = ep),
               loss = 'categorical_crossentropy', 
               metrics = ['accuracy', Precision(name = 'precision'), Recall(name = 'recall')])
```

### _Train model_
```
# Checkpoint
checkpoint1 = tf.keras.callbacks.ModelCheckpoint("modelo/best_model.h5", 
                                                verbose = 1, 
                                                save_best_only = True, 
                                                save_weights_only = True)

# Checkpoint
checkpoint2 = tf.keras.callbacks.ModelCheckpoint("modelo/last_model.h5", 
                                                verbose = 0, 
                                                save_best_only = False,
                                                save_weights_only = True,
                                                save_freq='epoch')
```
> Early stop serves to stop training if accuracy decreases after the threshold established as a parameter.
```
# Early stop
early_stop = tf.keras.callbacks.EarlyStopping(patience = 10) 
```
```
history = model.fit(train_dataset,
                     steps_per_epoch = len(X_train)//batch_size,
                     epochs = 40,
                     validation_data = valid_dataset,
                     validation_steps = len(y_train)//batch_size,
                     callbacks = [checkpoint1, checkpoint2])
```

### Model Evaluation
```
# To load the weights, we need to unfreeze the layers.
model.layers[0].trainable = True
# Load checkpoint weights and re-evaluate
model.load_weights("modelo/best_model.h5")
```
```
# Loading and preparing test data
path_test = list(path_test.glob("*/*"))
test_imgs = list(map(lambda x: str(x), path_test))
test_img_labels = list(map(lambda x: extract_label(x), test_imgs))
test_img_labels = encoder.fit_transform(test_img_labels)
test_img_labels = tf.keras.utils.to_categorical(test_img_labels)
test_image_paths = tf.convert_to_tensor(test_imgs)
test_img_labels = tf.convert_to_tensor(test_img_labels)
```
```
# Image decode function
def decode_images(image, label):
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image, channels = 3)
    image = tf.image.resize(image, [224,224], method = "bilinear")
    return image, label
```
```
# Create the test dataset
test_dataset = (tf.data.Dataset
                 .from_tensor_slices((test_imgs, test_img_labels))
                 .map(decode_images)
                 .batch(batch_size))
```
```
# Shape
image, label = next(iter(test_dataset))
print(image.shape)
print(label.shape)
```
```
# Evaluate the model
loss, acc, prec, rec = model.evaluate(test_dataset)
```
```
print("Accuracy: ", acc)
print("Precision: ", prec)
print("Recall: ", rec)
```
In my tests I got an accuracy of 0.97 in the model, but the dataset is small and the amount of times that the model was training was small.

### Predictions with the Trained Model

Finally, let's test the model in practice and see if it performed well.

```
# Function to load a new image
def load_img(image_path):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels = 3)
    image = tf.image.resize(image, [224,224], method = "bilinear")
    plt.imshow(image.numpy()/255)
    image = tf.expand_dims(image, 0) 
    return image
```
```
# Prediction function
def make_prevision(image_path, model, enc):
    image = load_img(image_path)
    prediction = model.predict(image)
    pred = np.argmax(prediction, axis = 1) 
    return enc.inverse_transform(pred)[0] 
```
### Some tests
```
# Prevision
prev = make_prevision("imagens/teste2.jpg", model, encoder)
print('The predicted object was: {}'.format(prev))
```
![image](https://user-images.githubusercontent.com/30851656/149157310-f016fcf2-de3c-44ca-a635-d0f572789592.png)
![image](https://user-images.githubusercontent.com/30851656/149157488-340d9441-8d43-4d05-b280-6e12e5e5f8b3.png)
![image](https://user-images.githubusercontent.com/30851656/149157529-1d5a7b90-d49b-4d0b-807c-9262ea0bed55.png)
![image](https://user-images.githubusercontent.com/30851656/149157568-9b0da104-b520-4a1e-8e16-2e689bb292da.png)
![image](https://user-images.githubusercontent.com/30851656/149157623-70c3891d-16eb-4252-8fbd-fe0dde978838.png)


