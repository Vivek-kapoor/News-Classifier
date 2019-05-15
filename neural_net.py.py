import csv
texts=[]
labels=[]
seq_len=0
# reading the proceesed file i.e., removed punctuation and stopwords
with open(r"G:\computer\pdf's\data Analytics\Project\final_training.csv", newline='',encoding='utf8') as myFile:
    reader = csv.reader(myFile)
    for row in reader:
        texts.append(row[0])
        seq_len += len(row[0].split())
        labels.append(row[1])

# selecting 80000 news from the data set
texts=texts[1:20000]
labels=labels[1:20000]

d = {'b':0,'m':1,'e':2,'t':3}

# will contain the count of each news article type.
count=[0,0,0,0]

classified_labels=[]
for i in labels:
    j=d.get(i)
    count[j]=count[j]+1
    classified_labels.append(j)

print(count)

from keras.utils.np_utils import to_categorical

# converting our labels i.e., our news category into one-hot encoding
one_hot_train_labels=to_categorical(classified_labels)
print(one_hot_train_labels)

print (len(texts),len(labels))
print ('mean seq len is:', seq_len / len(texts))

MAX_NB_WORDS = 5000    # signifies the vocabulary size
MAX_SEQUENCE_LENGTH = 261
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)   #use this to convert text into numeric value based on position of words in vocab length selected.
tokenizer.fit_on_texts(texts)                   #
sequences = tokenizer.texts_to_sequences(texts) # convert text into tokens wnd give each token a numeric value.

import numpy as np

#this function used to convert tokens converted into numeric value into one-hot vector
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

# creating training and testing data
train_data=sequences[:12000]
test_data=sequences[12000:]
train_labels=one_hot_train_labels[:12000]
test_labels=one_hot_train_labels[12000:]

# Our vectorized training data i.e., one-hot encoded format
x_train = vectorize_sequences(train_data)
# Our vectorized test data
x_test = vectorize_sequences(test_data)

# taking out validation set from training data
x_val=x_train[:2000]
partial_x_train=x_train[2000:]
y_val=train_labels[:2000]
partial_y_train=train_labels[2000:]

from keras.layers import K

# the below two function used is to get precision and recall
def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

from keras import models
from keras import layers
# building sequential model
model = models.Sequential()
# adding first hidden layer with 64 nodes and input vector to it will be nodes of dimension 10000 with activation function relu
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
# 2nd hidden layer
model.add(layers.Dense(64, activation='relu'))
# output layer with 4 nodes each for each category
model.add(layers.Dense(4, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', precision, recall])
# this the variable where all the computation out put such  as precision ,accuracy,loss,recall of the model will be stored
history=model.fit(partial_x_train,partial_y_train,epochs=5,batch_size=512,validation_data=(x_val,y_val))

# plot the graph
import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
import pylab
pylab.plot(epochs, loss_values, 'ko', label='training',)
pylab.plot(epochs, val_loss_values, 'k+', label='validating')
pylab.title("Loss vs Epoch(training_size = 80000)")
pylab.legend(loc='upper right')
pylab.xlabel("Epochs")
pylab.ylabel("Loss")
pylab.show()

acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
pylab.plot(epochs, acc_values, 'ko', label='training',)
pylab.plot(epochs, val_acc_values, 'k+', label='validating')
pylab.title("Accuracy vs Epoch(training_size = 80000)")
pylab.legend(loc='lower right')
pylab.xlabel("Epochs")
pylab.ylabel("Accuracy")
pylab.show()

results = model.evaluate(x_test, test_labels)
print(results)

# here giving the user input to classify the given news into the set of predefined categories

# "Bharti Airtel signs deal with Ericsson for strategic partnership on 5G technology"
# "Honda recalling 900,000 minivans because seats may tip forward"
a=["Bharti Airtel signs deal with Ericsson for strategic partnership on 5G technology"]
x= tokenizer.texts_to_sequences(a)
print(x)
y=vectorize_sequences(x)
print(y)

res=model.predict(y)
print(res)

# this would print the category with highest probability category.
print(np.argmax(res))

print(history.history)