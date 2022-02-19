# first neural network with keras tutorial
import random

from numpy import loadtxt
import csv
from keras.models import Sequential
from keras.layers import Dense

# load the dataset
rows = []
tpc_tags = []
data_tags = []
with open('digested.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        # print(', '.join(row))
        row[1] = row[1].split('.')[-1]
        row[2] = row[2].split('.')[-1]
        if row[3] not in tpc_tags:
            tpc_tags.append(row[3])
        row[3] = tpc_tags.index(row[3])
        if row[5].__contains__('[ACK]'):
            row[5] = 1
        else:
            row[5] = 0
        if row.__len__() > 7:
            row[6] = row[-1]
            row = row[:7]
        rows.append(row)

random.shuffle(rows)

with open('prepared.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row in rows:
        spamwriter.writerow(row)
exit()
dataset = open('digested.csv', delimiter=',')
for row in dataset:
    print(row)
exit()

# split into input (X) and output (y) variables
X = dataset[:,0:7]
y = dataset[:,7]

# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=7, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10)

# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
