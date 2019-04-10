from tqdm import tqdm
import pandas as pd
import numpy as np
import tensorflow as tf
import argparse
import logging

                        
parser = argparse.ArgumentParser()
parser.add_argument('test_index', type=int, default=0, help="choose a test set by it index [0, 1, ...]")
args = parser.parse_args()

test_index = args.test_index

logging.basicConfig(level=logging.INFO, filename='log_'+str(test_index)+'.txt',
                        filemode='w', format='%(message)s')

# path to the csv data file
data_path = "DataMura\Mura-Music.csv"

BLOCK_SIZE = 6
SAMPLE_SIZE = 2048
STEP_SIZE = 128

DROPOUT = 0.7

NUM_CLASSES = 3

LEARNING_RATE = 0.001
MOMENTUM = 0.9

NUM_EPOCHS =1000

BATCH_SIZE = 100

OUTPUT_STEP = 10

PREFETCH_SIZE = 1000

# read the data
data = pd.read_csv(data_path, header=None)
    
num_of_block = len(data)//BLOCK_SIZE
num_of_block_per_class = num_of_block // NUM_CLASSES

# Generate samples from the original data
block = {}
for i in tqdm(range(0, len(data)-BLOCK_SIZE+1, BLOCK_SIZE)):
    block[i//BLOCK_SIZE] = data.iloc[i:i+BLOCK_SIZE]
    
block_one_row = {}
for i in tqdm(range(len(block))):
    one_row = block[i].iloc[0]
    for j in range(1, BLOCK_SIZE, 1):
        one_row = one_row.append(block[i].iloc[j], ignore_index=True)
    block_one_row[i] = one_row
    
generated_data = {}
for i in tqdm(range(len(block))):
    new_data = {}
    for j in range(0,len(block_one_row[i])-SAMPLE_SIZE+1, STEP_SIZE):
        new_data[j] = block_one_row[i][j:j+SAMPLE_SIZE].values
    generated_data[i] = pd.DataFrame.from_dict(new_data, orient='index')


for i in tqdm(range(len(generated_data))):
    label = 0
    if i > (num_of_block_per_class-1):
        label = 2
    if i > (num_of_block_per_class*2-1):
        label = 1
    generated_data[i] = generated_data[i].reset_index()
    generated_data[i] = generated_data[i].join(pd.DataFrame({'label':np.full(len(generated_data[0]),label)}))


test_set_index = [test_index, num_of_block_per_class+test_index, num_of_block_per_class*2+test_index]

test_data = pd.DataFrame(columns=generated_data[0].columns)
train_data = pd.DataFrame(columns=generated_data[0].columns)
for i in tqdm(range(len(generated_data))):
    if i in test_set_index:
        test_data = pd.concat([test_data, generated_data[i]], ignore_index=True)
    else:
        train_data = pd.concat([train_data, generated_data[i]], ignore_index=True)

        
X_test = np.asarray(test_data.iloc[:,0:SAMPLE_SIZE], dtype=np.float32).reshape(-1,SAMPLE_SIZE)
y_test = np.asarray(pd.get_dummies(test_data['label']), dtype=np.float32)

X_train = np.asarray(train_data.iloc[:,0:SAMPLE_SIZE], dtype=np.float32).reshape(-1,SAMPLE_SIZE)
y_train = np.asarray(pd.get_dummies(train_data['label']), dtype=np.float32)


NUM_TRAIN_SAMPLE = len(y_train)
PER_EPOCH_STEPS = int(np.ceil(NUM_TRAIN_SAMPLE/BATCH_SIZE))

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

train_dataset = train_dataset.shuffle(NUM_TRAIN_SAMPLE)
train_dataset = train_dataset.repeat().batch(BATCH_SIZE)
#train_dataset = train_dataset.prefetch(buffer_size=PREFETCH_SIZE)

test_dataset = test_dataset.batch(BATCH_SIZE).repeat()
test_dataset = test_dataset.prefetch(buffer_size=PREFETCH_SIZE)

NUM_TEST_SAMPLE = len(y_test)
TEST_PER_EPOCH_STEPS = int(np.ceil(NUM_TEST_SAMPLE/BATCH_SIZE))




model = tf.keras.Sequential()

model.add(tf.keras.layers.Reshape((SAMPLE_SIZE, 1), input_shape=(SAMPLE_SIZE,)))
model.add(tf.keras.layers.LSTM(100, return_sequences=False, input_shape=(SAMPLE_SIZE, 1)))


#model.add(tf.keras.layers.AveragePooling1D(16))
#model.add(tf.keras.layers.Conv1D(100, 16, activation='relu'))
#model.add(tf.keras.layers.MaxPooling1D(9))
#model.add(tf.keras.layers.Conv1D(100, 10, activation='relu'))
#model.add(tf.keras.layers.MaxPooling1D(8))
#model.add(tf.keras.layers.Conv1D(160, 10, activation='relu'))
#model.add(tf.keras.layers.Conv1D(160, 10, activation='relu'))
#model.add(tf.keras.layers.MaxPooling1D(3))
#model.add(tf.keras.layers.LSTM(10, activation='relu', return_sequences=True))
#model.add(tf.keras.layers.LSTM(10, activation='relu'))

#model.add(tf.layers.Flatten())

#model.add(tf.keras.layers.GlobalAveragePooling1D())
#model.add(tf.keras.layers.Dropout(DROPOUT))
#model.add(tf.keras.layers.Dense(128, activation='relu'))
#model.add(tf.keras.layers.Dropout(DROPOUT))
#model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(DROPOUT))
model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))

print(model.summary())

model.compile(optimizer=tf.train.AdamOptimizer(LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_dataset, epochs=NUM_EPOCHS, steps_per_epoch=PER_EPOCH_STEPS, validation_data=test_dataset, validation_steps=TEST_PER_EPOCH_STEPS)

