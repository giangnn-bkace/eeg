import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

EXCERPT = 'DatabaseSpindles/excerpt1.txt'
HYPNO = 'DatabaseSpindles/Hypnogram_excerpt1.txt'
SAMPLE_FREQ = 100
SAVE_DATA_FILE_NAME = 'data_1.csv'

def generate_data(excerpt, hypnogram, sample_freq):
    SAMPLE_SIZE = sample_freq * 5
    rows = {}
    for i in range(len(hypnogram)):
        rows[i] = excerpt.iloc[i*SAMPLE_SIZE:(i+1)*SAMPLE_SIZE].values.transpose()[0]
    data = pd.DataFrame.from_dict(rows, orient="index")
    hypnogram.columns = ['Label']
    data = data.join(hypnogram)
    return data
	

excerpt_data = pd.read_csv(EXCERPT, header=0)
hypnogram_data = pd.read_csv(HYPNO, header=0)

generated_data = generate_data(excerpt_data, hypnogram_data, SAMPLE_FREQ)
generated_data.to_csv(SAVE_DATA_FILE_NAME, header=True)

print("%s is generated" %(SAVE_DATA_FILE_NAME))