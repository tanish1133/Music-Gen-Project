import pandas as pd
from ast import literal_eval
import numpy as np
from Dataset import midi_helper 

notes_df = pd.read_csv ('Dataset/notes.csv')
test_df = pd.read_csv ('Dataset/testset.csv')

data_test = test_df[['x_test','future']].to_numpy()

x_test_string = data_test[:,0]
y_test_string = data_test[:,1]
x_test = []
y_test = []
for i in x_test_string:

    b = "[]\n"
    for char in b:
        i = i.replace(char, "")
    input_x_test = [int(j) for j in i.split()]
    x_test.append(input_x_test)

for i in y_test_string:

    b = "[]\n"
    for char in b:
        i = i.replace(char, "")
    input_y_test = [int(j) for j in i.split()]
    y_test.append(input_y_test)
x_test = np.array(x_test)
y_test = np.array(y_test)
    


notes_ = notes_df.to_numpy()[:,1]
unique_notes = dict(enumerate(notes_.flatten(), 0))
# unique_notes = {value : key for (key, value) in unique_notes_reverse.items()}
