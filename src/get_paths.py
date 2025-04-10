import os
path = r"data\compcars\data\train_test_split"

# to store files in a list
list = []

# dirs=directories
for (root, dirs, file) in os.walk(path):
    for f in file:
        if '.txt' in f:
            print(f)