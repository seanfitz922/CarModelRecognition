import scipy.io

mat_file_path = r"C:\Users\seanf\Desktop\School\Pattern Recognition\CarModelRecognition\data\compcars\data\misc\make_model_name.mat"

# Load the .mat file (ensure the file path is correct)
data = scipy.io.loadmat(mat_file_path)

# 'model_names' is typically stored as a 2D array; we take the first row
model_names = data['model_names']

# Loop through and print each model with its corresponding id (starting at 1)
for i, model in enumerate(model_names, start=1):
    # Adjust extraction depending on the structure of model (here assuming it's a 1-element list or a string)
    model_name = model[0] if isinstance(model, (list, tuple)) else model
    print("Model ID: {}, Model Name: {}".format(i, model_name))
