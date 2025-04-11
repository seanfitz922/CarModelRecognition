# CarModelRecognition
 
Project: 
This project is a computer vision system that automatically identifies the make and model of a car from a photo using deep learning. It leverages a fine-tuned EfficientNetV2 neural network trained on the Stanford Cars Dataset to classify car images with high accuracy. Users can drag and drop images into a user-friendly GUI, and the system instantly predicts the car model.

How to run:
    Download required packages: pip install -r requirements.txt

    Run app.py

    Drag and drop jpg or png into GUI frame. 


Note:
    Due to limitations and age of datasets, modern cars (especially above model year ~2020), are rarely correctly identified due to no model id being present. 

    Model is excellent at classifying cars in the mid 1980's to 2010's.

    Recommeneded cars to test: 
                                2005 Silverado (my car)
                                2004 Spyker C8 (my favorite car)
                                Your dream car!

Future Improvements: 
    Clean up code base
    Expand dataset (possibly webscraping)
    Redo GUI

Datasets from: 
    Stanford Dataset https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset
    CompCars Dataset https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html
