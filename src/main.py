import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import measure
from skimage.transform import radon
from skimage.transform import probabilistic_hough_line
from skimage import measure
from scipy import interpolate
from scipy import stats



def build_model(input_shape, num_classes):
    """
    Builds a convolutional neural network (CNN) for image classification.

    Args:
        input_shape (tuple): The shape of the input images (height, width, channels).
        num_classes (int): The number of classes for classification.

    Returns:
        keras.Model: The compiled CNN model.
    """
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    """
    Main function to run the wafer defect detection model.
    """
    # Define your dataset parameters
    # Replace with your actual data dimensions and number of defect types
    INPUT_SHAPE = (128, 128, 1)  # Example: 128x128 grayscale images
    NUM_CLASSES = 9  # Example: 8 defect types + 1 normal

    # Build the model
    model = build_model(INPUT_SHAPE, NUM_CLASSES)

    # Print the model summary
    model.summary()

    # TODO: Load your wafer map data from the 'data' directory
    # TODO: Preprocess the data (resize, normalize, etc.)
    # TODO: Split the data into training and testing sets
    # TODO: Train the model using model.fit()
    # TODO: Evaluate the model using model.evaluate()
    # TODO: Save the trained model

    # Load the data from the pickle file
    file_path = os.path.join('data', 'LSWMD.pkl')
    df = pd.read_pickle(file_path)

    # Dont need wafer index feature in classification
    df = df.drop(['waferIndex'], axis = 1)

    # # Drawing the graph that shows wafer index distribution
    # uni_Index=np.unique(df.waferIndex, return_counts=True)
    # plt.bar(uni_Index[0],uni_Index[1], color='red', align='center', alpha=0.5)
    # plt.title(" wafer Index distribution")
    # plt.xlabel("index #")
    # plt.ylabel("frequency")
    # plt.xlim(0,26)
    # plt.ylim(30000,34000)
    # plt.show()

    df['failureNum']=df.failureType
    df['trainTestNum']=df.trianTestLabel
    mapping_type={'Center':0,'Donut':1,'Edge-Loc':2,'Edge-Ring':3,'Loc':4,'Random':5,'Scratch':6,'Near-full':7,'none':8}
    mapping_traintest={'Training':0,'Test':1}
    df=df.replace({'failureNum':mapping_type, 'trainTestNum':mapping_traintest})

    df_withpattern = df[(df['failureNum']>=0) & (df['failureNum']<=7)]
    df_withpattern = df_withpattern.reset_index()
    
    num_to_show = 20
    total_samples = df_withpattern.shape[0]

    # Calculate start index to get 50 samples from the middle
    start_idx = max(0, (total_samples // 2) - (num_to_show // 2))
    end_idx = start_idx + num_to_show

    fig, ax = plt.subplots(nrows = 2, ncols = 10, figsize=(30, 30))
    ax = ax.ravel(order='C')
    for i in (range(20)):
        img = df_withpattern.waferMap[i]
        ax[i].imshow(img)
        ax[i].set_title(df_withpattern.failureType[i][0][0], fontsize=10)
        ax[i].set_xlabel(df_withpattern.index[i], fontsize=8)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    plt.tight_layout()
    plt.show() 

    # START OF DATA TRANSFORMATION



if __name__ == '__main__':
    main()