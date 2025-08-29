# Wafer Defect Detection

## Project Overview
This project uses a **Convolutional Neural Network (CNN)** to detect and classify defects on semiconductor wafers.  
The notebook walks through **loading data, preprocessing, visualizing wafer maps, defining a model, and preparing for training**.  

### Critical Importance
**Semiconductor wafers are the foundation of all electronics.**  
Detecting defects early helps:  
- Improve manufacturing yield  
- Reduce costs from faulty chips  
- Ensure more reliable products  

Automating defect detection with deep learning helps scale up quality assurance advancement and prolong efficiency for semiconductors.

---

## Key Sections Explained

### 1. Imports
We import all the libraries needed for the project:
- `tensorflow` & `keras` - for building and training the CNN  
- `pandas` & `numpy` - for handling and manipulating data  
- `matplotlib` - for plotting wafer maps  
- `scikit-image` - for image processing  
- `scipy` - for interpolation and statistics  

**Relevance/Importance:** Without these libraries, the notebook cannot run. This step sets up the environment for everything else.

---

### 2. Model Definition
The `build_model()` function defines a CNN:
- Three convolutional layers followed by pooling layers  
- Flatten layer to prepare data for fully connected layers  
- Dense layers for combining features  
- Output layer with softmax activation for classifying defects  

**Relevance/Importance:** This is the core of the project. The CNN will learn patterns in wafer images to distinguish different types of defects.

---

### 3. Build and Summarize Model
- The CNN is instantiated using the input image size and number of defect classes.  
- `model.summary()` prints each layer and the number of parameters.  

**Relevance/Importance:** Helps ensure the model is built correctly and provides insight into its complexity.

---

### 4. Load Data
- The wafer dataset (`LSWMD.pkl`) is loaded with `pandas`.  
- Displaying the first rows confirms that the data loaded correctly.  

**Relevance/Importance:** You need to verify that your dataset is available and structured correctly before any analysis.

---

### 5. Preprocessing and Cleaning
- Removes unnecessary columns like `waferIndex`.  
- Converts categorical labels into numerical labels (`failureNum` and `trainTestNum`).  
- Splits data into training, testing, and unlabeled sets.  

**Relevance/Importance:** Preprocessing ensures the CNN can use the data, avoids errors during training, and separates training and testing data for proper evaluation.

---

### 6. Data Exploration and Visualization
- Wafer maps are visualized using a custom colormap to highlight defects.  
- Each defect type is shown for comparison.

<img width="690" height="1340" alt="image" src="https://github.com/user-attachments/assets/306f18c7-b2ad-46e4-b3ce-7a2c2b741cb2" />

**Relevance/Importance:** Visualization helps understand the dataset, detect imbalances, and confirm that defect patterns are visible and distinguishable.

---

### 7. Next Steps
- Preprocess images: resize, normalize  
- Train the CNN with the training set  
- Evaluate accuracy and performance on the test set  
- Save the trained model for future use  

**Relevance/Importance:** Outlines the remaining steps to turn this notebook into a fully functioning defect detection system.

---

## Project Structure

**Notes:**  
- `src/` - contains your notebook (`.ipynb`).  
- `data/` - store your dataset here (do not upload if large).  
- `images/` - save wafer plots for GitHub display.  
- `README.md` - sits in the root folder for automatic GitHub display.  

---

## Requirements
You’ll need the following Python packages:  

- tensorflow  
- keras  
- numpy  
- pandas  
- matplotlib  
- scikit-image  
- scipy  

Install them with:

```bash
pip install tensorflow numpy pandas matplotlib scikit-image scipy
