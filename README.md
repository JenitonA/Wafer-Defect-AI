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

## Vital Sections Explained

### 1. Imports

In this project, we load all the necessary libraries to handle data, process images, and train the CNN model:

- `tensorflow` & `keras` – used to build and train the convolutional neural network  
- `os` – for working with files and directories  
- `pandas` – to manage and manipulate datasets efficiently  
- `cv2` (OpenCV) – for image preprocessing and transformations  
- `numpy` – for numerical operations and array handling  
- `matplotlib` – to visualize wafer maps and plots  
- `sklearn.model_selection` – to split data into training, validation, and test sets  
- `skimage.transform.resize` – to resize wafer images to a consistent shape  

**Relevance/Importance:** These libraries are essential to the project workflow. They allow us to prepare the data, perform image transformations, and train a CNN. Without them, none of the processing or model training steps would be possible.

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

<img width="1490" height="622" alt="image" src="https://github.com/user-attachments/assets/25d03853-0f74-4820-b7dd-cc61f2a37dde" />

<img width="781" height="403" alt="image" src="https://github.com/user-attachments/assets/46ea6e41-163c-466d-b7b4-e9c918a29b05" />

**Relevance/Importance:** Visualization helps understand the dataset, detect imbalances, and confirm that defect patterns are visible and distinguishable.

---

### 7. Data Augmentation  

An inspection of the dataset shows that **the “none” (no defect) category accounts for about 68% of all samples**, while defect-related classes are significantly underrepresented.  
This **imbalance** can cause the CNN to overfit toward the dominant “none” class and misclassify rare defect types.  

To address this, **augmentation techniques** were applied depending on the defect type:  
- **Center, Edge-Loc, Edge-Ring, Loc:** wafer maps were shifted and flipped to simulate positional variation.  
- **Scratch:** rotated to preserve the elongated defect structure.  
- **Donut, Random, Near-full:** multiple rotations combined with light noise injection to enlarge the dataset.  

**Processing steps:**  
1. Reduced the “none” class from 117,945 samples to 8,000 to prevent overrepresentation.  
2. Augmented minority classes until each had ~2000 samples.  
3. Reconstructed the dataset to achieve a more balanced distribution across categories.  

**Impact of augmentation:**  
- Reduces bias toward the majority class  
- Encourages the CNN to learn distinctive defect features  
- Improves robustness by exposing the model to varied representations of defects  

**Class Distribution Before vs. After Augmentation:**  

| Defect Type   | Original Samples | After Augmentation |
|---------------|------------------|--------------------|
| none          | 117,945          | 8,000              |
| Edge-Ring     | 7,744            | 7,744              |
| Edge-Loc      | 4,151            | 4,151              |
| Center        | 3,435            | 3,435              |
| Loc           | 2,875            | 2,875              |
| Scratch       | 954              | 2,000              |
| Random        | 693              | 2,000              |
| Donut         | 444              | 2,000              |
| Near-full     | 119              | 2,000              |

Additionally, an analysis of wafer map dimensions helped determine the **input size for the CNN**, ensuring consistency across all samples.  

In summary, data augmentation created a balanced dataset that allows the model to learn rare defect types fairly, rather than defaulting to the majority class.  

<img width="859" height="477" alt="image" src="https://github.com/user-attachments/assets/d6a0941a-0e2c-459b-8415-0128cd345e69" />

---

### 8. Data Preparation for the Model  

Once the dataset has been augmented, it needs to be prepared for training the Convolutional Neural Network (CNN).  
This step ensures that all wafer maps are in the correct format, properly normalized, and split into training, validation, and test sets.  

## 8.2 Model Architecture

The convolutional neural network (CNN) used for wafer defect classification follows a structured, multi-layer design to extract features and perform classification.

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/a165ebf6-15c1-43f4-8cff-9bbcebb76f63" />

**Structure and Flow:**
- This architecture balances complexity and efficiency for wafer defect detection.  
- Convolutional blocks extract hierarchical features from wafer maps.  
- Fully connected layers combine learned features for accurate classification across all defect types.  
- The model is designed to handle imbalanced datasets while learning subtle patterns in the wafer maps.  
  
---

## Steps Required  

#### Separating Features and Labels  
- **Features (X):** Wafer map images  
- **Labels (y):** Defect category identifiers  
- This separation allows the model to learn the mapping from wafer images to defect classes.  

---

#### Resizing and Normalizing Wafer Maps  
- All wafer maps are resized to a fixed target dimension: **(25, 27)**.  
- Resizing ensures uniformity across images of varying shapes.  
- Each wafer map is normalized to pixel values between **0 and 1**, improving training efficiency and convergence.  

**Relevance/Importance:** CNNs require inputs of the same size, and normalization prevents any single feature from dominating the training process.  

---

#### Visualizing the Resizing Process  
To verify that resizing preserves important defect patterns:  

- **Before Resizing:** Wafer maps have inconsistent shapes.  
- **After Resizing:** All wafers are standardized to **25×27**, enabling batch training without losing key spatial structures.  

<img width="574" height="889" alt="image" src="https://github.com/user-attachments/assets/378aa354-db73-4bf2-9584-9f47fbfea04d" />

---

#### Train, Validation, and Test Split
The dataset is divided into:  
- **Training set:** 27,364 samples  
- **Validation set:** 6,841 samples  
- **Test set:** 34,590 samples  

**Relevance/Importance:** 
- The training set is used to fit the CNN.  
- Validation set helps tune hyperparameters and monitor overfitting.  
- The test set provides an unbiased performance evaluation.

---

## 9. Model Training

The model is trained on the processed wafer maps using supervised learning. Key techniques include:

- **Learning Rate Scheduling**  
  Adjusts learning rate dynamically to balance fast convergence and stability.

- **Class Weights**  
  Ensures minority defect classes are fairly represented, addressing dataset imbalance.

- **Early Stopping**  
  Stops training when validation loss no longer improves, avoiding overfitting.

- **Batch Training**  
  Uses mini-batches to stabilize gradient updates and improve generalization.

**Relevance/Importance:** Careful training design ensures the model learns effectively without bias or overfitting. An example is provided below for reference.

<img width="1234" height="693" alt="image" src="https://github.com/user-attachments/assets/feece42a-d8cf-42b3-8e5d-5d1b43a5d910" />

---

## 10. Training History Visualization

After training, we analyze learning curves:

- **Accuracy Curves**  
  Demonstrate how well the model performs over time for both the training and validation sets.

- **Loss Curves**  
  Indicate whether the model is converging correctly or overfitting.

**Relevance/Importance:** Visualization provides insights into training behaviour and allows early detection of issues.

<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/4a36c0d6-df69-48a9-92bc-794298b4290d" />

---

## 11. Model Evaluation

The model is tested on unseen data to measure generalization.

## 11.2 Confusion Matrix 
  Visualizes correct vs. incorrect predictions per defect class.

- **Classification Report**  
  Summarizes precision, recall, and F1-score for each class.

- **Overall Accuracy**  
  Provides a direct measure of how well the model generalizes to new wafer maps.

**Relevance/Importance:** Comprehensive evaluation ensures reliability in real-world wafer defect classification.

<img width="928" height="790" alt="image" src="https://github.com/user-attachments/assets/da1ccb81-35f2-4caa-b1a3-cb7b2650ecd3" />

---

## 12. Results & Insights

Key findings from training and evaluation:

- **High Accuracy**  
  The CNN achieved strong classification accuracy across defect types.

- **Class Imbalance Effects**  
  Minority defect classes benefited from class weighting, resulting in reduced misclassification.

- **Feature Learning**  
  CNN filters effectively captured spatial defect patterns (scratches, rings, clusters).

---

## Project Structure

**Notes:**  
- `src/` – contains the Jupyter notebook (`.ipynb`) with all code, data processing, and analysis steps.  
- `data/` – store your dataset files here. For large datasets, avoid uploading them directly to GitHub; instead, provide download links or instructions.  
- `images/` – save all plots and visualizations of wafer maps or other figures used in the README.  
- `README.md` – sits in the root folder; provides an overview of the project, methodology, results, and explanations for readers.

**Relevance/Importance:** Organizing your project in this way ensures clarity and reproducibility. It helps collaborators and reviewers quickly locate code, data, and results, and it keeps the repository tidy and professional.

---

## References



## Requirements

To run this project, the following Python packages are required:

- `tensorflow` & `keras` – for building and training the convolutional neural network  
- `numpy` – for numerical computations and array operations  
- `pandas` – for handling and manipulating datasets  
- `matplotlib` – for visualizing wafer maps and plots  
- `scikit-image` – for image processing and transformations  
- `scipy` – for interpolation and other scientific computations  
- `seaborn` – for creating advanced visualizations  

**Installation:** You are able to install all required packages using pip:

```bash
pip install tensorflow keras numpy pandas matplotlib scikit-image scipy seaborn
