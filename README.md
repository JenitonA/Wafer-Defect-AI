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

The `build_model()` function constructs a convolutional neural network (CNN) specifically designed for wafer map classification. The architecture includes:  

**Initial Convolutional Layer** – applies 16 filters to capture simple features such as edges and shapes.  
**Block 1** – a convolutional layer with 64 filters, followed by batch normalization and max pooling to stabilize training and reduce spatial size.  
**Block 2** – another convolutional layer with 128 filters, also paired with normalization and pooling, allowing the model to learn more detailed structures.  
**Global Average Pooling** – condenses feature maps into a compact representation, reducing parameters and mitigating overfitting.  
**Fully Connected Layer** – a dense layer with 128 neurons, batch normalization, and dropout for stronger generalization.  
**Output Layer** – a softmax layer that produces class probabilities across the defect categories.  

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/a165ebf6-15c1-43f4-8cff-9bbcebb76f63" />

**Relevance/Importance:**  
This section defines the backbone of the project. By stacking convolutional, pooling, and dense layers, the model can progressively extract meaningful patterns from wafer images and classify them into defect types with high accuracy.  

---

### 3. Build and Summarize Model  

The model is created using the specified input shape of **42×42×1** (grayscale wafer images) and **9 output classes** (the defect categories). Once built, the summary provides a layer-by-layer breakdown:  

**Conv2D + Batch Normalization** – first layers extract low-level features and normalize activations for more stable training.  
**Conv2D + Batch Normalization + MaxPooling (Block 1)** – captures more complex spatial patterns while reducing dimensionality.  
**Conv2D + Batch Normalization + MaxPooling (Block 2)** – learns high-level structures such as distinct defect shapes.  
- **Global Average Pooling** – compresses feature maps into a compact representation, lowering parameter count.  
- **Dense (128 units) + Batch Normalization + Dropout** – combines extracted features while controlling overfitting.  
- **Final Dense (9 units, softmax)** – outputs probabilities for each defect class.  

The summary also reports the total number of parameters (**102k**), showing the model is complex enough to learn meaningful wafer features but still efficient for training.  

**Relevance/Importance:**  
Summarizing the model confirms that the architecture matches the design intent. It provides transparency into how many parameters must be trained, helping assess model size, training requirements, and potential risks of overfitting.  

---

### 4. Load Data

- The wafer dataset (`LSWMD.pkl`) is loaded with `pandas`.  
- Displaying the first rows confirms that the data loaded correctly.  

**Relevance/Importance:** You need to verify that your dataset is available and structured correctly before any analysis.

---

### 5. Preprocessing and Cleaning

- Removed irrelevant columns such as `waferIndex` and `lotName` that are not useful for classification.  
- Corrected the mislabeled `trianTestLabel` column to `trainTestLabel` for consistency.  
- Converted categorical defect labels into numerical values (`failureNum`) and assigned numerical values to the training/test split (`trainTestNum`).  
- Separated unlabeled wafers (≈638k) from labeled wafers (≈173k) to ensure only usable data is included in training and evaluation.  

**Relevance/Importance:** Preprocessing standardizes the dataset so the CNN can interpret it correctly. Removing irrelevant columns avoids noise, encoding labels into numbers makes the data machine-readable, and splitting labeled from unlabeled wafers prevents training issues while preserving the option for semi-supervised learning later.

---

### 6. EDA and Visualization

- Visualized sample wafer maps for each defect type using a custom colour scheme (black = background, green = good die, red = defective die). This provided an intuitive look at how different defect patterns appear on wafers.  
- Analyzed wafer map shapes by computing aspect ratios to detect irregular or distorted maps. Wafers with aspect ratios greater than 1.3 (≈2.3% of the labelled set) were identified as outliers and removed.  
- Standardized all wafer maps by resizing them to 42×42 pixels to ensure consistent input dimensions for the CNN.  
- Split the cleaned dataset into training (85%) and testing (15%) subsets, stratified by defect type to preserve class distribution.  

<img width="1027" height="630" alt="image" src="https://github.com/user-attachments/assets/53d5e8d3-f556-4d4a-a866-739fcb6ed098" />

<img width="1490" height="622" alt="image" src="https://github.com/user-attachments/assets/8e8669e1-5a8d-49e2-b103-2599e5e5b98a" />

<img width="781" height="403" alt="image" src="https://github.com/user-attachments/assets/46492482-528e-4e4d-9c55-ca40e499ca2d" />

**Relevance/Importance:** EDA highlights data quality issues and ensures the dataset is clean, balanced, and standardized. By filtering out abnormal wafers and resizing inputs, the model is trained on consistent, reliable data. Splitting into training and testing sets enables unbiased performance evaluation.

---

### 7. Data Augmentation  

The original dataset was highly imbalanced, with almost 70% of the samples labelled as no visible defect. If left uncorrected, the model would become biased toward predicting this dominant class and fail to recognize less common but critical defect types.  

To correct this, the testing data was adjusted by reducing the number of “none” samples so that the evaluation would better reflect performance across all classes. For the training data, two strategies were combined:  

**Downsampling** was applied to classes with too many examples.  
**Data augmentation** techniques such as rotations, flips, scaling, shifting, and affine transformations were used to expand underrepresented classes, creating new but realistic wafer patterns.  

After balancing, each defect type contained 2,000 samples, resulting in a uniform training set of 18,000 wafers.  

**Relevance/Importance:** Balancing and augmentation prevent the network from overfitting to the most common class and force it to learn meaningful features across all defect types. This step improves accuracy, fairness, and generalization of the final CNN.

---

### 8. Split Training Data

To evaluate the model correctly, the labelled dataset was split into three groups:  

**Training set (80%)** – used to teach the CNN the features of each wafer defect.  
**Validation set (20%)** – held out during training to tune hyperparameters and check for overfitting.  
**Independent test set** – completely separate from training and validation, reserved for the final performance evaluation.  

All wafer maps were reshaped into 4D arrays (samples × height × width × channel) to match the CNN input format of **(26×26×1)**. Corresponding defect labels were also converted into numerical arrays for compatibility with the model.  

After splitting and formatting, the data was ready to be fed into the neural network.  

**Relevance/Importance:** This separation ensures the model is evaluated fairly. Using a validation set prevents overfitting, while the independent test set provides an unbiased estimate of real-world performance.

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
