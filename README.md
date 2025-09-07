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

### 9. Model Training  
The model was trained using the augmented wafer datasets with a strategy designed to encourage stable learning and prevent overfitting.  

**Epochs:** 20  
**Batch size:** 64  
**Optimizer:** Adam  
**Initial learning rate:** 0.001, reduced by half every five epochs  
**Class weights:** applied to account for class imbalance  
**Early stopping:** monitored the validation loss, stopping training if no improvement was seen within three epochs and restoring the best weights  

This training approach combined multiple safeguards. The step-based learning rate schedule allowed the optimizer to take larger steps early on and then refine the updates as training progressed. Early stopping prevented the model from continuing once validation performance stopped improving, which reduces overfitting. Balancing the class weights ensured that minority defect categories had an equal influence during training.  

Over the course of training, accuracy increased consistently while validation loss decreased, with the final model achieving close to **88% validation accuracy**. This demonstrates that the network was able to learn distinctive features across the full range of defect types rather than over-relying on the dominant “none” category.  

---

### 10. Visualizing Training History  
To evaluate how the model improved during training, the accuracy and loss values for both training and validation sets were plotted across all epochs.  

**Accuracy Curve:** Showed a steady increase in both training and validation accuracy, indicating that the model was successfully learning discriminative features over time.  
**Loss Curve:** Demonstrated a clear downward trend for training loss, while validation loss also decreased overall with some fluctuations. This behaviour suggests the model generalized well without severe overfitting.  

By examining these plots, it became clear that the learning rate schedule and early stopping mechanisms worked effectively, allowing the model to converge while maintaining good validation performance. The final curves reflected a strong balance between training and validation metrics, supporting the robustness of the trained CNN.  

<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/5270f61b-ff54-4ac0-b312-86caf43baf59" />

---

## 11. Model Evaluation  

After training the CNN, we evaluated its performance on the test set to assess how well it generalizes to unseen wafer maps. The final test results were:  

**Test Loss:** 0.4318  
**Test Accuracy:** 86.74%  

These results indicate that the model is capable of correctly identifying wafer defects in the majority of cases while maintaining a low loss, demonstrating good generalization.

### 11.2 Confusion Matrix

To understand how the model performs across different defect types, we analyzed its predictions using a normalized confusion matrix and standard classification metrics such as precision, recall, and F1 score.  

Key observations:  
- The model achieved high accuracy on most classes, with "Edge-Ring" and "Near-full" showing near-perfect identification.  
- Classes with fewer samples, such as "Scratch" and "Loc," showed slightly lower precision or recall, reflecting the challenges of limited data for these defect types.  
- The overall weighted F1 score was **0.87**, indicating balanced performance across all classes.  

The normalized confusion matrix provides a visual representation of correct and incorrect predictions, highlighting the model's strengths and areas for improvement. Comparing the first 100 predictions with their proper labels further confirms the model reliably distinguishes between defect types while maintaining robust performance on the majority "none" class.

This analysis demonstrates that the model can be used confidently for automated wafer defect classification, and it also provides guidance for future enhancements through targeted data augmentation or additional training on underrepresented classes.

<img width="924" height="790" alt="image" src="https://github.com/user-attachments/assets/6cad36d2-6d08-4ce1-bb7d-fec28bb3b3ba" />

---

## 12. Save the Trained Model

After completing training and evaluation, the final step is to save the trained CNN model. Saving the model allows it to be reused later without the need to retrain, which is especially useful for deployment or future experiments.

The model is stored in a dedicated `models` directory in the project. Once saved, it can be easily loaded for inference or further fine-tuning.  

**Saved Model Path:** `..\models\wafer_defect_model.keras`  

This ensures that the trained network, along with its learned weights and architecture, is preserved for future use.

---


## Project Structure

**Notes:**  
- `src/` – contains the Jupyter notebook (`.ipynb`) with all code, including data preprocessing, visualization, augmentation, model training, and evaluation steps.  
- `data/` – store your dataset files here. For large datasets, avoid uploading them directly to GitHub; instead, provide download links or instructions for obtaining the data.  
- `images/` – save all plots, wafer map visualizations, and figures used in the README, such as training history graphs, confusion matrices, and sample wafer maps.  
- `models/` – stores the trained CNN model (`wafer_defect_model.keras`) for later use without retraining.  
- `README.md` – sits in the root folder; provides a project overview, methodology, results, and explanations of the workflow and model performance.

**Workflow Summary:**  
**Preprocessing and Cleaning** – Prepare wafer maps, remove outliers, and convert categorical labels into numerical values.  
**Exploratory Data Analysis (EDA)** – Visualize wafer defects, resize wafer maps to a consistent shape, and analyze class distributions.  
**Data Augmentation** – Apply flips, rotations, and affine transformations to balance minority classes; downsample the majority `none` class for the test set.  
**Training-Validation Split** – Split the augmented dataset into 80% training and 20% validation; prepare data arrays for CNN input.  
**Model Training** – Train a CNN using Adam optimizer with learning rate scheduling, early stopping, and class weights to handle imbalance.  
**Visualize Training History** – Plot training and validation accuracy and loss over epochs to assess learning behavior.  
**Model Evaluation** – Evaluate the model on an independent test set, compute test loss, accuracy, F1-score, and generate a confusion matrix.  
**Save Trained Model** – Save the final trained model to `models/` for future use without retraining.

**Relevance/Importance:**  
Organizing the project this way ensures reproducibility, clarity, and professionalism. It allows collaborators or reviewers to quickly locate code, datasets, and results while keeping the repository clean and structured.

---

## References

[1] S. Biswas, D. A. Palanivel, and S. Gopalakrishnan, "A Novel Convolution Neural Network Model for Wafer Map Defect Patterns Classification," in *Proc. IEEE TENSYMP*, Jul. 2022, pp. 1–6, doi: 10.1109/TENSYMP.2022.XXXXXXX.


---

## Requirements

To run this project, you will need the following Python packages:

- `tensorflow` & `keras` – for building, training, and evaluating the convolutional neural network (CNN).  
- `numpy` – for numerical computations, array manipulation, and mathematical operations.  
- `pandas` – for managing and processing structured datasets.  
- `matplotlib` – for visualizing wafer maps, training history, and other plots.  
- `scikit-image` – for image processing, resizing, and augmentation of wafer maps.  
- `scipy` – for scientific computations and interpolation operations.  
- `seaborn` – for advanced visualization, including heatmaps and statistical plots.

**Installation:** You can install all required packages at once using pip:

```bash
pip install tensorflow keras numpy pandas matplotlib scikit-image scipy seaborn
