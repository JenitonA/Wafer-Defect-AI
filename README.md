# Wafer-Defect-AI

## Project Overview
This project leverages a **Convolutional Neural Network (CNN)** to detect and classify defects on semiconductor wafers.  
The notebook walks through **data loading, preprocessing, model building, and visualization**, preparing for training a defect classification model.  

### Why This Matters
Semiconductor wafers are the foundation of modern electronics.  
Detecting defects early:  
- Improves manufacturing yield  
- Reduces costs from faulty chips  
- Ensures higher product reliability  

Automating defect detection with deep learning helps scale up quality control advancement and allows for prolonging prime efficiency within semiconductors.

---

## Key Features
- Load and process wafer map data from the **LSWMD dataset**  
- Define a CNN architecture for defect classification  
- Convert labels into numerical form for modelling  
- Visualize wafer maps by defect type using custom colormaps  
- Outline next steps for training and evaluation  

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
