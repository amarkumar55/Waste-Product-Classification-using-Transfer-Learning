###  Waste Product Classification Using Transfer Learning (VGG16)

### Project Overview
    This project implements an image classification system using transfer learning with the VGG16 architecture to classify waste materials into two categories:
       
       Organic (O)
       Recyclable (R)

    The model is trained in two stages:
       Feature Extraction using a frozen VGG16 backbone
       Fine-Tuning by unfreezing selected convolutional layers to improve performance
    
    This approach demonstrates how pre-trained deep learning models can be adapted for real-world environmental sustainability applications, such as automated waste sorting systems.

### Objectives

    By completing this project, the following objectives are achieved:
    
    Apply transfer learning using a pre-trained VGG16 model
    
    Prepare and preprocess image datasets using ImageDataGenerator
    
    Train a model using feature extraction
    
    Improve performance through fine-tuning
    
    Evaluate model performance using accuracy and classification reports
    
    Interpret model predictions on unseen test images

### Dataset

    The dataset consists of labeled waste images organized into:

  ```text
  dataset/
  ├── train/
  │   ├── O/
  │   └── R/
  ├── test/
      ├── O/
      └── R/
  ```

    Images are resized to 150 × 150
    
    Pixel values are normalized to the range [0, 1]
    
    Dataset source URL is not included for security and licensing reasons
    (a placeholder is provided in the notebook)


 ### Model Architecture
 
    Base Model
    
    VGG16
    
    Pre-trained on ImageNet
    
    include_top=False
    
    Custom Classification Head
    
    Flatten
    
    Dense (512, ReLU)
    
    Dropout (0.3)
    
    Dense (512, ReLU)
    
    Dropout (0.3)
    
    Dense (1, Sigmoid)

### Training Strategy

    Stage 1: Feature Extraction
    
    All VGG16 layers frozen
    
    Optimizer: Adam
    
    Learning rate scheduling with exponential decay
    
    Early stopping & checkpointing
    
    Stage 2: Fine-Tuning
    
    Selected layers from block5 unfrozen
    
    Optimizer: RMSprop
    
    Lower learning rate for stable fine-tuning

### Evaluation

    Models evaluated on unseen test images
    
    Metrics used:
    
    Accuracy
    
    Precision
    
    Recall
    
    F1-score
    
    Classification reports generated for:
    
    Feature Extraction model
    
    Fine-Tuned model

### Results

    Fine-tuned model outperforms the feature-extraction-only model
    
    Improved generalization on test data
    
    Demonstrates the effectiveness of transfer learning for small datasets


 ### Technologies Used

    Python
    
    TensorFlow / Keras
    
    VGG16 (ImageNet weights)
    
    NumPy
    
    Matplotlib
    
    scikit-learn


### Repository Structure

```text
  waste-product-classification-vgg16/
  │
  ├── dataset/
  │   ├── train/
  │   └── test/
  │
  ├── vgg16_feature_extraction.keras
  ├── vgg16_fine_tuned.keras
  ├── waste_classification_vgg16.ipynb
  ├── README.md
```

### Real-World Applications

    Automated waste sorting systems
    
    Smart recycling facilities
    
    Environmental monitoring solutions
    
    Industrial waste management automation

### Notes

    This project is an original implementation based on independent learning and experimentation
    
    Dataset URLs and credentials are intentionally excluded for security and compliance
    
    The notebook has been cleaned and refactored for clarity and reproducibility

👤 Author
### Amar Kumar
### AI / Machine Learning Enthusiast
