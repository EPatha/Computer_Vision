# Computer Vision Project

## **Overview**  
This repository contains code and resources for a project focused on **Computer Vision**, leveraging state-of-the-art libraries and techniques. The project explores tasks such as image processing, feature extraction, and classification, emphasizing the practical applications of face detection and recognition.

---

## **Features**  
1. **Face Detection**  
   - Uses the `dlib.get_frontal_face_detector()` for robust face detection.  
   - Processes images efficiently under various conditions like lighting and orientation.  

2. **Feature Extraction**  
   - Utilizes `dlib.face_recognition_model_v1` to extract numerical face descriptors.  
   - Features include geometric landmarks (e.g., nose width, eyebrow distance).  

3. **Data Augmentation**  
   - Techniques: flipping, brightness adjustment, shifting, and grayscale conversion.  
   - Augmentations improve model generalization on diverse datasets.  

4. **Classification**  
   - Implements Support Vector Classifier (SVC) with a linear kernel for high-dimensional data.  

---

## **Technologies Used**  
- **Python**  
- **Dlib**: Face detection and feature extraction.  
- **Scikit-learn**: Classification and accuracy measurement.  
- **Albumentations**: Data augmentation.  
- **OpenCV**: Image preprocessing and manipulation.

---

## **Setup Instructions**  
1. Clone the repository:  
   ```bash
   git clone <repository_url>
   cd computer-vision-project
