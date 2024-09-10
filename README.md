# Machine Learning and Deep Learning Projects

## Project Overview
This repository contains a collection of machine learning and deep learning projects, each addressing different problem domains. The projects cover a range of topics including text preprocessing with autoencoders, image classification, segmentation, object detection, and multi-input models. These notebooks demonstrate various architectures and methodologies for tackling both computer vision and natural language processing tasks.

## Files in the Repository

1. **Autoencoding-Preprocessing-Text.ipynb**
   - Demonstrates how to use autoencoders for preprocessing text data. This notebook covers the steps for dimensionality reduction and encoding text sequences, which can be useful for downstream tasks like classification or clustering.

2. **Cat-vs-Dog-Image-Classification.ipynb**
   - Implements an image classification model to distinguish between cats and dogs. It uses Convolutional Neural Networks (CNN) for feature extraction and classification, showcasing a typical binary image classification pipeline.

3. **Flood-Image-Segmentation-UNET.ipynb**
   - Implements the UNET architecture for flood-image segmentation tasks. The model is trained to identify different flood regions within an image, focusing on accurate pixel-level segmentation.

4. **Kannada-MNIST-Digit-Recognizer-CNN.ipynb**
   - Builds a CNN model to classify Kannada digits from the Kannada-MNIST dataset, similar to the original MNIST dataset but involving a different character set. This project focuses on handwritten digit recognition.

5. **Multi-Input-Model-Classification.ipynb**
   - Demonstrates the use of a multi-input model for classification tasks, where the model takes both image data and metadata as inputs to make predictions. This notebook explores how to handle and combine different data types in a single model.

6. **Object-Detection-YOLO-Structure.ipynb**
   - Implements the structure of the YOLO (You Only Look Once) object detection algorithm. The notebook focuses on the model's architecture and how it can be used to detect objects in real time within an image.

7. **Text-Classification-CNN.ipynb**
   - Demonstrates the use of Convolutional Neural Networks (CNNs) for text classification. The notebook shows how CNNs, traditionally used for images, can be adapted to process text data for tasks such as sentiment analysis or topic classification.

8. **leaf-multi-input-model-classification.ipynb**
   - Similar to the multi-input model notebook, this project focuses on classifying leaf species using both image data and metadata. It explores the performance of multi-input models in this specific use case.

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/sohkachun/computer-vision.git
   cd machine-learning-projects
