# 🥔 Potato Disease Classifier

A deep learning image classification project that identifies potato leaf diseases using **ResNet18** and transfer learning.

The model classifies potato leaves into three categories:

* 🌿 Healthy
* 🍂 Early Blight
* 🍃 Late Blight

## Model

* Architecture: ResNet18
* Framework: PyTorch / fastai
* Dataset: PlantVillage
* Number of images: 2,152
* Validation accuracy: **99.07%**
* Model format: TorchScript

## Live Application

🥔 [Try the Potato Disease Classifier](https://potato-disease-classifier-lyiuksv5xfmhfrhrygcxxu.streamlit.app/)

## Model

🤗 [Hugging Face Model](https://huggingface.co/Rose-30/potato-disease-classifier)

## Blog

📝 [From Deep Learning Model to a Deployed Potato Disease Classifier 🥔](https://dev.to/rose_umutesi_86f0d45baef9/from-deep-learning-model-to-a-deployed-potato-disease-classifier-3p6f)

## Project Overview

This project demonstrates the process of taking a deep learning image classification model from training to deployment. The model was trained to recognize Healthy, Early Blight, and Late Blight potato leaves.

The trained model was exported to TorchScript and integrated into a Streamlit application. The model is hosted on Hugging Face, while the application is deployed using Streamlit Community Cloud.
