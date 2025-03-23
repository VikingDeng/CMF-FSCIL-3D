# CMF-FSCIL-3D: Cross-Modal Few-Shot Class-Incremental Learning for 3D Point Clouds

## Project Introduction
This project implements cross-modal learning for 3D point cloud recognition based on the research paper 
[On the Cross-Modality Fantasy for 3D Point Clouds Few-Shot Class-Incremental Learning]. 
Using MindSpore, we've developed a system that leverages pseudo-class data generation and CLIP's visual-linguistic 
capabilities to improve recognition models like PointNet, enabling effective few-shot class-incremental learning.

## Overview
Our approach consists of three key stages:
1. **Pseudo-class Data Generation**: Create synthetic data from public datasets (ShapeNet, ModelNet, etc.)
2. **Cross-modal Contrastive Learning**: Utilize CLIP's powerful visual-linguistic understanding
3. **Incremental Learning**: Implement few-shot class-incremental learning with an NCM classifier

This methodology significantly enhances the recognition capabilities of 3D point cloud models in scenarios with limited labeled data.

## Usage

### Project Setup
1. Download and extract the project files
2. Ensure Python 3.9 is installed
3. Install dependencies with:
```bash
pip install -r requirements.txt
```

### Data Preparation
- Download PC data to `CMF-FSCIL-3D/data/shapent`:
  [Download Link](https://pan.baidu.com/s/1fVJwG8MQonTMj-GrLpmWGQ?pwd=7qnd)
  
- Download pretrained checkpoint to `CMF-FSCIL-3D/savemodel`:
  [Download Link](https://drive.google.com/file/d/1iwiAkEPTjRMMUHdR0odYAoe99CBlTLBD/view?usp=drive_link)

### Run Preparation
```bash
python main.py
```
Test results will be stored in the `log` folder.

## Notes
Currently, the open-source portion of this project focuses on incremental learning training and testing using the base class-trained model. The complete codebase will be released in the future.