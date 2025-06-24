# AsRTNet
 ![image](https://github.com/piscookie/RT-descriptor-ML/blob/main/picture/P1.png)

 ├── data/                  # Data folder  
│   ├── training_data.csv  # Large training dataset  
│   ├── validation_data.csv  
│   ├── test_data.csv  
│   └── preprocessed/       # Preprocessed data folders  
│       ├── images/  
│       ├── features/  
│       └── labels/  

## Environmental requirements
pandas >= 1.5.0  
numpy >= 1.23.0  
scikit-learn >= 1.2.0  
joblib >= 1.2.0  
tqdm >= 4.64.0  
scipy >= 1.9.0  
matplotlib >= 3.5.0  
seaborn >= 0.12.0  


## Data preparation
### Training data format
The training data should include the following files:
  Main feature file: SMILES-SEED{seed}-Image_finturn.csv, containing molecular image features
Supplementary feature file:
  smiles22-SEED{seed}_Morder.csv: Mordred descriptor
  smiles22_seed{seed}_cddd.csv: Pre-trained features of CDDD
### Predicted data format
The predicted data should include the following files:
Main feature file:
  47-imagemol-Finturn.csv
Supplementary feature file:
  47-Morder.csv: Mordred descriptor
  47-cddd.csv: Pre-trained features of CDDD

## Model training
### Run the training script
python code/train.py

## New data prediction
### Run the prediction script
python code/predict.py
