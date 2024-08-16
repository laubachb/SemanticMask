# SemanticMask

This repository is the official implementation of SemanticMask: A Contrastive View Design for Anomaly Detection in Tabular Data. 

## Requirements

We used Python 3.9 for our experiments. The environment can be set up by installing the required packages specified in the requirements.txt file.

## Data
The Saheart dataset is already provided to demo the framework, and additional datasets can be found in Appendix F. Furthermore, the results of feature grouping for Saheart dataset using sentence-BERT and k-means, based on column names, can be found in the **group.ipynb** file.

## Code explanation
You can follow the tutorial **Tutorial_SemanticMask.ipynb** to run the code.

(1) data_loader.py
- Load and preprocess Saheart data. The preprocessed data used in paper are saved in the directory **data**.

(2) semanticmask_augmentation.py
- generate the contrastive view for SemanticMask and its variants.

(4) train.py
- Training of SemanticMask framework.
- Return the encoder part of SemanticMask framework.

(5) train_position.py
- Training of SemanticMask+position framework.
- Return the encoder part of SemanticMask+position framework.

(6) contrastive_loss.py
- Compute contrastive loss when training encoder.

(7) evaluate.py
- Evaluate the performance of SemanticMask encoder and SemanticMask+description encoder.

(8) evaluate_position.py
- Evaluate the performance of SemanticMask+position encoder.





## Pre-trained Models
Our pretrained models are saved in the directory **model**.

## Results

Our model achieves the following performance on Saheart:

| Model name         | SemanticMask   | SemanticMask+position |SemanticMask+description  |
| ------------------ |---------------- | -------------- | -------------- |
|AUCROC   |     70.4±1.0        |    70.5±2.1     |70.7±3.0      |

