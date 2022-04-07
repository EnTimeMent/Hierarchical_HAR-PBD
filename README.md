# Hierarchical_HAR-PBD



## Code Description
Each file contains 3 py files.
- HierarchicalHAR_PBD.py is the code for model construction
- main.py is the code for data pre-processing and main functions including model training. 
- utils.py  is the code tool functions.


Baselinde model is the Hierarchical HAR_PBD model only utilize MoCap data. 
Early Fusion model, Late Fusion model and Central Fusion model utilize MoCap data and EMG data with respective fusion strategies. 
Central Fusion model with attention is build based on the Central Fusion model, while attention mechanism is implemented in the GCN networks. It achieves the best performance regarding to all the indicators including Accuracy, Macro.F1, PR-AUC and confusion matrix.
