# Hierarchical_HAR-PBD



## Code Description
Each file contains 3 py files.
- HierarchicalHAR_PBD.py is the code for model construction
- main.py is the code for data pre-processing and main functions including model training. 
- utils.py  is the code tool functions.

## Model Description
- Baselinde model is the Hierarchical HAR_PBD model only utilize MoCap data. 
- Early Fusion model fuse MoCap and EMG in the early stage and build a new graph including 26 nodes.
- Late Fusion model fuse MoCap and EMG in the prediction level, MoCap and EMG are built as graph respectively and sent to each GCN network.
- Central Fusion model fuse MoCap and EMG in the GCN hidden layers.
- Central Fusion model with attention is build based on the Central Fusion model, while attention mechanism is implemented in the GCN networks. It achieves the best performance regarding to all the indicators including Accuracy, Macro.F1, PR-AUC and confusion matrix.
