# Multi-Dimensionality of Aging

Source code of the publication **“Analyzing the multidimensionality of biological aging with the tools of deep learning across diverse image-based and physiological indicators yields robust age predictors”**.

The source code contains only the part refering to Tabular Data : 
- the XWAS pipeline (X-Wide Association Study) 
- the Aging pipeline corresponding to the Tabular Biomarkers (e.g. blood biomarkers, anthropometric measures, ...)

More details and results can be found on the paper # LINK or on the website https://www.multidimensionality-of-aging.net/

## Description of the source code.
The code is separated in two parts : 
- The first part **aging** groups the scripts of the different experiments.
- The second part **batch_jobs** groups the source code of the experiments.

The code is able to run the experiments for the XWAS and also for the Aging project.

They both use the file **general_predictor.py** which is the core of the code. This file creates estimators and train them accordingly using a nested cross validation. (More details can be found on the paper).
Then, depending to whether we want to run an XWAS experiment or an Aging experiment, we build associated classes : 

- EnvironmentPredictor for the XWAS
- SpecificPredictor for the Aging 





