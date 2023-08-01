# Uncertainty-aware Network-level traffic speed, flow, and demand prediction

This is the source code of the uncertainty-aware Network-level traffic speed, flow, and demand prediction model. This model extends the Dynamic Graph Convolution (DGC) module proposed initially by [Li et al. (2021)](https://www.sciencedirect.com/science/article/pii/S0968090X21002011). 

- Visit the DiTTlab demo page below to visualize how the model works and how the predictions are interpreted:
### [Dittlab Online Demo (click)](http://mirrors-dev.citg.tudelft.nl:8082)
- The paper manuscript is still under review.

## Requirement
* Python = 3.9
* PyTorch ≥ 2.0.1
* shapely = 1.8.5
* zarr = 2.16.0

## Data Preparation

* The used Dutch highway dataset is provided by National Data Warehouse [NDW](https://www.ndw.nu/).
* Data examples can be obtained from the DittLab application page: [tools-dittlab](https://www.tudelft.nl/citg/over-faculteit/afdelingen/transport-planning/research/labs/data-analytics-and-traffic-simulation-lab/dittlab-tu-delft/tools-1).
* Run `python get_data.py` to get the speed and flow data from the NDW server. 
* Processed data will be in the `datasets` folder.
* If there is any difficulty in preparing the dataset, please send us an email: [G.Li-5@tudelft.nl](G.Li-5@tudelft.nl). We will share the fully-processed data that is ready to use.

### Model Training

* Run the `TrainingModels.ipynb` to train the model.
* Detailed instructions are provided in the notebook.
