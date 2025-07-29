# Dynamic Hyperparameter Adjustment via Reinforcement Learning in Asynchronous Federated Learning for Medical Image Analysi


This example uses 2D (axial slices) segmentation of the prostate in T2-weighted MRIs based on multiple datasets.

Please refer to [Prostate Example](https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/prostate) for details of data preparation and task specs. 
## Setup

Install required packages for training
```
pip install --upgrade pip
pip install -r ./requirements.txt
```

## Run automated experiments
We use the NVFlare simulator to run FL training automatically, the 6 clients are named `client_I2CVB,client_MSD,client_NCI_ISBI_3T,client_NCI_ISBI_Dx,client_Promise12,client_PROSTATEx`.

### Prepare local configs
First, we add the image directory root to `config_train.json` files for generating the absolute path to dataset and datalist.  


### Use NVFlare simulator to run the experiments
We use NVFlare simulator to run the FL training experiments. In this example, we run six clients on 2 GPUs.  We put the workspace in `/tmp` folder
```
nvflare simulator jobs/afedrl_prostate -w /tmp/nvflare/afedrl_prostate -c client_I2CVB,client_MSD,client_NCI_ISBI_3T,client_NCI_ISBI_Dx,client_Promise12,client_PROSTATEx -gpu 0,1,0,1,0,1
```



