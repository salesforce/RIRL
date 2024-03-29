This repository is inteded to provide reference implementations for the work in the
 paper *Solving Dynamic Principal-Agent Problems with a Rationally Inattentive
  Principal*.
 If you make use of this code, please cite our preprint:  
 ```
@misc{mu2022rirl,
      title={Solving Dynamic Principal-Agent Problems with a Rationally Inattentive Principal}, 
      author={Tong Mu and Stephan Zheng and Alexander Trott},
      year={2022},
      eprint={2202.01691},
      archivePrefix={arXiv},
      primaryClass={cs.MA}
}

```
 
 
### RIRL-actor and agent code
The main technical contribution of this work, the RIRL-actor policy class supporting
 temporally extended settings with heterogeneous information channels, is implemented
  in ```multi_channel_RI.py```. Other supporting implementations of trainable modules
   are organized within the ```agents/``` directory.
   
### Experiment code
This repository includes reference training pipelines for each of the two
 experimental settings:
 - Payment schedule optimization in a single-step ("bandit") setting.
 - Wage optimization in a multi-step, multi-Agent, multi-channel setting.
 
These reference pipelines are organized into jupyter notebooks:
 - ```PA_bandit_game/PA_Bandit-training_pipeline_reference.ipynb``` (payment
  optimization)
 - ```PA_multiagent_game/multiagent_signaling-training_pipeline_reference.ipynb``` (wage optimization) 
 
Each notebook is structured to exemplify a single training run. Static
 hyperparameters are defined in associated config YAML files. Important experimental
  variables (such as mutual information costs) are set within the notebooks. In this
   way, the notebooks can be run repeatedly to manually perform sweeps over variables of
    interest. 
