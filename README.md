# Code base for DiffStitch:


### train Dynamic models 
    cd scripts/mopo 
    python train.py 

### generate stitching transitions: 
    cd scripts
    python batch_stitch.py --config=hopper_medium_replay_v2 


### ERROR & BUGS
- from tap import Tap  
    Install this package 
    ```python 
    pip install typed-argument-parser
    ```