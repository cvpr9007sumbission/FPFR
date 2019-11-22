# Learning Fused Pixel and Feature-based View Reconstructions for Light Fields (FPFR)
<br>
CVPR submission ID 9007
<br>
This project includes a demo version of our program demonstrating how our model interpolates or extrapolates a light field based on 2x2 sparsely sampled views. We strongly recommand that the readers follow the instructions below to synthesize a light field by yourself.

## Check environment
Our project is implemented in **Python** with the deep learning framework **tensorflow (version 1.13.1)**

Before launching the demo code, please make sure that the following packages are correctly installed.   
Check list: **tensorflow**,**numpy**,**matplotlib**

## Launch demo
We provide several light field scenes to test our demo code. Please run the `run_demo.sh` script to launch our demo. The command in `run_demo.sh` is:  
`python demo.py --scene_name=stilllife --mode=FPFR* --data_type=synthetic --angular_resolution=7 --inter_extra=inter`  
- ***scene_name*** is the name of the light field scene. 
- ***mode*** offers two mode, FPFR and FPFR*  (more details in our paper). Default: FPFR*. 
- ***data_type*** indicates the type of light field data (synthetic or lytro). Default: synthetic.
- ***inter_extra*** indicates view interpolation or extrapolation (inter or extra). Default: inter.
- ***angular_resolution*** is the angular resolution of the generated light field. In the inter mode, angular resolution should be an integer greater or equal to 3; in the extra mode, angular resolution should be an odd integer greater than 3. Default value: 7.

## Other informations
We offered several test scenes in the folder **scenes**. The synthesized light fields can be found in the folder **results**.
Thank you for your attention, we sincerely hope you enjoy our project.  
