# StatAssist & GradBoost: Object Detection
StatAssist & GradBoost: A Study on Optimal INT8 Quantization-aware Training from Scratch


## Version
Training tested on Nightly build torch 1.5.0, Conversion tested on Nightly build torch 1.6.0 
See detailed settings in requirements.txt
 
## Training Operation
We only support VOC for trianing the data.
See arguments for the detailed setting.

## QNNPACK convert and testing
You can use qeval_convert.py for testing converted version. Just type
```
python3 qeval_convert.py --net_type tdsod
```
To test the ssd, type 'ssd' instead of 'tdsod'. 





