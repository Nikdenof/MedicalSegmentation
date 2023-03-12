# Medical Segmentation 

## Requirements
* scikit-image >= 0.20.0
* PyTorch >= 1.2
* Python >= 3.5
TBD

## Dataset 
The data for this project was obtained from the official site of the ADDI Project.
https://www.fc.up.pt/addi/ph2%20database.html

## Features
* SegNet model, implemented from scratch, available to use in any of your project using Pytorch Hub
`torch.hub.load('Nikdenof/MedicalSegmentation', 'testmodel')`
* 3 different types of loss functions for binary segmentation, implemented from scratch using Pytorch
* Train and test code with visualization of the segmentation
* Jupyter Notebook containing the comparison between two different segmentation architectures

## Folder Structure
TBD

## Usage
At this point you can run the code in this repo using Jupyter Notebook titled SegmentationScreening.ipynb
Train and test python scripts are still in development.

## Customization
TBD

## Report

In this report, we present the work done on the task of deep neural network segmentation. Two models were implemented: SegNet and U-net. The main focus of the report is to compare the results of these models and to identify the best performing model.

### Methods

- SegNet was implemented using indices from the Maxpool layers to obtain an activation map with HxW sizes corresponding to the input image at the output. A bottleneck of the last block of VGG layers was used, along with a mirror decoder, maxpooling, and maxunpooling.
- U-net was implemented using a transposed convolution as an upsampler. To obtain an activation map with HxW dimensions at the output, the Overlap-tile strategy was used. A mirroring of the borders using nn.ReflectionPad2d was applied before each convolution encoder and decoder as pooling.

### Results

The results of the two models were compared using various metrics such as accuracy, precision, and recall. The results were also visualized using graphs of losses and metrics on validation and test sets.

### Comparison

The comparison of the two models showed that U-net performed better in terms of overall quality. However, SegNet had a faster training time and used fewer model parameters.

### Conclusion

Based on the results, U-net is the best performing model for this task. However, if faster training time and fewer model parameters are a priority, SegNet may be a suitable alternative. The report includes a detailed comparison of the models, as well as the choice of the best models from several clusters and the choice of the overall best model. Additionally, a small conclusion throughout the report is included.


## TODOs
- [x] Implement SegNet architecture from scratch using PyTorch, in separate file
- [x] Implement different loss function for binary segmentation tasks using PyTorch
- [ ] Make train.py and test.py for more readable, reproducable code
- [ ] Remake the folder structure to make it more clear and reusable
- [ ] Publish the model and relevant tools using PyTorch Hub
- [ ] Make a requirements.txt file to make the code more reproducable
- [ ] Make a config.json file for convinient parameter tuning
- [ ] Checkpoint saving and resuming
- [ ] Make comprehensible logging system
- [ ] Customizable command line options for train.py
- [ ] Deploy the model using Flask
- [ ] Check the code using Flake8



## License
This project is licensed under the MIT License. See  LICENSE for more details.

## Acknowledgements
TBD
