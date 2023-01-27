# Medical Segmentation README

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

### Necessary libraries
scikit-image - library for image processing
