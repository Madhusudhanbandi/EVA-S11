# EVA-S11
Implementation of given below ResNet Architecture and OneCycleLR scheduler and plotting Cyclic LR plot targetting 90% accuracy

Setps:
1.Using of Padding,Flip,Rancomcrop,Cutout transforms
2.Implementation given Architecture
3.Finding max_lr using range test
4.Training network with OneCycle policy for 24 epochs and targetting 90% test accuracy
5.Plotting of Cyclic LR plot between max and min LR by using max_lr, min_lr, stepsize and cycle 

Architecture:

![Resnet_Architecture](https://user-images.githubusercontent.com/19210895/79043757-e6b97b80-7c1e-11ea-8d1c-be289472d4fb.JPG)


Cyclic LR plot

![Triangular plot](https://user-images.githubusercontent.com/19210895/79043785-15cfed00-7c1f-11ea-9aa8-819d2ac0d07d.JPG)

