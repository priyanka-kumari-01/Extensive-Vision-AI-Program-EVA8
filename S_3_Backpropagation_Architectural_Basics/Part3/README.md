# Session 3 - Backpropagation and Architectural Basics

![Task](https://raw.githubusercontent.com/pandian-raja/EVA8/main/Resources/S3/s3_part_3.png)


---

## [Colab Link](EVA4_Session_3.ipynb)

##  Model Summary: 

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 26, 26]              80
              ReLU-2            [-1, 8, 26, 26]               0
       BatchNorm2d-3            [-1, 8, 26, 26]              16
            Conv2d-4           [-1, 16, 24, 24]           1,168
              ReLU-5           [-1, 16, 24, 24]               0
       BatchNorm2d-6           [-1, 16, 24, 24]              32
         MaxPool2d-7           [-1, 16, 12, 12]               0
            Conv2d-8           [-1, 32, 10, 10]           4,640
              ReLU-9           [-1, 32, 10, 10]               0
      BatchNorm2d-10           [-1, 32, 10, 10]              64
          Dropout-11           [-1, 32, 10, 10]               0
           Conv2d-12             [-1, 16, 8, 8]           4,624
             ReLU-13             [-1, 16, 8, 8]               0
      BatchNorm2d-14             [-1, 16, 8, 8]              32
AdaptiveAvgPool2d-15             [-1, 16, 6, 6]               0
           Linear-16                   [-1, 10]           5,770
================================================================
Total params: 16,426
Trainable params: 16,426
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.48
Params size (MB): 0.06
Estimated Total Size (MB): 0.54
----------------------------------------------------------------

```

---

## Result:


Epoch:  16
loss=0.0005416961503215134 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 30.43it/s]

**Test set: Average loss: 0.0203, Accuracy: 9940/10000 (99.4000%)**

---

## Logs:

```
Epoch:  1
  0%|          | 0/469 [00:00<?, ?it/s]<ipython-input-8-14c2a1af8d27>:33: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
  return F.log_softmax(x)
loss=0.03100750781595707 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 30.79it/s]

Test set: Average loss: 0.0500, Accuracy: 9849/10000 (98.4900%)

Epoch:  2
loss=0.006264335010200739 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 29.35it/s]

Test set: Average loss: 0.0449, Accuracy: 9870/10000 (98.7000%)

Epoch:  3
loss=0.016612181439995766 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 30.99it/s]

Test set: Average loss: 0.0301, Accuracy: 9903/10000 (99.0300%)

Epoch:  4
loss=0.012095619924366474 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 30.90it/s]

Test set: Average loss: 0.0257, Accuracy: 9915/10000 (99.1500%)

Epoch:  5
loss=0.01567385531961918 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 30.88it/s]

Test set: Average loss: 0.0242, Accuracy: 9920/10000 (99.2000%)

Epoch:  6
loss=0.015753788873553276 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 30.90it/s]

Test set: Average loss: 0.0243, Accuracy: 9920/10000 (99.2000%)

Epoch:  7
loss=0.022013826295733452 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 31.38it/s]

Test set: Average loss: 0.0226, Accuracy: 9927/10000 (99.2700%)

Epoch:  8
loss=0.020139100030064583 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 31.17it/s]

Test set: Average loss: 0.0235, Accuracy: 9925/10000 (99.2500%)

Epoch:  9
loss=0.023578651249408722 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 30.80it/s]

Test set: Average loss: 0.0216, Accuracy: 9923/10000 (99.2300%)

Epoch:  10
loss=0.014694436453282833 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 30.95it/s]

Test set: Average loss: 0.0250, Accuracy: 9916/10000 (99.1600%)

Epoch:  11
loss=0.01405383925884962 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 30.63it/s]

Test set: Average loss: 0.0212, Accuracy: 9934/10000 (99.3400%)

Epoch:  12
loss=0.07690025120973587 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 31.17it/s]

Test set: Average loss: 0.0209, Accuracy: 9935/10000 (99.3500%)

Epoch:  13
loss=0.05652768909931183 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 29.56it/s]

Test set: Average loss: 0.0199, Accuracy: 9930/10000 (99.3000%)

Epoch:  14
loss=0.0005885807331651449 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 30.37it/s]

Test set: Average loss: 0.0221, Accuracy: 9926/10000 (99.2600%)

Epoch:  15
loss=0.011556155048310757 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 31.27it/s]

Test set: Average loss: 0.0196, Accuracy: 9939/10000 (99.3900%)

Epoch:  16
loss=0.0005416961503215134 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 30.43it/s]

Test set: Average loss: 0.0203, Accuracy: 9940/10000 (99.4000%)

Epoch:  17
loss=0.0049404301680624485 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 30.13it/s]

Test set: Average loss: 0.0198, Accuracy: 9935/10000 (99.3500%)

Epoch:  18
loss=0.0112218102440238 batch_id=468: 100%|██████████| 469/469 [00:15<00:00, 31.16it/s]

Test set: Average loss: 0.0206, Accuracy: 9928/10000 (99.2800%)

Epoch:  19
loss=0.0018511704402044415 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 31.30it/s]

Test set: Average loss: 0.0207, Accuracy: 9932/10000 (99.3200%)

Epoch:  20
loss=0.0015828651376068592 batch_id=468: 100%|██████████| 469/469 [00:14<00:00, 31.51it/s]

Test set: Average loss: 0.0208, Accuracy: 9936/10000 (99.3600%)

```