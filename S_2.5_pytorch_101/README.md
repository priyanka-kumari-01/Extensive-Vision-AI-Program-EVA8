# EVA8

## Session 2.5 PyTorch 101

### Task: 

![Alt text](https://user-images.githubusercontent.com/5630870/211112238-b6512297-f5e5-4103-be3c-15577b09cfc4.png)


### Input Description:

1. Generating MNIST images:  I have downloaded them from torchvision. 

```python 

   self.data = torchvision.datasets.MNIST('/content/mnist', train=False, download=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
```        
2. Generating Random numbers: I have created a 1D torch tensor. With a random number, I used eq() to get 1D True/False values and converted them to long int.

```python 

randomNumber = torch.tensor([0,1,2,3,4,5,6,7,8,9])
self.random = torch.randint(0, 10, (len(self.data),))
randomInput = randomNumber.eq(self.random[index]).long()
```

The Dataset implementation as follows: 

```python 

class TrainDataSet(Dataset):
    def __init__(self):
        self.data = torchvision.datasets.MNIST('/content/mnist', train=True, download=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
        self.random = torch.randint(0, 10, (len(self.data),))
        
    def __getitem__(self, index):
        image, label = self.data[index]
        randomInput = randomNumber.eq(self.random[index]).long()
        return image, label, randomInput

    def __len__(self):
        return len(self.data)
```

### Model Architecture: 
In the below summary, The output of fc2 is 10 channels, and I returned it for mnist prediction, Also added with random input and performed two fully connected layers and returned 19 channels.  

```
Network(
  (conv1): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1))
  (conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1))
  (conv3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
  (conv4): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
  (fc1): Linear(in_features=10368, out_features=64, bias=True)
  (fc2): Linear(in_features=64, out_features=10, bias=True)
  (fc3): Linear(in_features=10, out_features=20, bias=True)
  (out): Linear(in_features=20, out_features=19, bias=True)
)
```

```python 
class Network(nn.Module):

    def __init__(self):
        super().__init__()
        #3x3: SIZE { in = 28 , out= 26}, recptive field = 3, CHANNEL { in = 1, out= 16 }
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3)
        #3x3: SIZE { in = 26 , out= 24}, recptive field = 35, CHANNEL { in = 16, out= 32 }
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3)
        #3x3: SIZE { in = 24 , out= 22}, recptive field = 7, CHANNEL { in = 32, out= 64 }
        self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3)

        #MX: SIZE { in = 22, out = 11},  recptive field = 14, CHANNEL { in = 64, out= 64 } Implemented in forward()

        #3x3: SIZE { in = 11 , out= 9}, recptive field = 17, CHANNEL { in = 64, out= 128 }
        self.conv4 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3)

        self.fc1 = nn.Linear(in_features=128*9*9,out_features=64)
        self.fc2 = nn.Linear(in_features=64,out_features=10)

        self.fc3 = nn.Linear(in_features=10,out_features=20)
        self.out = nn.Linear(in_features=20,out_features=19)

    def forward(self, image, random):

        x = image
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2) 
        x = self.conv4(x)
        x = F.relu(x)
        x = x.reshape(-1, 128*9*9)
        # x = x.reshape(1, -1)
        x = self.fc1(x)
        x = F.relu(x)
        # Now we have 10 channels and return for mnist and add with random  
        mnist_output = self.fc2(x)
        # Add mnist_output+random
        x = mnist_output+random
        # Expand channel to 20 
        x = self.fc3(x)
        x = F.relu(x)
        # Maximum value we get by adding label and random is 19. So made the output as 19 channel 
        x = self.out(x)
        mnist_output = F.softmax(mnist_output, dim=1)
        x = F.softmax(x, dim=1)
        return mnist_output, x
``` 

### Result Evaluation: 

For getting the number of correct values, I first extracted the number from prediction using argmax and equate it with truth data, and divided it with the total input 

```
def get_num_correct(images, labels, random, random_label):
    
    return images.argmax(dim=1).eq(labels).sum().item(), random.argmax(dim=1).eq(random_label).sum().item()

```
**After training my model for 80 epoch, I got 99.8% accuracy for mnist 47.0% accuracy for random number. So my final model accuracy is 73.4%**

### Loss Function: 

I used Cross Entropy because it is used for classification tasks which measure of the difference between the predicted probability distribution and the true probability distribution. 


### Training Logs

```
epoch: 1 MNIST {Correct: 53744 Accuracy: 0.896 Loss: 470.44 } RANDOM {Correct: 5750 Accuracy: 0.096 Loss: 874.68 } Total {Correct: 59494 Accuracy: 0.496 loss: 1345.12 }
epoch: 2 MNIST {Correct: 58533 Accuracy: 0.976 Loss: 446.07 } RANDOM {Correct: 6471 Accuracy: 0.108 Loss: 871.16 } Total {Correct: 65004 Accuracy: 0.542 loss: 1317.23 }
epoch: 3 MNIST {Correct: 59043 Accuracy: 0.984 Loss: 443.44 } RANDOM {Correct: 6768 Accuracy: 0.113 Loss: 870.09 } Total {Correct: 65811 Accuracy: 0.548 loss: 1313.53 }
epoch: 4 MNIST {Correct: 59258 Accuracy: 0.988 Loss: 442.27 } RANDOM {Correct: 7054 Accuracy: 0.118 Loss: 869.20 } Total {Correct: 66312 Accuracy: 0.553 loss: 1311.47 }
epoch: 5 MNIST {Correct: 59410 Accuracy: 0.990 Loss: 441.55 } RANDOM {Correct: 7381 Accuracy: 0.123 Loss: 868.31 } Total {Correct: 66791 Accuracy: 0.557 loss: 1309.86 }
epoch: 6 MNIST {Correct: 59459 Accuracy: 0.991 Loss: 441.26 } RANDOM {Correct: 7746 Accuracy: 0.129 Loss: 867.17 } Total {Correct: 67205 Accuracy: 0.560 loss: 1308.42 }
epoch: 7 MNIST {Correct: 59551 Accuracy: 0.993 Loss: 440.84 } RANDOM {Correct: 8089 Accuracy: 0.135 Loss: 866.01 } Total {Correct: 67640 Accuracy: 0.564 loss: 1306.86 }
epoch: 8 MNIST {Correct: 59616 Accuracy: 0.994 Loss: 440.51 } RANDOM {Correct: 8539 Accuracy: 0.142 Loss: 864.46 } Total {Correct: 68155 Accuracy: 0.568 loss: 1304.96 }
epoch: 9 MNIST {Correct: 59631 Accuracy: 0.994 Loss: 440.48 } RANDOM {Correct: 9026 Accuracy: 0.150 Loss: 862.58 } Total {Correct: 68657 Accuracy: 0.572 loss: 1303.07 }
epoch: 10 MNIST {Correct: 59668 Accuracy: 0.994 Loss: 440.22 } RANDOM {Correct: 9547 Accuracy: 0.159 Loss: 860.26 } Total {Correct: 69215 Accuracy: 0.577 loss: 1300.48 }
epoch: 11 MNIST {Correct: 59653 Accuracy: 0.994 Loss: 440.26 } RANDOM {Correct: 10166 Accuracy: 0.169 Loss: 857.65 } Total {Correct: 69819 Accuracy: 0.582 loss: 1297.92 }
epoch: 12 MNIST {Correct: 59700 Accuracy: 0.995 Loss: 440.00 } RANDOM {Correct: 10909 Accuracy: 0.182 Loss: 854.65 } Total {Correct: 70609 Accuracy: 0.588 loss: 1294.66 }
epoch: 13 MNIST {Correct: 59728 Accuracy: 0.995 Loss: 439.92 } RANDOM {Correct: 11666 Accuracy: 0.194 Loss: 850.99 } Total {Correct: 71394 Accuracy: 0.595 loss: 1290.91 }
epoch: 14 MNIST {Correct: 59713 Accuracy: 0.995 Loss: 439.89 } RANDOM {Correct: 12619 Accuracy: 0.210 Loss: 846.88 } Total {Correct: 72332 Accuracy: 0.603 loss: 1286.77 }
epoch: 15 MNIST {Correct: 59728 Accuracy: 0.995 Loss: 439.85 } RANDOM {Correct: 13463 Accuracy: 0.224 Loss: 843.05 } Total {Correct: 73191 Accuracy: 0.610 loss: 1282.89 }
epoch: 16 MNIST {Correct: 59744 Accuracy: 0.996 Loss: 439.72 } RANDOM {Correct: 14361 Accuracy: 0.239 Loss: 838.80 } Total {Correct: 74105 Accuracy: 0.618 loss: 1278.52 }
epoch: 17 MNIST {Correct: 59771 Accuracy: 0.996 Loss: 439.65 } RANDOM {Correct: 15062 Accuracy: 0.251 Loss: 835.24 } Total {Correct: 74833 Accuracy: 0.624 loss: 1274.89 }
epoch: 18 MNIST {Correct: 59790 Accuracy: 0.997 Loss: 439.56 } RANDOM {Correct: 15845 Accuracy: 0.264 Loss: 831.65 } Total {Correct: 75635 Accuracy: 0.630 loss: 1271.20 }
epoch: 19 MNIST {Correct: 59786 Accuracy: 0.996 Loss: 439.54 } RANDOM {Correct: 16521 Accuracy: 0.275 Loss: 828.26 } Total {Correct: 76307 Accuracy: 0.636 loss: 1267.80 }
epoch: 20 MNIST {Correct: 59799 Accuracy: 0.997 Loss: 439.44 } RANDOM {Correct: 17271 Accuracy: 0.288 Loss: 824.59 } Total {Correct: 77070 Accuracy: 0.642 loss: 1264.04 }
epoch: 21 MNIST {Correct: 59808 Accuracy: 0.997 Loss: 439.42 } RANDOM {Correct: 17926 Accuracy: 0.299 Loss: 821.32 } Total {Correct: 77734 Accuracy: 0.648 loss: 1260.74 }
epoch: 22 MNIST {Correct: 59826 Accuracy: 0.997 Loss: 439.33 } RANDOM {Correct: 18566 Accuracy: 0.309 Loss: 818.51 } Total {Correct: 78392 Accuracy: 0.653 loss: 1257.84 }
epoch: 23 MNIST {Correct: 59820 Accuracy: 0.997 Loss: 439.35 } RANDOM {Correct: 19076 Accuracy: 0.318 Loss: 815.99 } Total {Correct: 78896 Accuracy: 0.657 loss: 1255.34 }
epoch: 24 MNIST {Correct: 59838 Accuracy: 0.997 Loss: 439.27 } RANDOM {Correct: 19584 Accuracy: 0.326 Loss: 813.34 } Total {Correct: 79422 Accuracy: 0.662 loss: 1252.62 }
epoch: 25 MNIST {Correct: 59825 Accuracy: 0.997 Loss: 439.33 } RANDOM {Correct: 20003 Accuracy: 0.333 Loss: 811.19 } Total {Correct: 79828 Accuracy: 0.665 loss: 1250.52 }
epoch: 26 MNIST {Correct: 59830 Accuracy: 0.997 Loss: 439.31 } RANDOM {Correct: 20410 Accuracy: 0.340 Loss: 809.09 } Total {Correct: 80240 Accuracy: 0.669 loss: 1248.40 }
epoch: 27 MNIST {Correct: 59842 Accuracy: 0.997 Loss: 439.24 } RANDOM {Correct: 20656 Accuracy: 0.344 Loss: 807.57 } Total {Correct: 80498 Accuracy: 0.671 loss: 1246.82 }
epoch: 28 MNIST {Correct: 59840 Accuracy: 0.997 Loss: 439.25 } RANDOM {Correct: 21155 Accuracy: 0.353 Loss: 805.27 } Total {Correct: 80995 Accuracy: 0.675 loss: 1244.51 }
epoch: 29 MNIST {Correct: 59840 Accuracy: 0.997 Loss: 439.24 } RANDOM {Correct: 21397 Accuracy: 0.357 Loss: 804.10 } Total {Correct: 81237 Accuracy: 0.677 loss: 1243.34 }
epoch: 30 MNIST {Correct: 59852 Accuracy: 0.998 Loss: 439.18 } RANDOM {Correct: 21696 Accuracy: 0.362 Loss: 802.62 } Total {Correct: 81548 Accuracy: 0.680 loss: 1241.80 }
epoch: 31 MNIST {Correct: 59845 Accuracy: 0.997 Loss: 439.21 } RANDOM {Correct: 21931 Accuracy: 0.366 Loss: 801.39 } Total {Correct: 81776 Accuracy: 0.681 loss: 1240.60 }
epoch: 32 MNIST {Correct: 59847 Accuracy: 0.997 Loss: 439.20 } RANDOM {Correct: 22043 Accuracy: 0.367 Loss: 800.64 } Total {Correct: 81890 Accuracy: 0.682 loss: 1239.85 }
epoch: 33 MNIST {Correct: 59852 Accuracy: 0.998 Loss: 439.15 } RANDOM {Correct: 22423 Accuracy: 0.374 Loss: 798.69 } Total {Correct: 82275 Accuracy: 0.686 loss: 1237.84 }
epoch: 34 MNIST {Correct: 59858 Accuracy: 0.998 Loss: 439.15 } RANDOM {Correct: 22686 Accuracy: 0.378 Loss: 797.43 } Total {Correct: 82544 Accuracy: 0.688 loss: 1236.58 }
epoch: 35 MNIST {Correct: 59866 Accuracy: 0.998 Loss: 439.07 } RANDOM {Correct: 22866 Accuracy: 0.381 Loss: 796.38 } Total {Correct: 82732 Accuracy: 0.689 loss: 1235.45 }
epoch: 36 MNIST {Correct: 59868 Accuracy: 0.998 Loss: 439.09 } RANDOM {Correct: 23079 Accuracy: 0.385 Loss: 795.51 } Total {Correct: 82947 Accuracy: 0.691 loss: 1234.60 }
epoch: 37 MNIST {Correct: 59872 Accuracy: 0.998 Loss: 439.09 } RANDOM {Correct: 23267 Accuracy: 0.388 Loss: 794.39 } Total {Correct: 83139 Accuracy: 0.693 loss: 1233.48 }
epoch: 38 MNIST {Correct: 59861 Accuracy: 0.998 Loss: 439.13 } RANDOM {Correct: 23458 Accuracy: 0.391 Loss: 793.46 } Total {Correct: 83319 Accuracy: 0.694 loss: 1232.59 }
epoch: 39 MNIST {Correct: 59873 Accuracy: 0.998 Loss: 439.08 } RANDOM {Correct: 23579 Accuracy: 0.393 Loss: 792.81 } Total {Correct: 83452 Accuracy: 0.695 loss: 1231.89 }
epoch: 40 MNIST {Correct: 59877 Accuracy: 0.998 Loss: 439.05 } RANDOM {Correct: 23748 Accuracy: 0.396 Loss: 791.99 } Total {Correct: 83625 Accuracy: 0.697 loss: 1231.05 }

```
