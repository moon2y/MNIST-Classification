# MNIST-Classification
인공신경망과 딥러닝 과제
  1. LeNet5
  2. Lenet5_regularized
  3. Custom MLP
     
## 파일 설명
- `main.py`: 모델 학습 & 테스트
- `dataset.py`: MNIST 데이터셋 전처리
- `model.py`: LeNet-5(일반/rugularized) & Custom MLP 모델 구현

## 모델 설명
### LeNet5 (Number of parameters : 61,706)
![Lenet5](https://github.com/moon2y/MNIST-Classification/blob/main/plot/321547232-f8d850d2-0329-47b8-b528-cce4a6177682.png)
- ReLU

### LeNet5_regularized (Number of parameters :61990)
- dropout : 0.5
- Batch Normalization applied
- ReLU

### CustomMLP (Number of parameters : 62930)
- Number of Layers : 3
  - Layer 1 : img_size (1, 32, 32) -> (1, 32*32) * 60
  - Layer 2 : 60 * 20
  - Layer 3 : 20 * 10
- ReLU

## 실험결과
### LeNet5
![LeNet5_train_loss](https://github.com/moon2y/MNIST-Classification/blob/main/plot/plotLeNet5_train_loss.png)
![LeNet5_train_acc](https://github.com/moon2y/MNIST-Classification/blob/main/plot/plotLeNet5_train_acc.png)
![LeNet5_test_loss](https://github.com/moon2y/MNIST-Classification/blob/main/plot/plotLeNet5_test_loss.png)
![LeNet5_test_acc](https://github.com/moon2y/MNIST-Classification/blob/main/plot/plotLeNet5_test_acc.png)

### LeNet5_regularized
![LeNet5_regularized_train_loss](https://github.com/moon2y/MNIST-Classification/blob/main/plot/plotLeNet5R_train_loss.png)
![LeNet5_regularized_train_acc](https://github.com/moon2y/MNIST-Classification/blob/main/plot/plotLeNet5R_train_acc.png)
![LeNet5_regularized_test_loss](https://github.com/moon2y/MNIST-Classification/blob/main/plot/plotLeNet5R_test_loss.png)
![LeNet5_regularized_test_acc](https://github.com/moon2y/MNIST-Classification/blob/main/plot/plotLeNet5R_test_acc.png)

### Custom MLP
![CustomMLP_train_loss](https://github.com/moon2y/MNIST-Classification/blob/main/plot/plotMLP_train_loss.png)
![CustomMLP_train_acc](https://github.com/moon2y/MNIST-Classification/blob/main/plot/plotMLP_train_acc.png)
![CustomMLP_test_loss](https://github.com/moon2y/MNIST-Classification/blob/main/plot/plotMLP_test_loss.png)
![CustomMLP_test_acc](https://github.com/moon2y/MNIST-Classification/blob/main/plot/plotMLP_test_acc.png)

### Test Loss & Accuracy
|                    | Loss      | Accuracy  |
|--------------------|-----------|-----------|
| LeNet5             | 0.125     | 0.977     |
| LeNet5_regularized | 0.115     | 0.981     |
| Custom MLP         | 0.238     | 0.949     |
