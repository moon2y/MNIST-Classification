# MNIST-Classification
인공신경망과 딥러닝 과제
MNIST 손글씨 숫자를 분류하는 신경망 모델 구현
  1. LeNet5
  2. Lenet5_regularized
  3. Custom MLP
     
## 파일 설명
- `main.py`: 모델 학습 & 테스트
- `dataset.py`: MNIST 데이터셋 전처리
- `model.py`: LeNet-5(일반/rugularized) & Custom MLP 모델 구현

## 모델 설명
### LeNet5 (Number of parameters : 61,706)
![Lenet5](https://github.com/Chayuho/MNIST_classification/assets/94342487/f8d850d2-0329-47b8-b528-cce4a6177682)
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
![NeNet5_train_loss](https://github.com/Chayuho/MNIST_classification/assets/94342487/f30ecf4d-bf72-4bee-9ad8-3059bdf12467s=0.3)
![NeNet5_train_acc](https://github.com/Chayuho/MNIST_classification/assets/94342487/ac3438e4-8830-4db9-824c-af833821c059)
![NeNet5_test_loss](https://github.com/Chayuho/MNIST_classification/assets/94342487/06e16218-7f63-40a8-a0ba-58feecee890e)
![NeNet5_test_acc](https://github.com/Chayuho/MNIST_classification/assets/94342487/564789d0-2d18-4b53-9c40-c65479973cbc)

### Custom MLP
![CustomMLP_train_loss](https://github.com/Chayuho/MNIST_classification/assets/94342487/0faf4433-53fd-4977-9994-71298bc85105)
![CustomMLP_train_acc](https://github.com/Chayuho/MNIST_classification/assets/94342487/bb8fd77e-608e-47f6-9eaa-bbfa707eb9e6)
![CustomMLP_test_loss](https://github.com/Chayuho/MNIST_classification/assets/94342487/5535032e-82aa-4de5-a670-953928801dda)
![CustomMLP_test_acc](https://github.com/Chayuho/MNIST_classification/assets/94342487/0e776def-3fc3-413a-a361-575a1f1de70a)

### Test Loss & Accuracy
                     | Loss      | Accuracy  |
|--------------------|-----------|-----------|
| LeNet5             | 0.125     | 0.977     |
| LeNet5_regularized | 0.115     | 0.981     |
| Custom MLP         | 0.238     | 0.949     |
