import torch
import cv2
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn.modules.pooling import MaxPool2d
from torch.nn.modules.activation import ReLU
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from timeit import default_timer as timer

data_path = Path("/Users/waqwaq/Python Projects/Pytorch/Fruit Detection Model/")
train_dir = data_path / "train"
test_dir = data_path / "test"

train_transform = transforms.Compose([transforms.Resize(size=(64,64)),
                                      transforms.TrivialAugmentWide(num_magnitude_bins=31),
                                      transforms.ToTensor()])

test_transform = transforms.Compose([transforms.Resize(size=(64,64)),
                                     transforms.ToTensor()])

# Use ImageFolder to create dataset
train_data = datasets.ImageFolder(root=train_dir,
                                  transform=train_transform, # Transform for data
                                  target_transform=None) # Transform for labels

test_data = datasets.ImageFolder(root=test_dir,
                                 transform=test_transform)

class_names = train_data.classes 

BATCH_SIZE = 1
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              num_workers=1,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             num_workers=1,
                             shuffle=False)

class FruitModel(nn.Module):
  def __init__(self,input_shape: int, hidden_units: int, output_shape: int) -> None:
    super().__init__()
    
    self.conv_block_1 = nn.Sequential(
        nn.Conv2d(in_channels=input_shape,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.conv_block_2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2) 
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_units*16*16,
                  out_features=output_shape)
    )

  def forward(self,x):
      return self.classifier(self.conv_block_2(self.conv_block_1(x)))

model_0 = FruitModel(input_shape=3, # num of color channels in image data
                  hidden_units=10,
                  output_shape=len(class_names))

def accuracy_fn(y_true, y_pred):
  correct = torch.eq(y_true,y_pred).sum().item()
  acc = (correct/len(y_pred)) * 100
  return acc

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(),lr=0.001)

epochs = 6

start_time = timer()

for epoch in range(epochs):
  train_loss, train_acc = 0,0

  for batch, (X,y) in enumerate(train_dataloader):
    model_0.train()
    
    y_pred = model_0(X)

    loss = loss_fn(y_pred,y)
    train_loss += loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_acc += accuracy_fn(y_true=y,y_pred=y_pred.argmax(dim=1))
  
  train_loss /= len(train_dataloader)
  train_acc /= len(train_dataloader)

  test_loss, test_acc = 0,0
  model_0.eval()
  
  with torch.inference_mode():
    for batch, (X,y) in enumerate(test_dataloader):

      test_pred = model_0(X)

      loss = loss_fn(test_pred,y)
      test_loss += loss

      test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)

  print(f"Train_Loss: {train_loss:.4f} | Train_Accuracy: {train_acc:.2f}% | Test_Loss: {test_loss:.4f} | Test_Accuracy: {test_acc:.2f}%")

end_time = timer()
print(f"Total training time: {end_time-start_time: .3f} seconds")

MODEL_PATH = Path("FruitModel")
MODEL_PATH.mkdir(parents=True,
                 exist_ok=True)
MODEL_NAME = "Fruit_Vision.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(),
           f=MODEL_SAVE_PATH)

loaded_model_0 = FruitModel(input_shape=3,
                            hidden_units=10,
                            output_shape=len(class_names))

loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    resized_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(resized_frame, (64, 64))
    resized_frame = resized_frame / 255.0
    tensor_frame = torch.Tensor(resized_frame).permute(2,0,1).unsqueeze(dim=0)
    output = loaded_model_0(tensor_frame)
    predicted_class = torch.softmax(output,dim=1)    
    pred_class = torch.softmax(predicted_class,dim=1)
    pred = torch.argmax(predicted_class).item()
    conf = pred_class[0][pred]

    cv2.putText(frame, "Predicted Class: " + str(class_names[pred]) 
                + " Confidence: " + str(f"{conf * 1000:.0f}%"), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow('frame',frame)
    
    key = cv2.waitKey(30) & 0xff
    if key == 27:
        break

cap.release()
cv2.waitKey(1)
cv2.destroyAllWindows()
for i in range (1,5):
    cv2.waitKey(1)