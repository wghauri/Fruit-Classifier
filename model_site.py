from flask import Flask, render_template, request, Response
import torch
import cv2
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn.modules.pooling import MaxPool2d
from torch.nn.modules.activation import ReLU
from pathlib import Path

app = Flask(__name__)

cap = cv2.VideoCapture(0)

def fruitmodel():

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

    MODEL_PATH = data_path
    MODEL_NAME = "Fruit_Vision.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    loaded_model_0 = FruitModel(input_shape=3,
                                hidden_units=10,
                                output_shape=len(class_names))

    loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH)) 

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
        classification = class_names[pred]

        cv2.putText(frame, "Fruit: " + str(classification) 
                    + " Confidence: " + str(f"{conf * 1000:.0f}%"), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        
        if not ret:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break

    cap.release()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    for i in range (1,5):
        cv2.waitKey(1)
            
@app.route('/')
def getworld():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(fruitmodel(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(port=3000,debug=True)
