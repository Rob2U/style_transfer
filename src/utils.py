import cv2
import torch
import PIL.Image as Image
from src.models import JohnsonsImageTransformNet
from src.config import ACCELERATOR
from src.dataset import test_transform

# load model
pretrained_filename = "/dhc/home/robert.weeke/repos/style_transfer/checkpoints/<class 'src.models.johnson_model.JohnsonsImageTransformNet'>--2023-07-22_11-26-02vanGogh--up--in--vgg16--1-10-1e-5--allreflect--long.pth"
print("Loading pretrained model from %s..." % pretrained_filename)
# Automatically loads the model with the saved hyperparameters
model = JohnsonsImageTransformNet()
model.load_state_dict(torch.load(pretrained_filename))

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame)
    frame_tensor = test_transform(frame_pil)
    frame_out = model(frame_tensor.unsqueeze(0).to(ACCELERATOR)).squeeze(0)
    # convert to cv2
    frame_out = frame_out.permute(1, 2, 0).cpu().detach().numpy()
    frame = cv2.fromarray(frame_out)
    
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

cv2.destroyWindow("preview")
vc.release()