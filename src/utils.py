import cv2
import torch
import PIL.Image as Image
import torchvision.transforms as transforms
from src.models import JohnsonsImageTransformNet

ACCELERATOR = torch.device("cpu")

# load model
pretrained_filename = "./checkpoints/<class 'src.models.johnson_model.JohnsonsImageTransformNet'>--2023-07-22_11-26-02vanGogh--up--in--vgg16--1-10-1e-5--allreflect--long.pth"
print("Loading pretrained model from %s..." % pretrained_filename)
# Automatically loads the model with the saved hyperparameters
model = JohnsonsImageTransformNet()
model.load_state_dict(torch.load(pretrained_filename, map_location=torch.device('cpu')))

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
    
    
    frame_pil = transforms.Resize(256, antialias=True)(frame_pil)
    frame_pil = transforms.CenterCrop((256, 256))(frame_pil)
    frame_pil = transforms.ToTensor()(frame_pil)
    
    frame_out = model(frame_pil.unsqueeze(0).to(ACCELERATOR)).squeeze(0)
    # convert to cv2
    frame_out = frame_out.permute(1, 2, 0).cpu().detach().numpy()
    # convert to BGR
    frame_out = cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR)
    frame = frame_out
    
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

cv2.destroyWindow("preview")
vc.release()