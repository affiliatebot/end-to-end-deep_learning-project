import torch 
from pathlib import Path
from torchvision import transforms
from PIL import Image
from io import BytesIO

#Steps:

#Import necessary libraries: You will need the Image module from Pillow and the BytesIO class from the io module.
#Wrap the bytes in BytesIO: Create a BytesIO object using your image bytes.
#Open with Image.open(): Pass the BytesIO object to Image.open() to get the PIL image object. 




class Prediction:
    def __init__(self):
        self.model = torch.load(Path("artifacts","training", "model.pt"))
        self.model.eval()
        self.transform = transforms.Compose([ transforms.ToTensor(),
                                             transforms.Resize((224,224))])


    def predict(self, img_bytes:bytes):

        # 1. convert bytes -> PIL
        img = Image.open(BytesIO(img_bytes)).convert("RGB")

        # 2. convert PIL -> tensor
        #    PIL → Tensor [C,H,W]
        input_img = self.transform(img)

        # 3. Add batch dim [1,C,H,W]
        xb =  input_img.unsqueeze(0)

        # 4. Inference
        with torch.no_grad():
            out = self.model(xb)
        pred = torch.argmax(out,dim=1).item()
        
        if pred == 0:
            return "CAT"
        elif pred == 1:
            return "DOG"
        else:
            return pred



