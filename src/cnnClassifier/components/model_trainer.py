import torch
import torch.nn as nn
from pathlib import Path
from torchvision.datasets import ImageFolder # class 
from torch.utils.data import DataLoader
from torch.utils.data import random_split # function
from torchvision import transforms # module
from cnnClassifier.entity.config_entity import TrainingConfig


class Training:

  def __init__(self,config:TrainingConfig):
    
    self.config = config
    self.model = torch.load(self.config.updated_base_model_path)
    self.train_loader = None
    self.val_loader = None
    self.loss_func = None
    self.opt = None
    self.train_data = None
    self.val_data = None
    self.train_loader = None
    self.val_loader = None


  def load_data(self):

    data_path = Path(self.config.training_data)
    img_size = self.config.params_image_size

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor()
                                  ])

    dataset = ImageFolder(data_path,transform=transform)
    self.split_dataset(dataset)
    

  
  def split_dataset(self,dataset):

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    self.train_data, self.val_data = random_split(dataset, [train_size, val_size])

    
  def setup_training(self):

    
    batch_size = self.config.params_batch_size
    lr = self.config.params_learning_rate

    self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
    self.val_loader = DataLoader(self.val_data, batch_size=batch_size)
    self.loss_func = nn.CrossEntropyLoss()
    self.optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=lr)

  @staticmethod
  def accuracy(out,label):

    preds = torch.argmax(out,dim=1)
    batch_acc= torch.sum(preds == label) / len(label)
    return batch_acc


  def train_model(self, loader=None, opt=None):

    losses = []
    accs = []
    total_samples = 0

    # -------------------------
    # TRAINING mode
    # -------------------------
    if opt is not None:
        self.model.train()

        for xb, yb in loader:
            out = self.model(xb)
            batch_loss = self.loss_func(out, yb)
            batch_acc = self.accuracy(out, yb)

            # update classifier parameters
            opt.zero_grad()
            batch_loss.backward()
            opt.step()

            batch_size = len(xb)
            losses.append(batch_loss.item() * batch_size)
            accs.append(batch_acc.item() * batch_size)
            total_samples += batch_size


    # -------------------------
    # VALIDATION mode
    # -------------------------
    else:
        self.model.eval()

        with torch.no_grad():       # <---- This is all you need
            for xb, yb in loader:
                out = self.model(xb)
                batch_loss = self.loss_func(out, yb)
                batch_acc = self.accuracy(out, yb)

                batch_size = len(xb)
                losses.append(batch_loss.item() * batch_size)
                accs.append(batch_acc.item() * batch_size)
                total_samples += batch_size


    avg_loss = sum(losses) / total_samples
    avg_acc = sum(accs) / total_samples

    return avg_loss, avg_acc


  def fit(self):

    epochs = self.config.params_epochs
    train_losses, train_accs, val_losses, val_accs = [],[],[],[]

    for epoch in range(epochs):
      loss, acc = self.train_model(loader=self.train_loader,opt=self.optimizer)
      train_losses.append(loss)
      train_accs.append(acc)

      val_loss, val_acc = self.train_model(loader=self.val_loader,opt=None)
      val_losses.append(val_loss)
      val_accs.append(val_acc)


      print(f"Epoch {epoch+1} Loss: {loss:.4f} Accuracy : {acc} val_Loss: {val_loss:.4f} val_Accuracy : {val_acc}")

    #return train_losses, train_accs, val_losses, val_accs

  
  def save_model(self):
    """Save full PyTorch model"""
    path = self.config.trained_model_path
    torch.save(self.model, path)

  def run(self):
    self.load_data()
    self.setup_training()
    self.fit()
    self.save_model()












    
