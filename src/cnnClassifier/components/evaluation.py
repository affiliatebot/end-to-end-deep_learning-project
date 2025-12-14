import torch
import torch.nn as nn
from pathlib import Path
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json



class Evaluation:

  def __init__(self,config:EvaluationConfig):

    self.config = config
    self.model = torch.load(self.config.path_of_model)
    self.val_loader = None
    self.loss_func = None
    self.val_data = None
    self.avg_loss = None
    self.avg_acc = None



  def load_data(self):

    data_path = Path(self.config.test_data)
    img_size = self.config.params_image_size

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor()
                                    #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                  ])
    
    
                                                        

    self.val_data = ImageFolder(data_path,transform=transform)


  def setup(self):

    batch_size = self.config.params_batch_size
    self.val_loader = DataLoader(self.val_data, batch_size=batch_size, shuffle=False)
    self.loss_func = nn.CrossEntropyLoss()


  @staticmethod
  def accuracy(out,label):

    preds = torch.argmax(out,dim=1)
    batch_acc= torch.sum(preds == label) / len(label)
    return batch_acc


  def inference(self):

    losses = []
    accs = []
    total_samples = 0

    self.model.eval()

    with torch.no_grad():
        for xb, yb in self.val_loader:
            out = self.model(xb)
            batch_loss = self.loss_func(out, yb)
            batch_acc = self.accuracy(out, yb)

            batch_size = len(xb)
            losses.append(batch_loss.item() * batch_size)
            accs.append(batch_acc.item() * batch_size)
            total_samples += batch_size


    self.avg_loss = sum(losses) / total_samples
    self.avg_acc = sum(accs) / total_samples





  def save_score(self):
        scores = {"loss": self.avg_loss, "accuracy": self.avg_acc}
        save_json(path=Path("scores.json"), data=scores)

  def evaluation(self):

    self.load_data()
    self.setup()
    self.inference()
    self.save_score()
    

  def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
            {"val_loss": self.avg_loss, "val_accuracy": self.avg_acc}
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.pytorch.log_model(self.model, "model", registered_model_name="VGG16Model")
            else:
                mlflow.pytorch.log_model(self.model, "model")
        mlflow.end_run()
