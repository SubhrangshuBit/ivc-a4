import torch
import fcn_model
import fcn_dataset
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from fcn_dataset import CamVidDataset, rev_normalize
from fcn_model import FCN8s
manualSeed = 42
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)
# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model
num_classes = 32
model = FCN8s(num_classes).to(device)



# Define logs
writer = SummaryWriter()

# Hyperparameters
resolution = (384, 512)
batch_size = 16
num_epochs = 100
learning_rate = 0.001
min_delta = 0.1
patience = 10

# Define the early stopping function
class EarlyStopping:
    def __init__(self, tolerance=10, min_delta=0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter=0
        self.early_stop=False
        self.best_loss = -9999999

    def __call__(self,  mean_iou):
        """ 
        If the mean_iou is mode than the best loss by min_delta, then reset the counter.
        """
        if  mean_iou - self.best_loss > self.min_delta:
            print(f'Mean IOU increased ({self.best_loss:.6f} --> {mean_iou:.6f}).  Resetting counter to 0')
            self.best_loss = mean_iou
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.tolerance}')
            print(f'Best mIoU: {self.best_loss}, Current mIoU: {mean_iou}')
            if self.counter >= self.tolerance:
                self.early_stop = True
                print("Early stopping....")
            
                
# Define the loss function and optimizer
def loss_fn(outputs, labels):
    """ 
    In the original paper, the authors mention a per-pixel multinomial logistic loss, which is equivalent to the standard cross-entropy loss.
    """ 
    return torch.nn.CrossEntropyLoss()(outputs, labels)

def calculate_metrics(pred, target, num_classes):
    """ 
    Calculate the pixel accuracy, mean IoU, and frequency weighted IoU.
    """
    total_wt = np.sum([(target == i).sum() for i in range(num_classes)])
    pixel_acc = (pred == target).sum() / (total_wt)
    iou = []
    for i in range(num_classes):
        intersection = ((pred == i) & (target == i)).sum()
        union = ((pred == i) | (target == i)).sum()
        # For a target image, one or more classes might not be in that image leading to zero union
        if union>0:
            iou.append(intersection / union)
        else:
            iou.append(0)

    mean_iou = np.mean(iou)
    freq_iou = np.sum([(target == i).sum() * iou[i] for i in range(num_classes)]) / (total_wt)
    return pixel_acc, mean_iou, freq_iou

def eval_model(model, dataloader, device, save_pred=False):
    model.eval()
    loss_list = []
    if save_pred:
        pred_list = []
    with torch.no_grad():
        for images, labels in (dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss_list.append(loss.item())
            _, predicted = torch.max(outputs, 1)
            if save_pred:
                pred_list.append(predicted.cpu().numpy())
        
        loss = sum(loss_list) / len(loss_list)
        pixel_acc, mean_iou, freq_iou = calculate_metrics(predicted.cpu().numpy(), labels.cpu().numpy(), num_classes)
        print("Validation")
        print('Pixel accuracy: {:.4f}, Mean IoU: {:.4f}, Frequency weighted IoU: {:.4f}, Val Loss: {:.4f}'.format(pixel_acc, mean_iou, freq_iou, loss))
        print('='*20)

    if save_pred:
        pred_list = np.concatenate(pred_list, axis=0)
        np.save('test_pred.npy', pred_list)
        print(f"{len(pred_list)} predictions saved at test_pred.npy")
    model.train()
    return loss, pixel_acc, mean_iou, freq_iou

def visualize_model(model, dataloader, device):
    log_dir = "vis/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    cls_dict = dataloader.dataset.class_dict.copy()
    cls_list = [cls_dict[i] for i in range(len(cls_dict))]
    model.eval()
    with torch.no_grad():
        for ind, (images, labels) in enumerate(tqdm(dataloader)):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            images_vis = fcn_dataset.rev_normalize(images)
            # Save the images and labels
            img = images_vis[0].permute(1, 2, 0).cpu().numpy()
            img = img * 255
            img = img.astype('uint8')
            label = labels[0].cpu().numpy()
            pred = predicted[0].cpu().numpy()

            label_img = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
            pred_img = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
            for j in range(len(cls_list)):
                mask = label == j
                label_img[mask] = cls_list[j][0]
                mask = pred == j
                pred_img[mask] = cls_list[j][0]
            # horizontally concatenate the image, label, and prediction, and save the visualization
            vis_img = np.concatenate([img, label_img, pred_img], axis=1)
            vis_img = Image.fromarray(vis_img)
            vis_img.save(os.path.join(log_dir, 'img_{:04d}.png'.format(ind)))
            
    model.train()
    


# main function
if __name__ == "__main__":

    # Define the dataset and dataloader

    class_dict_path = "class_dict.csv"
    class_dict = pd.read_csv("CamVid/" + class_dict_path)



    images_dir = "train/"
    labels_dir = "train_labels/"
    camvid_dataset = CamVidDataset(root='CamVid/', images_dir=images_dir, labels_dir=labels_dir, class_dict_path=class_dict_path, resolution=resolution)
    train_loader = torch.utils.data.DataLoader(camvid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    val_images_dir = "val/"
    val_labels_dir = "val_labels/"
    val_camvid_dataset = CamVidDataset(root='CamVid/', images_dir=val_images_dir, labels_dir=val_labels_dir, class_dict_path=class_dict_path, resolution=resolution)
    val_loader = torch.utils.data.DataLoader(val_camvid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    test_images_dir = "test/"
    test_labels_dir = "test_labels/"
    test_camvid_dataset = CamVidDataset(root='CamVid/', images_dir=test_images_dir, labels_dir=test_labels_dir, class_dict_path=class_dict_path, resolution=resolution)
    test_loader = torch.utils.data.DataLoader(test_camvid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    print("Number of classes:", class_dict.shape[0])
    print(f"Train dataset size: {len(camvid_dataset)}")
    print(f"Validation dataset size: {len(val_camvid_dataset)}")
    print(f"Test dataset size: {len(test_camvid_dataset)}")
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FCN8s(num_classes=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Train the model
    loss_list = []
    epoch_train_loss = []
    early_stopping = EarlyStopping(tolerance=patience, min_delta=min_delta)
    
    for epoch in range(num_epochs):
        for i, (images, labels) in (enumerate(train_loader)):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            epoch_train_loss.append(loss.item())

            if (i+1) % 10 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), sum(loss_list)/len(loss_list)))
                loss_list = []

        train_loss = sum(epoch_train_loss)/len(epoch_train_loss)
        pixel_acc, mean_iou, freq_iou = calculate_metrics(torch.argmax(outputs, dim=1).cpu().numpy(), labels.cpu().numpy(), num_classes)
        print("Train")
        print('Pixel accuracy: {:.4f}, Mean IoU: {:.4f}, Frequency weighted IoU: {:.4f} Train Loss: {:.4f}'.format(pixel_acc, mean_iou, freq_iou, train_loss))
        print('='*20)

        # eval the model        
        eval_loss, eval_pixel_acc, eval_mean_iou, eval_freq_iou = eval_model(model, val_loader, device)

        # Log performance
        writer.add_scalars("Loss", {"train": train_loss, "val": eval_loss}, epoch)
        writer.add_scalars("PixAcc", {"train": pixel_acc, "val": eval_pixel_acc}, epoch)
        writer.add_scalars("meanIOU", {"train": mean_iou, "val": eval_mean_iou}, epoch)

        # save the best model

        if eval_loss < early_stopping.best_loss:
            print(f'Validation loss decreased ({early_stopping.best_loss:.6f} --> {eval_loss:.6f}).  Saving model ...')
            early_stopping.best_loss = eval_loss
            model_save_path = "best_model.pth"
            # save the model and the optimizer
            torch.save(model.state_dict(), model_save_path)

        # early stoppping
        early_stopping( eval_loss)
        if early_stopping.early_stop:
            print("We are at epoch:", epoch)
            break
    writer.close()
    print('='*20)
    print('Finished Training, evaluating the best model on the test set')

    # load the best model
    best_model = FCN8s(num_classes=32).to(device)
    best_model.load_state_dict(torch.load("best_model.pth"))


    eval_model(best_model, test_loader, device, save_pred=True)
    print('='*20)
    print('Visualizing the model on the test set, the results will be saved in the vis/ directory')
    visualize_model(model, test_loader, device)
