# import required modules
import pandas as pd
import numpy as np
import os
import skimage
import itertools
import sys
import matplotlib.pyplot as plt
from skimage.io import imread
import torch
from torchvision import transforms
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data.utils import decollate_batch
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
from monai.data import ITKReader, PILReader
from monai.transforms import (
    Activations,
    AddChannel,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    Resize,
    ScaleIntensity,
    EnsureType,
)
from monai.utils import set_determinism
from monai.visualize import GradCAM
from PIL import Image
import collections
from monai.visualize import CAM
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)


def main():
    base_project_directory = os.path.join(os.path.dirname(os.path.abspath(__file__))) # get the directory of this file
    print(base_project_directory) # print the directory of this file
    dr_df=pd.read_csv(os.path.join(base_project_directory, 'trainLabels.csv')) # read the csv file
    print(dr_df.head()) # print the first 5 rows of the df
    print(dr_df['level'].value_counts()) # print the number of each level
    dr_df['PatientId'] = dr_df['image'].str.split('_').str[-2] # split the image name by _ and get the patient id
    print(dr_df.tail()) # print the last 5 rows of the df

    dr_df['path'] = dr_df['image'].apply(lambda x: os.path.join(base_project_directory, 'data', '{}.jpeg'.format(x))) # get the path of the images in the data folder and store in "path" column
    print(dr_df.tail()) # print the last 5 rows of the df
    print(dr_df['path'][0]) # print the first path of the images

    dr_df['exists'] = dr_df['path'].apply(os.path.exists) # check if the image exists and store in "exists" column
    print(dr_df.tail()) # print the last 5 rows of the df

    print(dr_df['exists'].sum(), 'images found out of'  , len(dr_df), 'total') # print the number of images found out of the total number of images

    dr_df['label']=dr_df['level'].apply(lambda x: 0 if x<2 else 1) # set the label to 0 if the level is less than 2 and 1 if the level is greater than or equal to 2


    print(dr_df[dr_df['exists']==True].head()) # print the first 5 rows of the df with images that exist
    print(dr_df[dr_df['exists']==False].head()) # print the first 5 rows of the df with images that do not exist

    drPres_df = dr_df[dr_df['exists']==True] # get the rows of the df with images that exist
    drPres_df = drPres_df.reset_index(drop=True) # reset the index of df with images that exist

    print(drPres_df.head()) # print the first 5 rows of the df  with images that exist
    print(len(drPres_df)) # print the number of the df with images that exist

    # %%
    # plot the number of images with each label
    plt.figure(figsize=(10,2))
    plt.hist(drPres_df['label'])
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.show()
    print(drPres_df['label'].value_counts()) 

    val_frac = 0.1 # set the validation fraction to 0.1
    test_frac = 0.1 # set the test fraction to 0.1
    length = len(drPres_df['path']) # get the number of images in the df
    indices = np.arange(length) # get the indices of the df
    np.random.shuffle(indices) # shuffle the indices

    test_split = int(test_frac * length) # get the number of images in the test set
    val_split = int(val_frac * length) + test_split # get the number of images in the validation set
    test_indices = indices[:test_split] # get the indices of the test set
    val_indices = indices[test_split:val_split] # get the indices of the validation set
    train_indices = indices[val_split:] # get the indices of the train set

    # Use a small subset of the datasetfro train, test, validate so that we can run the code quickly
    train_indices = train_indices[:int(len(train_indices)*0.001)] # get the first 0.1% of the train indices
    val_indices = val_indices[:int(len(val_indices)*0.01)] # get the first 1% of the validation indices
    test_indices = test_indices[:int(len(test_indices)*0.01)] # get the first 1% of the test indices

    train_x = [drPres_df['path'][i] for i in train_indices] # get the paths of the images in the train set
    train_y = [drPres_df['label'][i] for i in train_indices] # get the labels of the images in the train set
    val_x = [drPres_df['path'][i] for i in val_indices] # get the paths of the images in the validation set
    val_y = [drPres_df['label'][i] for i in val_indices] # get the labels of the images in the validation set
    test_x = [drPres_df['path'][i] for i in test_indices] # get the paths of the images in the test set
    test_y = [drPres_df['label'][i] for i in test_indices] # get the labels of the images in the test set

    print(
        f"Training count: {len(train_x)}, Validation count: "
        f"{len(val_x)}, Test count: {len(test_x)}") # print the number of images in the train, validation, and test set




    im= Image.open(train_x[0]) # open the first image in the train set
    image_width, image_height = im.size # get the width and height of the image
    im.close() # close the image

    class_names = set(train_y) # get the set of labels in the train set


    num_each=collections.Counter(train_y) # get the number of each label in the train set


    # Train set
    print('Train set') 
    print(f"Total image count: {len(train_x)}")
    print(f"Image dimensions: {image_width} x {image_height}")
    print(f"Label names: {class_names}")
    print(f"Label counts: {num_each}")


    # Validation set
    print('Validation set')
    im= Image.open(val_x[0]) # open the first image in the validation set
    image_width, image_height = im.size # get the width and height of the image
    print(f"Total image count: {len(val_x)}") 
    print(f"Image dimensions: {image_width} x {image_height}")
    print(f"Image shape:: {im.size}")
    class_names = set(val_y) # get the set of labels in the validation set
    print(f"Label names: {class_names}")
    num_each= collections.Counter(val_y) # get the number of each label in the validation set
    print(f"Label counts: {num_each}")


     
    # Test set
    print('Test set')
    im= Image.open(test_x[0]) # open the first image in the test set
    image_width, image_height = im.size # get the width and height of the image
    im.close() # close the image

    print(f"Total image count: {len(test_x)}")
    print(f"Image dimensions: {image_width} x {image_height}")
    class_names = set(test_y) # get the set of labels in the test set
    print(f"Label names: {class_names}")
    num_each= collections.Counter(test_y) # get the number of each label in the test set
    print(f"Label counts: {num_each}")

    # Plot 4 training images
    plt.subplots(2, 2, figsize=(8, 8))
    for i in range(4):
        im = Image.open(train_x[i])
        arr = np.array(im)
        plt.subplot(2, 2, i + 1)
        plt.xlabel(train_y[i])
        plt.imshow(arr, vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()


    num_class = len(class_names) # get the number of classes
    # To Do: Add transformations to training data
    """ train_transforms = Compose( 
        [
            #LoadImage(image_only=True, reader=PILReader(channels_last=False)),
            #AddChannel(),
            
            #Resize((256, 256)),
            
            
            ScaleIntensity(),

            RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
            RandFlip(spatial_axis=0, prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
            EnsureType(),
        ]
    )

    val_transforms = Compose(
        [
            #LoadImage(image_only=True, reader=PILReader()),  
            #AddChannel(), 
            #Resize([256, 256,3]), 
            #Resize((256, 256)),
            #transforms.Grayscale(num_output_channels=3),
            ScaleIntensity(), EnsureType()
        ])
 """
    y_pred_trans = Compose([EnsureType(), Activations(softmax=True)]) # Add transformations to y_pred_trans
    y_trans = Compose([EnsureType(), AsDiscrete(to_onehot=num_class)]) # Add transformations to y_trans
    
    # The Diabetic Retinopathy Dataset Class
    class DRDataset(torch.utils.data.Dataset): 
        def __init__(self, image_files, labels):
            self.image_files = image_files
            self.labels = labels
            
        

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, index):
            
            image_file = self.image_files[index] # get the image file
            label = self.labels[index] # get the label
            image_file = Image.open(image_file) # open the image file
            img = image_file.resize((300,300)) # resize the image
            img=np.array(img) # convert the image to a numpy array
            img=img.transpose((2,0,1)) #get the channels first as required by pytorch (opp in tensorflow)
            img=img/255 # normalize the image
            img=torch.from_numpy(img) # convert the image to a torch tensor
            img=img.float() # convert the image to a float tensor
            label=torch.from_numpy(np.array(label)) # convert the label to a torch tensor
            return (img, label) # return the image and label



    train_ds = DRDataset(train_x, train_y) # create the train dataset
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=2, shuffle=True, num_workers=0) # create the train loader
    print(f"length of train data loader: {len(train_loader)}") # print the length of the train loader
    print(type(train_loader)) # print the type of the train loader
 
    val_ds = DRDataset(val_x, val_y) # create the validation dataset
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=2, num_workers=0) # create the validation loader

    test_ds = DRDataset(test_x, test_y) # create the test dataset
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=2, num_workers=0) # create the test loader
   

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # get the device
    model = DenseNet121(spatial_dims=2, in_channels=3, out_channels=2).to(device) # create the model
    loss_function = torch.nn.CrossEntropyLoss() # create the loss function
    optimizer = torch.optim.Adam(model.parameters(), 1e-5) # create the optimizer
    max_epochs = 2 # set the maximum number of epochs
    val_interval = 1 # set the validation interval
    auc_metric = ROCAUCMetric() # create the AUC metric



    best_metric = -1 # set the best metric to -1
    best_metric_epoch = -1 # set the best metric epoch to -1
    epoch_loss_values = [] # create an empty list for the epoch loss values
    metric_values = [] # create an empty list for the metric values

    for epoch in range(max_epochs):
        print("-" * 10) # print a divider
        print(f"epoch {epoch + 1}/{max_epochs}") # print the epoch
        model.train() # set the model to training mode
        epoch_loss = 0 # set the epoch loss to 0
        step = 0 # set the step to 0
        for batch_data in train_loader: # for each batch in the train loader
            step += 1 # increment the step
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device) # get the inputs and labels
            optimizer.zero_grad() # zero the gradients
            outputs = model(inputs) # get the outputs
            loss = loss_function(outputs, labels) # get the loss
            loss.backward() # backpropagate the loss
            optimizer.step() # update the weights
            epoch_loss += loss.item() # add the loss to the epoch loss
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size}, " # print the step and the epoch loss
                f"train_loss: {loss.item():.4f}") # print the epoch loss
            epoch_len = len(train_ds) // train_loader.batch_size # get the epoch length
        epoch_loss /= step # get the average epoch loss
        epoch_loss_values.append(epoch_loss) # add the epoch loss to the epoch loss values
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}") # print the epoch average loss
        print("label",batch_data[1][0]) # print the label
        if (epoch + 1) % val_interval == 0: # if the epoch is a validation epoch
            model.eval() # set the model to evaluation mode
            with torch.no_grad(): # do not compute gradients
                y_pred = torch.tensor([], dtype=torch.float32, device=device) # create an empty tensor for the predictions
                y = torch.tensor([], dtype=torch.long, device=device) # create an empty tensor for the labels
                for val_data in val_loader: # for each batch in the validation loader
                    val_images, val_labels = (
                        val_data[0].to(device),
                        val_data[1].to(device),
                    ) # get the validation images and labels
                    y_pred = torch.cat([y_pred, model(val_images)], dim=0) # get the predictions
                    y_pred = torch.softmax(y_pred, dim=1) # get the softmax predictions
                    y_pred_sign = y_pred # get the predictions
                    y = torch.cat([y, val_labels], dim=0) # get the labels
                print(f"y_pred: {y_pred.shape}") # print the predictions shape
                print(f"y: {y.shape}") # print the labels shape
                print(f"y_pred: {y_pred}") # print the predictions
                print(f"y: {y}") # print the labels


                y_onehot = [y_trans(i) for i in decollate_batch(y)] # De-collate a batch of data (for example, as produced by a DataLoader). Originally stored as (B,C,H,W,[D]) will be returned as (C,H,W,[D]).
                y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)] # De-collate a batch of data (for example, as produced by a DataLoader). Originally stored as (B,C,H,W,[D]) will be returned as (C,H,W,[D]).
                auc_metric(y_pred_act, y_onehot) # update the AUC metric
                result = auc_metric.aggregate() # get the AUC metric result
                auc_metric.reset() # reset the AUC metric
                del y_pred_act, y_onehot # delete the predictions and labels
                metric_values.append(result) # add the metric result to the metric values
                acc_value = torch.eq(y_pred.argmax(dim=1), y) # get the accuracy
                acc_metric = acc_value.sum().item() / len(acc_value) # get the accuracy metric
                # print the validation results and the metric at current epoch
                if result > best_metric:
                    best_metric = result
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(
                        base_project_directory, "best_metric_model_8.pth"))
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current AUC: {result:.4f}"
                    f" current accuracy: {acc_metric:.4f}"
                    f" best AUC: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )

    print(
        f"train completed, best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch}") # print the best metric and the epoch at which it was obtained

    # plot the epoch loss values and Validation AUC values
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Val AUC")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.show()

    model.load_state_dict(torch.load(   
    os.path.join(base_project_directory, "best_metric_model_8.pth"))) # load the best metric model
    model.eval() # set the model to evaluation mode
    y_true = [] # create an empty list for the true labels
    y_pred = [] # create an empty list for the predicted labels
    with torch.no_grad(): # do not compute gradients
        for test_data in test_loader:   # for each batch in the test loader
            test_images, test_labels = (   # get the test images and labels
                test_data[0].to(device), # get the test images
                test_data[1].to(device), # get the test labels
            )
            pred = model(test_images).argmax(dim=1) # get the predictions
            for i in range(len(pred)): # for each prediction
                y_true.append(test_labels[i].item()) # add the true label to the true labels list
                y_pred.append(pred[i].item()) # add the predicted label to the predicted labels list
    class_names = set(train_y) # get the class names
    class_names=[str(i) for i in class_names] # convert the class names to strings
    print(y_true) # print the true labels
    print(y_pred) # print the predicted labels
    print("class names", ['Mild', 'Severe']) # print the class names
    print(classification_report(
        y_true, y_pred, target_names=class_names, digits=4)) # print the classification report

    cm = confusion_matrix(
        y_true,
        y_pred,
        normalize="true",
    ) # get the confusion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Mild", "Severe"],
    ) # create a confusion matrix display
    disp.plot(include_values=True, cmap="Blues", ax=None) # plot the confusion matrix
    plt.show()

    #disp.plot(ax=plt.subplots(1, 1, facecolor="white")[1])
    #print(model)
    #n_examples = 1
    #subplot_shape = [2, n_examples] # create a subplot shape
    #fig, axes = plt.subplots(*subplot_shape, figsize=(25, 15), facecolor="white") # create a figure and axes
    cam = CAM(nn_module=model, target_layers="class_layers.relu", fc_layers="class_layers.out") # create a CAM object

    for batch_data in test_loader: # for each batch in the test loader
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device) # get the inputs and labels
        result = cam(x=inputs) # get the CAM result
        y_pred = model(inputs) # get the predictions
        y_pred=torch.softmax(y_pred,dim=1) # get the softmax predictions
        print('labels',labels[0]) # print the labels
        print('y_pred',y_pred[0]) # print the predictions
        print('y_pred_abs',y_pred[0].argmax(-1)) # print the predictions
        fig, ax = plt.subplots(1,2, figsize=(10,10)) # create a figure and axes
        ima=np.transpose(inputs[0].cpu().detach().numpy(), (1,2,0)) # get the image
        ax[0].imshow(ima) # plot the image
        ax[0].set_title("Original Retinal Image") # set the title
        ax[1].imshow(ima) # plot the image
        ax[1].imshow(skimage.transform.resize(result[0].reshape(300,300), (ima.shape[0],ima.shape[1] )), alpha=0.25, cmap='jet') # plot the heatmap
        y_pred = str(y_pred[0]) # get the predictions
        ax[1].set_title(y_pred) # set the title
        fig.tight_layout() # make the layout tight
        plt.show() # show the figure
        break




if __name__ == "__main__":
    main()




