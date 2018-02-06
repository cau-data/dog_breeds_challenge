# Dog breed classification with Pytorch

In this repository you will find the codes to run a Dog breed classifier and evaluate it. A complete documentation is available in this [google drive](https://drive.google.com/drive/u/1/folders/1m4KSRP_kCq6rxbzAyqgHBg4kF3-Qxx0p) alongside with models I pretrained and that can be loaded.

## Requirements
-   Install PyTorch ([pytorch.org](http://pytorch.org/))
-   `pip install -r requirements.txt`
-   run `set_train_test.py`
   
This last code will download the [Standford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) and split it between a train and test folder into your root directory. 

## Training

To train a model, run `main.py` with the desired model architecture:

```shell
python main.py --model 'squeezenet' --epoch 5
```
In this example a pretrained model of the squeezenet1.1 will be loaded and running through 5 epochs.

### Usage


```
usage: main.py [-h] [--model MODEL] [--pretrained PRETRAINED] [--step STEP]
               [--gamma GAMMA] [--batchsize BATCHSIZE] [--epoch EPOCH]
               [--lr LR] [--save_name SAVE_NAME]

Dog breed classifier

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Name or path to the model
  --pretrained PRETRAINED
                        If the model is pretrained or not, freeze all the
                        layers except the last
  --step STEP           Number of steps before decreasing the learning rate
  --gamma GAMMA         Factor by whitch the learning rate is decreased
  --batchsize BATCHSIZE
                        training batch size
  --epoch EPOCH         number of epochs to train for
  --lr LR               Learning Rate. Default=0.08
  --save_name SAVE_NAME
                        name of the files to write for the logs.
```

### Example usage
The different models are: resnet101, resnet34 and squeezenet. It is important to know the capacity of your computer, the code will automatically use a gpu if you have one, but some models are very RAM consuming. You can chose a smaller batch size or a lighter network (squeezenet < resnet34 < resnet101).

If you have downloaded the `saved_models` folder in the [google drive](https://drive.google.com/drive/u/1/folders/1m4KSRP_kCq6rxbzAyqgHBg4kF3-Qxx0p) therefore you can run model that I have already pretrained on the dog dataset, if doing so use a low learning rate (~0.00001) in order to continue the training where it was left. Here is an example of the correct command:

```shell
python main.py --model './saved_models/128_squeezenet_pretrained.pth' --batchsize 128 --epoch 5 --lr 0.00001
```

If you want to run a model from scratches and save it:
```shell
python main.py --model 'resnet34' --epoch 30 --step 8 --save 'myresnet'
```
It will also save the accuracy and loss that you can plot with the `evaluation.ipynb` notebook.

## Evaluation
The `evaluation.ipynb` contains all the evaluation functions used to realise the Documentation: Plots of the accuracies and loss of the models, visualisation of the predicted images in the test set, classification of an image of your choice and confusion matrix.