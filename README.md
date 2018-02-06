# Dog breed classification

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
  --gamma GAMMA         Factor by witch you would like to decrease the
                        learning rate
  --batchsize BATCHSIZE
                        training batch size
  --epoch EPOCH         number of epochs to train for
  --lr LR               Learning Rate. Default=0.08
  --save_name SAVE_NAME
                        name of the files to write for the logs.
