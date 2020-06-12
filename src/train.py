import torch 
import argparse
from models import MODEL_DISPATCHER

def main():
    parser = argparse.ArgumentParser()

    #TODO: Add paramaeters as arguments
    # Required paramaters
    parser.add_argument(
        "--device", 
        default=None, 
        type=str, 
        required=True, 
        help="device on which to run the training"
    )
    parser.add_argument(
        '--training_folds_csv', 
        default=None, 
        type=str, 
        required=True, 
        help="training file with Kfolds"
    )
    parser.add_argument(
        '--model_name', 
        default='se_resnext_50',
        type=str, 
        required=True, 
        help="Name selected in the list: " + f"{','.join(MODEL_DISPATCHER.keys())}"
    )
    args = parser.parse_args()
    
    model = MODEL_DISPATCHER[args.model_name](pretrained='imagenet')

if __name__=='__main__':
    main()



