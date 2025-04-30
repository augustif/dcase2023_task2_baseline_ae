import mlflow
import os
import random
import numpy as np
import torch

# original lib
import common as com
from networks.models import Models

########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
########################################################################

def main():
    parser = com.get_argparse()
    # read parameters from yaml
    flat_param = com.param_to_args_list(params=param)
    args = parser.parse_args(args=flat_param)
    # read parameters from command line
    args = parser.parse_args(namespace=args)
    print(args)

    if args.train_only and args.test_only:
        raise ValueError("--train_only and --test_only cannot be used together.")
    elif args.train_only:
        train = True
        test = False
    elif args.test_only:
        train = False
        test = True
    else:
        train = True
        test = True
    
    args.cuda = args.use_cuda and torch.cuda.is_available()

    # Python random
    random.seed(args.seed)
    # Numpy
    np.random.seed(args.seed)
    # Pytorch
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

    net = Models(args.model).net(
        args=args,
        train=train,
        test=test
    )


    print(args.model)

    print("============== BEGIN TRAIN ==============")
    if train:
        for epoch in range(1, args.epochs + 2):
            net.train(epoch)
            mlflow.log_metric('train_loss', net.train_loss, step=epoch)
            mlflow.log_metric('valid_loss', net.val_loss, step=epoch)

    print("============ END OF TRAIN ============")
    
    if test:
        net.test()
        mlflow.log_metric('test_AUC_source', net.test_AUC_source)
        mlflow.log_metric('test_AUC_target', net.test_AUC_target)
        mlflow.log_metric('test_precision_source', net.test_precision_source)
        mlflow.log_metric('test_recall_source', net.test_recall_source)
        mlflow.log_metric('test_f1score_source', net.test_f1score_source)
        mlflow.log_metric('test_precision_target', net.test_precision_target)
        mlflow.log_metric('test_recall_target', net.test_recall_target)
        mlflow.log_metric('test_f1score_target', net.test_f1score_target)

    mlflow.pytorch.log_model(net.model, "model")

if __name__ == "__main__":

    data_path = r'data\dcase2024t2\dev_data\raw\CoffeeGrinder\train'
    log_params_source = {'grind':[], 'bkg':[]}
    log_params_target = {'grind':[], 'bkg':[]}
    for data in os.listdir(data_path):
        if 'source' in data:
            grind_size_source = data.split('grind_')[-1].split('_')[0]  
            if grind_size_source not in log_params_source['grind']:
                log_params_source['grind'].append(grind_size_source)
            bkg_source = data.split('bkg_')[-1].split('_')[0].split('.')[0]
            if bkg_source not in log_params_source['bkg']:
                log_params_source['bkg'].append(bkg_source)
        elif 'target' in data:
            grind_size_target = data.split('grind_')[-1].split('_')[0]  
            if grind_size_target not in log_params_target['grind']:
                log_params_target['grind'].append(grind_size_target)
            bkg_target = data.split('bkg_')[-1].split('_')[0].split('.')[0]
            if bkg_target not in log_params_target['bkg']:
                log_params_target['bkg'].append(bkg_target)
    Experiment_name = "Table"
    run_name = "2mic_f3daysRec_bkg[-10dB]"
    mlflow.set_experiment(Experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("src_grind", log_params_source)
        mlflow.log_param("trg_grind", log_params_target)
        main()