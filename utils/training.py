from utils.test import test
from torch.nn import BCEWithLogitsLoss
import torch
import os
import numpy as np
import wandb
from argparse import Namespace
import time
from utils.metrics import computeMetrics

def args_to_dict(args):
    # Convert argparse.Namespace object to a dictionary
    d = {
        "name": args.modeltype,
        "optimizer": {"lr": args.lr,
                      "weight_decay": args.weight_decay
                      },
        "gnn": {"residual": args.residual,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "dropout_rate": args.dropout,
                "layer_name": args.layer_name,
                "heads": args.heads
                },
        "training": {"seed": args.seed,
                     "n_epoch": args.n_epoch},
        "dimensions": {"n_classes": args.n_classes,
                       "c_hidden": args.c_hidden,
                       "input_size": args.input_size},
        "dataset": {"scale": args.scale,
                    "dataset_name": args.dataset,
                    "dataset_path": args.datasetpath,
                    "root": args.root},
        "distillation": {"lamb": args.lamb,
                         "beta": args.beta,
                         "tau": args.temperature},

        "main": {"target": args.target,
                 "kl": args.kl},

    }
    return d


def train(model: torch.nn.Module,
          trainloader: torch.nn.Module,
          valloader: torch.nn.Module,
          testloader: torch.nn.Module,
          args: Namespace) -> torch.nn.Module:
    """train model

    Args:
        model (torch.nn.Module): model
        trainloader (torch.nn.Module): train loader
        valloader (torch.nn.Module): validation loader
        testloader (torch.nn.Module): test loader
        args (Namespace): configurations

    Returns:
        torch.nn.Module: trained model
    """

    #config wandb
    run=wandb.init(project=args.project,entity=args.entity,name=args.wandbname,save_code=True,settings=wandb.Settings(start_method='fork'),tags=[args.tag])
    wandb.config.update(args)
    #-----
    
    epochs=args.n_epoch

    model.train()
    model= model.cuda()
    loss_module_instance = BCEWithLogitsLoss()
    optimal=args.optimalthreshold
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 0.000005)
        
    BestPerformance=0
    #start training
    for epoch in range(epochs):
        start_training=time.time()
        if hasattr(model,"preloop"):
            model.preloop(epoch,trainloader)
        tloss=[]
        names=[]
        testloss=[]
        train_labels=[]
        train_predictions=[]
        for _,data in enumerate(trainloader):

            model.train()
            optimizer.zero_grad()
            data= data.cuda()
            x, edge_index,childof,level, name = data.x, data.edge_index,data.childof,data.level, data.name
            if data.__contains__("edge_index_2") and data.__contains__("edge_index_3"):
                edge_index2,edge_index3=data.edge_index_2,data.edge_index_3
            else:
                edge_index2=None
                edge_index3=None
            if "lung" in args.dataset:
                x=x[level==3]
            try:
                results = model(x, edge_index,level,childof,edge_index2,edge_index3)
            except:
                continue
            bag_label=data.y.float()
            loss= model.compute_loss(loss_module_instance,results,bag_label)
            tloss.append(loss.item())
            loss.backward()
            optimizer.step()
            if model.classes==2:
                if bag_label==1:
                    bag_label= torch.LongTensor([[0,1]]).float().squeeze().cpu().numpy()
                else:
                    bag_label= torch.LongTensor([[1,0]]).float().squeeze().cpu().numpy()
            names.append(name[0])
            train_labels.extend([bag_label.squeeze().cpu().numpy()])
            preds=model.predict(results)
            train_predictions.extend([(preds[0]).squeeze().cpu().detach().numpy()])
        end_training=time.time()
        print("training_time: %s  seconds" % (end_training-start_training))
        scheduler.step()
        tloss=np.array(tloss)
        trainloss=np.mean(tloss)
        _,_,acc,auc,f1,precision,recall,cm,specificity,image,threshold=computeMetrics(np.array(train_labels),np.array(train_predictions),model.classes,names,optimal)
        wandb.log({"Train/loss":trainloss,
                    "Train/Acc":acc,
                    "Train/f1":f1,
                    "Train/precision":precision,
                    "Train/recall":recall,
                    "Train/confusion_matrix":cm,
                    "Train/specificity":specificity,
                    "Train/AUC":auc,
                    "Train/Confusion_matrix":wandb.Image(image),
                    "Train/Threshold":threshold,
                    "epoch":epoch,
                    "lr": scheduler.get_last_lr()[0]
                    })
        if epoch>15:
            with torch.no_grad():
                start_test=time.time()
                BestPerformance = test(model,testloader=testloader,args=args,bestperformance=BestPerformance,epoch=epoch)

                model.eval()

                end_test=time.time() 
                print("test_time: %s  seconds" % (end_test-start_test))

    
    torch.save(model.state_dict(),os.path.join(wandb.run.dir, "final.pt"))
    wandb.save(os.path.join(wandb.run.dir, "final.pt"))
    wandb.finish()
    return model
