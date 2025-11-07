import torch
import numpy as np
from torch.nn import BCEWithLogitsLoss
from utils.metrics import computeMetrics
import wandb
import pandas as pd
from utils.qupath import processjson

def test(model,testloader,args,bestperformance,epoch):
    lr=args.lr
    dr=args.dropout_rate
    seed=args.seed
    task=args.task
    datasetpath=args.datasetpath
    optimal=args.optimalthreshold
    modello=args.modeltype
    model.eval()
    results=[]
    test_predictions0=[]
    test_predictions1=[]
    test_labels=[]
    testloss=[]
    idxs=[]
    names=[]
    loss_module_instance = BCEWithLogitsLoss()
    try:
        if task == "PFI":
            pfis=[]
            dfpfi=pd.read_csv("./clinical_export_2023-07-05.csv")
            expfi=pd.read_excel("./240607_Clinical_export_Decider_Collab.xlsx")
        elif task == "HR":
            hrs=[]
    except:
        print("Task file not found")
        dfpfi=None
        expfi=None
    # elif task == "Stadio":
    #     stadios=[]
    for _,data in enumerate(testloader):
        data= data.cuda()
        x, edge_index,childof,level,x_coords,y_coords,name = data.x, data.edge_index,data.childof,data.level,data.x_coord,data.y_coord,data.name
        if data.__contains__("edge_index_2") and data.__contains__("edge_index_3"):
            edge_index2,edge_index3=data.edge_index_2,data.edge_index_3
        else:
            edge_index2=None
            edge_index3=None
        id= name[0].split("_")[0]
        idxs.append(id)        
        names.append(name[0])
        if task == "PFI" and dfpfi is not None and expfi is not None:
            # dfpfi['platinum_free_interval']=dfpfi['platinum_free_interval'].fillna(dfpfi['platinum_free_interval_at_update'])
            # try:
            #     pfitmp=int(dfpfi[dfpfi["cohort_code"]==id]["platinum_free_interval"].item())
            # except:
            try:
                pfitmp=int(expfi[expfi["Patient card::Patient cohort code_Patient Card"]==id]["PFI_KaplanM_allHGSC"].item())
            except:
                pfitmp=0
                print('PFI non trovato')
            pfis.append(pfitmp)
        elif task == "HR":
            if name[0].split("_")[-1]=="1":
                hr="HRP"
            else:
                hr="HRD"
            hrs.append(hr)
        # elif task == "Stadio":
        #     if name[0].split("_")[-1]=="1":
        #         stadio="IV"
        #     else:
        #         stadio="III"
        #     stadios.append(stadio)

        results = model(x, edge_index,level,childof,edge_index2,edge_index3)
        bag_label=data.y.float().squeeze().cpu().numpy()
        levelmax=level.max()
        x_coords=x_coords[level==levelmax]
        y_coords=y_coords[level==levelmax]
        loss= model.compute_loss(loss_module_instance,results,data.y.float())
        testloss.append(loss.item())
        if model.classes==2:
            if bag_label==1:
                bag_label= torch.LongTensor([[0,1]]).float().squeeze().cpu().numpy()
            else:
                bag_label= torch.LongTensor([[1,0]]).float().squeeze().cpu().numpy()
        test_labels.extend([bag_label])
        preds=model.predict(results)
        test_predictions0.extend([(preds[0]).squeeze().cpu().detach().numpy()])

        if preds[1] is not None:
            test_predictions1.extend([(preds[1]).squeeze().cpu().detach().numpy()])
    testloss=np.array(testloss)
    test_loss = np.mean(testloss)
    test_labels = np.array(test_labels)
    test_predictions0 = np.array(test_predictions0)
    
    test_predictions1 = np.array(test_predictions1)
    test_predictions,probabilities,acc,auc,f1,precision,recall,cm,specificity,image,threshold=computeMetrics(test_labels,test_predictions0,model.classes,names,optimal)

    wandb.log({
            "Test/Loss":test_loss,
            "Test/Acc":acc,
            "Test/f1":f1,
            "Test/precision":precision,
            "Test/recall":recall,
            "Test/confusion_matrix":wandb.Image(image),
            "Test/specificity":specificity,
            "Test/auc":auc,
            "Test/optimal_threshold":threshold,
            "epoch":epoch,
        })
    performance=float(auc)
    if performance>bestperformance:
        bestperformance=performance
        for _,data in enumerate(testloader):
            data= data.cuda()
            x, edge_index,childof,level,x_coords,y_coords,name = data.x, data.edge_index,data.childof,data.level,data.x_coord,data.y_coord,data.name
            results = model(x, edge_index,level,childof,None,None)
            x_coords=x_coords[level==levelmax]
            y_coords=y_coords[level==levelmax]
           
            processjson(A= results["higher"][2].cpu().detach().numpy(),x=x_coords.cpu().detach().numpy(),y=y_coords.cpu().detach().numpy(),name=name[0],levelmax=levelmax,epoch=epoch,modello=modello,learning_rate=lr,dropout_rate=dr,seed=seed,task=task,dataset=datasetpath.split('/')[-1],optimal=optimal) 
        if task == "PFI" and dfpfi is not None and expfi is not None:
            df=pd.DataFrame({"id":idxs,"slide_name":names,"test_labels":test_labels,"probability":probabilities,"predictions":test_predictions,"pfi":pfis})
        elif task == "HR":
            df=pd.DataFrame({"id":idxs,"slide_name":names,"test_labels":test_labels,"probability":probabilities,"predictions":test_predictions,"hr":hrs})
        if df is not None:
            wandb.log({"table":wandb.Table(columns=list(df.columns),data=df)})

    
    return bestperformance


