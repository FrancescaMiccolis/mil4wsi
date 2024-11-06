import copy
import itertools

def launch_PFI(args):

    multiscale_hyperparameters = [
        [3],  # ntop
        [10],  # freq,
        [48],  # seed
        ["Buffermil"],
        ["mean"],
        # [ "/work/H2020DeciderFicarra/decider/feats/geometric/pfi_45_180/x20",
        #     "/work/H2020DeciderFicarra/decider/feats/geometric/pfi_45_180/x20s_2",
        #     "/work/H2020DeciderFicarra/decider/feats/geometric/pfi_45_180/x20s_3",
        #     "/work/H2020DeciderFicarra/decider/feats/geometric/pfi_45_180/x20s_4",
        #  "/work/H2020DeciderFicarra/decider/feats/geometric/pfi_45_180/x20s_7"],
        ["/work/H2020DeciderFicarra/decider/feats/geometric/pfi_45_180/x20_missing"],
        ["PFI"],
        [True,False]
    ]
  
    args_list = []
    for element in itertools.product(*multiscale_hyperparameters):
        args1 = copy.copy(args)
        args1.scale = "3"
        args1.tag = "Prova_bufferTRUE"
        args1.project = "Downstream_analysis_PFI"
        args1.ntop, args1.buffer_freq, args1.seed, args1.modeltype, args1.bufferaggregate, args1.datasetpath, args1.task,args1.optimalthreshold= element
        args1.wandbname = args1.modeltype+'_'+args1.datasetpath.split("/")[-1]+'OptimalThreshold'+str(args1.optimalthreshold)#+'_optimalthreshold'
        args_list.append(args1)

    return args_list



def launch_DASMIL_cam(args):
    args1= copy.copy(args)
    args1.modeltype="DASMIL"
    args1.lamb=1
    args1.beta=1
    args1.temperature=1.5
    args1.scale="23"
    args1.dataset="cam"
    args1.seed=42
    args1.lr=0.0002
    args1.weight_decay=0.005
    args1.c_hidden=256
    args1.layer_name="GAT"
    args1.residual=False
    return [args1]


def launch_DASMIL_lung(args):
    args1= copy.copy(args)
    args1.modeltype="DASMIL"
    args1.lamb=1
    args1.beta=1
    args1.temperature=1.5
    args1.scale="13"
    args1.dataset="lung"
    args1.seed=42
    args1.lr=0.0002
    args1.weight_decay=0.005
    args1.c_hidden=384
    args1.layer_name="GAT"
    args1.residual=True
    return [args1]
