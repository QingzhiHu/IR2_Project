
from matchmaker.losses.lambdarank import *
from matchmaker.losses.listnet import *
from matchmaker.losses.ranknet import *
from matchmaker.losses.msmargin import *


def merge_loss(losses, log_vars):
    loss = torch.zeros(1,device=log_vars.device)
    weighted_losses = []
    for l in range(len(losses)):
        precision = torch.exp(-log_vars[l])
        wl = torch.sum(precision * losses[l] + log_vars[l], -1)
        loss += wl
        weighted_losses.append(wl.detach())
    return torch.mean(loss),weighted_losses

def get_loss(config):

    use_list_loss=False
    use_inbatch_list_loss=False
    inbatch_loss=None
    qa_loss=None

    if config["loss"] == "ranknet":
        loss = RankNetLoss()
    elif config["loss"] == "margin":
        loss = torch.nn.MarginRankingLoss(margin=1, reduction='mean')
    elif config["loss"] == "mrr":
        loss = SmoothMRRLoss()
        use_list_loss = True
    elif config["loss"] == "listnet":
        loss = ListNetLoss()
        use_list_loss = True
    elif config["loss"] == "lambdarank":
        loss = LambdaLoss("ndcgLoss2_scheme")
        use_list_loss = True
    else:
        raise Exception("Loss not known")

    return loss, qa_loss, inbatch_loss, use_list_loss,use_inbatch_list_loss
