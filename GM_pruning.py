import torch_pruning as tp
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn 
import iresnet
import insightface
from scipy.spatial import distance

def prune_model(model):
    model.cpu()
    DG = tp.DependencyGraph().build_dependency( model, torch.randn(1, 3, 112, 112) )
    def prune_conv(conv, Norm_pruned_percent, GM_pruned_percent):
        weight = conv.weight.detach().cpu()
        out_channels = weight.numpy().shape[0]
        
        # Norm Criteria (L2)
        # for L1 Norm use: norm = np.sum( np.abs(weight.numpy()), axis=(1,2,3))
        norm = np.sum( np.square(weight.numpy()), axis=(1,2,3)) # no need to do square root, it is only for sorting
        num_norm_pruned = int(out_channels * Norm_pruned_percent)
        prune_index = np.argsort(norm)[:num_norm_pruned].tolist() # remove filters with small L2-Norm

        # Distance to GM Criteria (Prune layers closest to Geometric Distance)
        num_GM_pruned = int(out_channels * GM_pruned_percent)

        # indices of unprunned layers
        large_norm_index = []
        large_norm_index = np.argsort(norm)[num_norm_pruned:] # based on norm calculated in "Norm Criteria"
        indices = torch.LongTensor(large_norm_index).cuda()

        # isolate layer left layer indices
        weight_vec = weight.view(weight.size()[0], -1)
        weight_after_norm_prune = torch.index_select(weight_vec.cuda(), 0, indices).cpu().numpy()

        # Calculate distance matrix

        # for euclidean distance
        distance_matrix = distance.cdist(weight_after_norm_prune, weight_after_norm_prune, 'euclidean')
        # for cos similarity
        # distance_matrix = 1 - distance.cdist(weight_vec, weight_vecs, 'cosine')
        distance_sum = np.sum(np.abs(distance_matrix), axis=0)

        # for distance similar: get the filter index with largest similarity == small distance
        sorted_distances_index = distance_sum.argsort()[: num_GM_pruned]
        prune_index_GM = [large_norm_index[i] for i in sorted_distances_index]

        total_prune_index = prune_index + list(prune_index_GM)

        print("norm:", prune_index)
        print("GM:", prune_index_GM)
        print("total:", total_prune_index)

        plan = DG.get_pruning_plan(conv, tp.prune_conv, total_prune_index)
        plan.exec()
    
    block_prune_probs = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49]
    blk_id = 0
    for m in model.modules():
        if isinstance( m, iresnet.IBasicBlock ):
            #print("conv")
            prune_conv( m.conv1, block_prune_probs[blk_id], 0.3 ) # (layer, Norm_prune_percent, GM_prune_percent from rest)
            prune_conv( m.conv2, block_prune_probs[blk_id], 0.3 )
            blk_id+=1
    return model
    
model = iresnet.iresnet100(pretrained=True)
torch.save(model, "before.pt")

dummy = torch.randn(1, 3, 112, 112)

params = sum([np.prod(p.size()) for p in model.parameters()])
print("Number of Parameters before: %.1fM"%(params/1e6))
model.eval()
with torch.no_grad():
  before = (model(dummy))

model2 = prune_model(model)
params = sum([np.prod(p.size()) for p in model2.parameters()])
print("Number of Parameters after: %.1fM"%(params/1e6))

cos = nn.CosineSimilarity(dim=1)
print("--------------")
print ("Cosine Similarity: ")
print(cos(before, model2(dummy)))
torch.save(model2, "after.pt")
