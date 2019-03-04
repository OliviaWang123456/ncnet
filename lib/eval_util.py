import torch
import torch.nn
import numpy as np
import os
from skimage import draw
import torch.nn.functional as F
from torch.autograd import Variable
from lib.pf_dataset import PFPascalDataset
from lib.point_tnf import PointsToUnitCoords, PointsToPixelCoords, bilinearInterpPointTnf
from geotnf.point_tnf import PointTnf
from util.py_util import create_file_path
from geotnf.flow import th_sampling_grid_to_np_flow, write_flo_file


def pck(source_points,warped_points,L_pck,alpha=0.15):
    # compute precentage of correct keypoints
    batch_size=source_points.size(0)
    pck=torch.zeros((batch_size))
    for i in range(batch_size):
        p_src = source_points[i,:]
        p_wrp = warped_points[i,:]
        N_pts = torch.sum(torch.ne(p_src[0,:],-1)*torch.ne(p_src[1,:],-1))
        point_distance = torch.pow(torch.sum(torch.pow(p_src[:,:N_pts]-p_wrp[:,:N_pts],2),0),0.5)
        L_pck_mat = L_pck[i].expand_as(point_distance)
        correct_points = torch.le(point_distance,L_pck_mat*alpha)
        pck[i]=torch.mean(correct_points.float())
    return pck


def pck_metric(batch,batch_start_idx,matches,stats,args,use_cuda=True):
       
    source_im_size = batch['source_im_size']
    target_im_size = batch['target_im_size']
    import ipdb; ipdb.set_trace()

    source_points = batch['source_points']
    target_points = batch['target_points']
    
    # warp points with estimated transformations
    target_points_norm = PointsToUnitCoords(target_points,target_im_size)

    # compute points stage 1 only
    warped_points_norm = bilinearInterpPointTnf(matches,target_points_norm)
    warped_points = PointsToPixelCoords(warped_points_norm,source_im_size)
    
    L_pck = batch['L_pck'].data
    
    current_batch_size=batch['source_im_size'].size(0)
    indices = range(batch_start_idx,batch_start_idx+current_batch_size)

    # compute PCK
    pck_batch = pck(source_points.data, warped_points.data, L_pck)
    stats['point_tnf']['pck'][indices] = pck_batch.unsqueeze(1).cpu().numpy()
        
    return stats


def flow_metrics(batch,batch_start_idx,matches,stats,args,use_cuda=True):
    result_path = args.flow_output_dir

    pt = PointTnf(use_cuda=use_cuda)

    batch_size = batch['source_im_size'].size(0)
    for b in range(batch_size):
        h_src = int(batch['source_im_size'][b, 0].data.cpu().numpy())
        w_src = int(batch['source_im_size'][b, 1].data.cpu().numpy())
        h_tgt = int(batch['target_im_size'][b, 0].data.cpu().numpy())
        w_tgt = int(batch['target_im_size'][b, 1].data.cpu().numpy())

        grid_X, grid_Y = np.meshgrid(np.linspace(-1, 1, w_tgt), np.linspace(-1, 1, h_tgt))
        grid_X = torch.FloatTensor(grid_X).unsqueeze(0).unsqueeze(3)
        grid_Y = torch.FloatTensor(grid_Y).unsqueeze(0).unsqueeze(3)
        grid_X = Variable(grid_X, requires_grad=False)
        grid_Y = Variable(grid_Y, requires_grad=False)
        if use_cuda:
            grid_X = grid_X.cuda()
            grid_Y = grid_Y.cuda()

        grid_X_vec = grid_X.view(1, 1, -1)
        grid_Y_vec = grid_Y.view(1, 1, -1)

        grid_XY_vec = torch.cat((grid_X_vec, grid_Y_vec), 1)
        def pointsToGrid(x, h_tgt=h_tgt, w_tgt=w_tgt):
            return x.contiguous().view(1, 2, h_tgt, w_tgt).transpose(1, 2).transpose(2, 3)

        idx = batch_start_idx + b
        source_im_size = batch['source_im_size']
        warped_points_norm = bilinearInterpPointTnf(matches,grid_XY_vec)
        
        # warped_points = PointsToPixelCoords(warped_points_norm,source_im_size)
        warped_points = pointsToGrid(warped_points_norm)

        # grid_aff = pointsToGrid(pt.affPointTnf(theta_aff[b, :].unsqueeze(0), grid_XY_vec))
        flow_aff = th_sampling_grid_to_np_flow(source_grid=warped_points, h_src=h_src, w_src=w_src)
        flow_aff_path = os.path.join(result_path, 'nc', batch['flow_path'][b])
        create_file_path(flow_aff_path)
        write_flo_file(flow_aff, flow_aff_path)

        idx = batch_start_idx + b
    return stats