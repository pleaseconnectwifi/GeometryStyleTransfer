import torch
import torch.nn as nn

class SplineInterpolation(nn.Module):
    def __init__(self, output_size):
        super(SplineInterpolation, self).__init__()
        self.output_size = output_size
        self.pdist = nn.PairwiseDistance(p=2)

    def similarity_matrix(self, mat):
        
        
        
        r = torch.mm(mat, mat.t())
        
        diag = r.diag().unsqueeze(0)
        diag = diag.expand_as(r)
        
        D = diag + diag.t() - 2*r
        return D.sqrt()

    def forward(self, pts, vals):
        pts_num = pts.size(0)
        dists = self.similarity_matrix(pts) 
        A = dists
        B = torch.cat((pts.new(pts_num, 1).zero_() + 1, pts), 1)
        Y = torch.cat((vals, pts.new(3, 2).zero_()), 0)
        tmpX0 = torch.cat((dists, B.transpose(0, 1)), 0)
        tmpX1 = torch.cat((B, pts.new(3, 3).zero_()), 0)
        X = torch.cat((tmpX0, tmpX1), 1)
        wv, LU = torch.gesv(Y, X)
        
        coord_x = torch.arange(0, self.output_size[1]).unsqueeze(0).expand(self.output_size)
        coord_y = torch.arange(0, self.output_size[0]).unsqueeze(1).expand(self.output_size)
        coord = torch.cat((coord_x.unsqueeze(2), coord_y.unsqueeze(2)), 2)
        coord = coord.view(-1, 2)
        dists_list = []
        for i in range(pts_num):
            dists_list.append(self.pdist(coord, pts[i, :].expand_as(coord)))
        A = torch.cat(dists_list, 1)
        A = torch.cat((A, pts.new(coord.size(0), 1).zero_() + 1, coord), 1)
        res = torch.mm(A, wv).view(self.output_size[0], self.output_size[1], 2)
        return res