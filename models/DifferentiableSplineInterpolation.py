import torch
import torch.nn as nn
from torch.autograd import Variable


class DifferentiableSplineInterpolation(nn.Module):
    def __init__(self, output_size):
        super(DifferentiableSplineInterpolation, self).__init__()
        self.output_size = output_size
        self.pdist = nn.PairwiseDistance(p=2)

    def similarity_matrix(self, mat):
        
        
        
        r = torch.mm(mat, mat.t())
        
        diag = r.diag().unsqueeze(0)
        diag = diag.expand_as(r)
        
        D = diag + diag.t() - 2*r + 1e-8
        return D.sqrt()

    def forward(self, pts, vals):
        pts_num = pts.size(0)
        dists = self.similarity_matrix(pts) 
        A = dists
        B = torch.cat((Variable(pts.data.new(pts_num, 1).zero_() + 1), pts), 1)
        Y = torch.cat((vals, Variable(pts.data.new(3, 2).zero_())), 0)
        tmpX0 = torch.cat((dists, B.transpose(0, 1)), 0)
        tmpX1 = torch.cat((B, Variable(pts.data.new(3, 3).zero_())), 0)
        X = torch.cat((tmpX0, tmpX1), 1)
        
        wv, LU = torch.solve(Y, X)
        
        coord_x = torch.arange(0, self.output_size[1]).unsqueeze(0).expand(self.output_size)
        coord_y = torch.arange(0, self.output_size[0]).unsqueeze(1).expand(self.output_size)
        coord = torch.cat((coord_x.unsqueeze(2), coord_y.unsqueeze(2)), 2)
        coord = coord.view(-1, 2).cuda()
        coord = Variable(coord)
        dists_list = []
        for i in range(pts_num):
            
            dists_list.append(torch.unsqueeze(self.pdist(coord, pts[i, :]),1))
        AA = torch.cat(dists_list, 1)
        
        AA = torch.cat((AA, Variable(pts.data.new(coord.size(0), 1).zero_() + 1.0), coord.float()), 1)
        res = torch.mm(AA, wv).view(self.output_size[0], self.output_size[1], 2)
        return res


if __name__ == '__main__':
    dsi = DifferentiableSplineInterpolation((10, 10))
    pts = Variable(torch.rand((1, 10, 2)) * 10, requires_grad=True)
    vals = Variable(torch.rand(1, 10, 2), requires_grad=True)
    out = dsi(pts, vals)
    loss_fn = nn.MSELoss()
    loss = loss_fn(out, Variable(torch.zeros(out.size())))
    loss.backward()
