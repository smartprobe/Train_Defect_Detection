import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class _Structure_inference(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self, n_inputs,n_hidden_e,n_hidden_o):
        super(_Structure_inference, self).__init__()

        # self.boxes = boxes
        self.n_inputs = n_inputs
        self.weight_u = torch.FloatTensor(12, 1).cuda()
        self.weight_u = Variable(self.weight_u, requires_grad=True)
        self.weight_ua = torch.FloatTensor(9, 1).cuda()
        self.weight_ua = Variable(self.weight_ua, requires_grad=True)
        self.Concat_w = torch.FloatTensor( self.n_inputs * 2,1).cuda()
        self.Concat_w = Variable(self.Concat_w, requires_grad=True)
        self.Concat_w2 = torch.FloatTensor(self.n_inputs * 2, 1).cuda()
        self.Concat_w2 = Variable(self.Concat_w2, requires_grad=True)
        self.E_cell = nn.GRUCell(n_inputs,n_hidden_e).cuda()
        self.O_cell = nn.GRUCell(n_inputs, n_hidden_o).cuda()
        self.weight_confusion =  torch.FloatTensor(1).cuda()
        self.weight_confusion = Variable(self.weight_confusion, requires_grad=True)
        self.L_cell = nn.GRUCell(n_inputs, n_hidden_o).cuda()
        self.relu = nn.ReLU(inplace=True)

    def forward(self,edges_all,rois,whole,local,boxes):
        n_steps = 2
        n_boxes = boxes  # train 128, test 256
        n_inputs = self.n_inputs  # edit D  2048
        # n_hidden_o = 2048
        # n_hidden_e = 2048

        # ofe = edges
        ofa = edges_all
        # ofo, ofs = tf.split(input[0], [n_boxes, 1], 0)

        fo = rois.view([n_boxes, n_inputs])
        fs = whole.view([n_boxes, n_inputs])
        fl = local.view([n_boxes, n_inputs])

        # fs = torch.cat(n_boxes * [fs], 0)
        fs = fs.view([n_boxes, 1, n_inputs])
        fl = fl.view([n_boxes, 1, n_inputs])

        # fe = ofe.view([n_boxes * n_boxes, 12]).cuda()
        fa = ofa.view([n_boxes, 9]).cuda()

        # u = torch.get_variable('u', [12, 1], initializer=tf.contrib.layers.xavier_initializer())
        # u = torch.FloatTensor(12, 1)
        # nn.init.xavier_uniform(u)

        # Form 1
        # W = tf.get_variable('CW', [n_inputs, n_inputs], initializer = tf.orthogonal_initializer())

        # Form 2
        # Concat_w = torch.get_variable('Concat_w', [n_inputs * 2, 1], initializer=tf.contrib.layers.xavier_initializer())
        # Concat_w = torch.FloatTensor(n_inputs * 2, 1)
        # Concat_w = Concat_w.type(torch.cuda.FloatTensor)
        # nn.init.xavier_uniform(Concat_w)

        # E_cell = nn.GRUCell(2048, n_hidden_e)
        # E_cell = E_cell.cuda()
        # O_cell = nn.GRUCell(2048, n_hidden_o)
        # O_cell = O_cell.cuda()
        # PE = torch.mm(fe, self.weight_u)
        PA = torch.mm(fa, self.weight_ua)
        #PA = torch.cat(n_boxes * [PA], 1)
        # PE = nn.ReLU(PE.view(torch.mm(fe, u), [n_boxes, n_boxes]))
        # PE = F.relu(PE.view([n_boxes, n_boxes]))
        PA = F.relu(PA.view([n_boxes, 1]))
        oinput = fs[:, 0, :]
        # oinput = oinput.type(torch.cuda.FloatTensor)
        linput = fl[:, 0, :]

        hi = fo
        # hi = hi.type(torch.cuda.FloatTensor)

        for t in range(n_steps):
            # with tf.variable_scope('e_gru', reuse=(t!=0)):
            #        _, he = E_cell(inputs= einput, state = he)
            residual = hi
            # X = torch.cat(n_boxes * [hi], 0)
            # X = X.view([n_boxes * n_boxes, n_inputs])
            # Y = hi  # Y = fo: 128 * 4096
            # VE form 1:
            # VE = tf.nn.tanh(tf.matmul(tf.matmul(Y, W), tf.transpose(Y))) # Y*W*Y_T = (128 * 4096) * (4096 * 4096) * (4096 * 128) = 128 * 128
            # VE form 2:
            # Y1 = torch.cat(n_boxes * [Y], 1)
            # print (1)
            # Y1 = Y1.view([n_boxes * n_boxes, n_inputs])
            # Y2 = torch.cat(n_boxes * [Y], 0)
            # Y2 = Y2.view([n_boxes * n_boxes, n_inputs])
            # VE = torch.mm(torch.cat([Y1, Y2], 1), self.Concat_w)
            # # VE = nn.Tanh(torch.reshape(torch.mm(torch.cat([Y1, Y2], 1), Concat_w), [n_boxes, n_boxes]))
            # VE = VE.view([n_boxes, n_boxes])
            # VE = F.tanh(VE)
            # E = torch.mul(PE.type(torch.cuda.FloatTensor), VE)
            # Z = F.softmax(E)  # edge relationships
            # X = X.view([n_boxes, n_boxes, n_inputs])  # Nodes
            # M = Z.view([n_boxes, n_boxes, 1]) * X  # messages

            ###########  Maxpooling ##########
            # M = torch.max(M, 1)  # intergated message
            # me_max = M[0].view([n_boxes, 1, n_inputs])

            # ####### Attention ######
            # w_u = torch.FloatTensor(n_boxes, 1).cuda()
            # nn.init.xavier_uniform(w_u)
            # M = torch.mul(M,w_u)
            # # M = F.softmax(M)
            # M = torch.sum(M,1)
            # me_max = M.view([n_boxes, 1, n_inputs])

            # einput = me_max[:, 0, :]

            #oinput:

            X = torch.cat(n_boxes * [hi], 0)
            X = X.view([n_boxes * n_boxes, n_inputs])

            Y = hi  # Y = fo: 128 * 4096
            Y1 = torch.cat(n_boxes * [Y], 1)
            Y1 = Y1.view([n_boxes * n_boxes, n_inputs])
            Y2 = torch.cat(n_boxes * [oinput], 0)
            Y2 = Y2.view([n_boxes * n_boxes, n_inputs])

            VA = torch.mm(torch.cat([Y1, Y2], 1), self.Concat_w2)
            VA = VA.view([n_boxes, n_boxes])
            VA = F.tanh(VA)
            A = torch.mul(PA.type(torch.cuda.FloatTensor), VA)
            Z = F.softmax(A)
            X = X.view([n_boxes, n_boxes, n_inputs])  # Nodes
            M = Z.view([n_boxes, n_boxes, 1]) * X
            me_max = M[0].view([n_boxes, 1, n_inputs])
            oinput = me_max[:, 0, :]

            # with torch.variable_scope('o_gru', reuse=(t != 0)):
            # ho1, hi1 = O_cell(oinput, hi)
            hi1 = self.O_cell(oinput, hi)
            # with torch.variable_scope('e_gru', reuse=(t != 0)):
            # hi2 = self.E_cell(einput, hi)

            # hi3 = self.L_cell(linput, hi)

            # maxpooling
            # hi = tf.maximum(hi1, hi2)

            # meanpooling
            # hi = torch.cat([hi1, hi2], 0)
            # hi = hi.view([2, n_boxes, n_inputs])
            # hi = torch.mean(hi, 0)

            # weight_confusion
            hi = torch.mul(hi1, self.weight_confusion[0]) # + torch.mul(hi2, self.weight_confusion[1]) \
                 #+ torch.mul(hi3, self.weight_confusion[2])
            hi += residual
            hi = self.relu(hi)

        return hi




