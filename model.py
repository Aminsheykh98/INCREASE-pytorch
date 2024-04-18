import torch
import torch.nn as nn
import torch.nn.functional as F


class Two_layers(nn.Module):
    def __init__(self, activations, inputs, outputs):
        super(Two_layers, self).__init__()
        self.layer1 = nn.Linear(inputs[0], outputs[0])
        self.act1 = activations[0]()
        self.layer2 = nn.Linear(inputs[1], outputs[1])
        self.act2 = activations[1]()
        self.activations = activations

    def forward(self, x):
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        x = self.act2(x)
        return x


class Model(nn.Module):
    def __init__(self, mean, std, args):
        super(Model, self).__init__()
        self.args = args
        self.mean = mean
        self.std = std
        self.h = args.h
        self.K = args.K
        # d = args.d
        self.d = args.d

        self.linset1_fw = Two_layers(activations=[nn.ReLU, nn.Identity], inputs=[1, self.d], outputs=[self.d, self.d])
        self.linset1_bw = Two_layers(activations=[nn.ReLU, nn.Identity], inputs=[1, self.d], outputs=[self.d, self.d])

        self.linset2_fw = Two_layers(activations=[nn.ReLU, nn.Identity], inputs=[self.d, self.d], outputs=[self.d, self.d])
        self.linset2_bw = Two_layers(activations=[nn.ReLU, nn.Identity], inputs=[self.d, self.d], outputs=[self.d, self.d])

        self.linset3_fw = Two_layers(activations=[nn.ReLU, nn.Identity], inputs=[self.d, self.d], outputs=[self.d, self.d])
        self.linset3_bw = Two_layers(activations=[nn.ReLU, nn.Identity], inputs=[self.d, self.d], outputs=[self.d, self.d])

        self.linset4_fw = Two_layers(activations=[nn.ReLU, nn.Tanh], inputs=[self.d, self.d], outputs=[self.d, self.d])
        self.linset4_bw = Two_layers(activations=[nn.ReLU, nn.Tanh], inputs=[self.d, self.d], outputs=[self.d, self.d])

        self.linset5_fw = nn.Linear(self.d, self.d)
        self.linset5_bw = nn.Linear(self.d, self.d)

        self.linset6_fw = Two_layers(activations=[nn.ReLU, nn.Identity], inputs=[self.d, self.d], outputs=[self.d, self.d])
        self.linset6_bw = Two_layers(activations=[nn.ReLU, nn.Identity], inputs=[self.d, self.d], outputs=[self.d, self.d])

        ## Temporal Modeling
        self.linTE = Two_layers(activations=[nn.ReLU, nn.Identity], inputs=[288, self.d], outputs=[self.d, self.d])
        self.liny = Two_layers(activations=[nn.ReLU, nn.Identity], inputs=[2*self.d, self.d], outputs=[self.d, self.d])
        self.linx = Two_layers(activations=[nn.ReLU, nn.Identity], inputs=[2*self.d, self.d], outputs=[self.d, self.d])
        self.ling = Two_layers(activations=[nn.ReLU, nn.ReLU], inputs=[self.d, self.d], outputs=[self.d, self.d])
        self.cell = nn.GRUCell(input_size=2*self.d, hidden_size=self.d)
        self.ling2 = nn.Linear(self.d, self.d)
        self.linset_out = Two_layers(activations=[nn.ReLU, nn.ReLU], inputs=[self.d, self.d], outputs=[self.d, self.d])
        self.lin_out = nn.Linear(self.d, 1)

    def forward(self, x_gp_fw, x_gp_bw, TE, gp_fw, gp_bw):
        N_target = x_gp_fw.shape[0]
        x_gp_fw = (x_gp_fw - self.mean) / self.std
        x_gp_bw = (x_gp_bw - self.mean) / self.std

        x_gp_fw = self.linset1_fw(x_gp_fw)
        x_gp_bw = self.linset1_bw(x_gp_bw)

        ## Spatial Aggregation
        gp_fw = torch.tile(gp_fw, dims=(1, self.h, 1, 1))
        gp_bw = torch.tile(gp_bw, dims=(1, self.h, 1, 1))

        y_gp_fw = torch.matmul(gp_fw, x_gp_fw)
        y_gp_bw = torch.matmul(gp_bw, x_gp_bw)
        y_gp_fw = self.linset2_fw(y_gp_fw)
        y_gp_bw = self.linset2_bw(y_gp_bw)

        x_gp_fw = self.linset3_fw(x_gp_fw)
        x_gp_bw = self.linset3_bw(x_gp_bw)

        x_gp_fw = torch.abs(y_gp_fw - x_gp_fw)
        x_gp_bw = torch.abs(y_gp_bw - x_gp_bw)

        x_gp_fw = torch.matmul(gp_fw, x_gp_fw)
        x_gp_bw = torch.matmul(gp_bw, x_gp_bw)

        x_gp_fw = self.linset4_fw(x_gp_fw)
        x_gp_bw = self.linset4_bw(x_gp_bw)

        x_gp_fw = self.linset5_fw(x_gp_fw)
        x_gp_bw = self.linset5_bw(x_gp_bw)

        y_gp_fw = x_gp_fw + y_gp_fw
        y_gp_bw = x_gp_bw + y_gp_bw

        y_gp_fw = self.linset6_fw(y_gp_fw)
        y_gp_bw = self.linset6_bw(y_gp_bw)

        ## Temporal Modeling
        TE = F.one_hot(TE.long(), num_classes=288).float()
        TE = self.linTE(TE)
        TE = torch.tile(input=TE , dims=(N_target, 1, 1))

        y = torch.concat((y_gp_fw, y_gp_bw), dim=-1)
        y = torch.squeeze(y, dim=2)
        y = self.liny(y)
        x = torch.concat((x_gp_fw, x_gp_bw), dim=-1)
        x = torch.squeeze(x, dim=2)
        x = self.linx(x)
        g1 = self.ling(x)

        g1 = 1 / torch.exp(g1)
        y = g1 * y
        y = torch.concat((y, TE), dim= -1)
        # print(y.shape)
        pred = []
        state = torch.zeros((N_target, self.d), device=self.args.device)
        for i in range(self.h):
            if i == 0:
                g2 = F.relu(self.ling2(state))
                g2 = 1 / torch.exp(g2)
                state = g2 * state
                state = self.cell(y[:, i], state)
                pred.append(torch.unsqueeze(input=state, dim=1))

            else:
                g2 = F.relu(self.ling2(state))
                g2 = 1 / torch.exp(g2)
                state = g2 * state
                state = self.cell(y[:, i], state)
                pred.append(torch.unsqueeze(input=state, dim=1))

        pred = torch.concat(pred, dim=1)
        pred = self.linset_out(pred)
        pred = self.lin_out(pred)
        return pred * self.std + self.mean