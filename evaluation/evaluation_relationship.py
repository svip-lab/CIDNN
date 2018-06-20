# -*- coding: utf-8 -*-
# @Time    : 2017/9/3 下午1:18
# @Author  : Zhixin Piao 
# @Email   : piaozhx@shanghaitech.edu.cn


import scipy.io as sio
from PIL import Image, ImageDraw
from base.base_network import *


# target_frame is indeterminate
class DecoderNet(nn.Module):
    def __init__(self, target_size, hidden_size):
        super(DecoderNet, self).__init__()

        self.target_size = target_size  # 2
        self.hidden_size = hidden_size

        hidden1_size = 32
        hidden2_size = 64

        self.fc1 = torch.nn.Linear(target_size, hidden1_size)
        self.fc2 = torch.nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = torch.nn.Linear(hidden2_size, hidden_size)

    def forward(self, target_traces, p_num):
        # target_trace: (B, pedestrian_num, 2)
        hidden_list = []

        for i in xrange(p_num):
            target_trace = target_traces[:, i, :]
            hidden_trace = F.relu(self.fc1(target_trace))
            hidden_trace = F.relu(self.fc2(hidden_trace))
            hidden_trace = self.fc3(hidden_trace)

            hidden_list.append(hidden_trace)

        # stack all person
        hidden_traces = torch.stack(hidden_list, 1)

        # hidden_trace: (B, pedestrian_num, hidden_size)
        return hidden_traces


class L2_Distance(nn.Module):
    def __init__(self):
        super(L2_Distance, self).__init__()

    def forward(self, input_hidden_traces, target_hidden_traces):
        standard_size = (input_hidden_traces.size(0), input_hidden_traces.size(1), input_hidden_traces.size(1))

        # L2 distance
        target_hidden_traces_square = (target_hidden_traces ** 2).sum(2).unsqueeze(2).expand(standard_size)
        input_hidden_traces_square = (input_hidden_traces ** 2).transpose(1, 2).sum(1).unsqueeze(1).expand(standard_size)
        input_target_mm = torch.bmm(target_hidden_traces, input_hidden_traces.transpose(1, 2))
        inner_distance = target_hidden_traces_square + input_hidden_traces_square - 2 * input_target_mm

        return inner_distance


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, input_hidden_traces, target_hidden_traces):
        Attn = torch.bmm(target_hidden_traces, input_hidden_traces.transpose(1, 2))

        inner_Attn = Attn

        Attn_size = Attn.size()
        Attn = Attn - Attn.max(2)[0].unsqueeze(2).expand(Attn_size)
        exp_Attn = torch.exp(Attn)

        # batch-based softmax
        Attn = exp_Attn / exp_Attn.sum(2).unsqueeze(2).expand(Attn_size)
        return Attn, inner_Attn


class Trace2Trace:
    def __init__(self, data_path):
        self.data_path = data_path
        self.frame_num = 500

        self.target_size = 2
        self.hidden_size = 40
        self.accept_affinity_diff = 1e-2
        self.accept_location_diff = 1e-2
        self.accept_min_location = 2e-2

    def load_data(self):
        with open(self.data_path, 'r') as f:
            data = pickle.load(f)
        f_data = data['frame_data'][:self.frame_num]
        p_data = data['pedestrian_data']

        self.location_list = []
        for i in xrange(self.frame_num):
            location = []
            for pid in f_data[i]:
                location.append(p_data[pid][i])
            location = np.array([location])
            self.location_list.append(Variable(torch.FloatTensor(location)).cuda())

    def load_model(self, model_path):
        self.net1 = DecoderNet(self.target_size, self.hidden_size)
        self.attn = Attention()
        self.loc_attn = L2_Distance()

        self.net1.load_state_dict(torch.load('%s/decoder_net.pkl' % model_path))
        self.net1.cuda()

    def main(self, model_path, img_path):
        def data_standard(x):
            return ((x - x.min()) / (x.max() - x.min())).data.cpu().numpy()[0, :, :]

        def get_max_location_difference_pair():
            max_affinity_diff = 0

            '''location: (1, p_num, 2), hidden_location: (1, p_num. h)'''
            for t, location in enumerate(self.location_list):
                p_num = location.size(1)

                hidden_location = self.net1(location, p_num)
                inner_Attn = data_standard(self.attn(hidden_location, hidden_location)[1])
                loc_Attn = data_standard(self.loc_attn(location, location))

                for i in xrange(p_num):
                    for j1 in xrange(p_num):
                        for j2 in xrange(j1 + 1, p_num):
                            if j1 == i or j2 == i:
                                continue
                            affinity_diff = np.abs(inner_Attn[i, j1] - inner_Attn[i, j2])
                            loc_diff = np.abs(loc_Attn[i, j1] - loc_Attn[i, j2])
                            min_loc_diff = min(loc_Attn[i, j1], loc_Attn[i, j2])

                            if loc_diff < self.accept_location_diff and min_loc_diff < self.accept_min_location and affinity_diff > max_affinity_diff:
                                max_affinity_diff = affinity_diff
                                location_pkg = [location, inner_Attn, loc_Attn, (t, i, j1, j2)]
                print t
            print 'accept_location_diff:', self.accept_location_diff
            print 'max_loc_diff:', max_affinity_diff
            print 'location_pkg', location_pkg[3]

            return max_affinity_diff, location_pkg

        def draw_figure(max_loc_diff, location_pkg):
            location, inner_Attn, loc_Attn, (t, i, j1, j2) = location_pkg
            location = location.data.cpu().numpy()

            plt.scatter(location[0, :, 0], location[0, :, 1], color='r')
            p_num = location.shape[1]
            fig = plt.figure(figsize=(10, 10))

            for pid in xrange(p_num):
                if pid != i:
                    if pid == j1 or pid == j2:
                        plt.scatter(location[0, pid, 0], location[0, pid, 1], color='b')
                    else:
                        plt.scatter(location[0, pid, 0], location[0, pid, 1], color='r')
                    text = '%d(%.2f/%.2f)' % (pid, inner_Attn[i, pid], loc_Attn[i, pid])
                    plt.text(location[0, pid, 0], location[0, pid, 1] + 0.003, text, ha='center', va='bottom', fontsize=12)
                else:
                    plt.text(location[0, pid, 0], location[0, pid, 1], str(pid), ha='center', va='bottom', fontsize=12)
                    plt.scatter(location[0, pid, 0], location[0, pid, 1], color='g')

            fig.savefig('%s/%d-%d-%d-%d.png' % (img_path, t, i, j1, j2))

        self.load_model(model_path)
        self.load_data()

        if not os.path.exists(img_path):
            os.system('mkdir -p %s' % img_path)

        max_loc_diff, location_pkg = get_max_location_difference_pair()
        draw_figure(max_loc_diff, location_pkg)

    def draw_point(self, fid, i, j1, j2, img_path, save_path):
        def draw_circle(xy, fill, r=3):
            xy = [xy[0] * 1920, xy[1] * 1080]
            draw.ellipse((xy[0] - r, xy[1] - r, xy[0] + r, xy[1] + r), fill=fill)

        with open(self.data_path, 'r') as f:
            data = pickle.load(f)
        f_data = data['frame_data'][:self.frame_num]
        p_data = data['pedestrian_data']

        iid, j1id, j2id = f_data[fid][i], f_data[fid][j1], f_data[fid][j2]

        p_xy = p_data[iid][fid]
        j1_xy = p_data[j1id][fid]
        j2_xy = p_data[j2id][fid]

        image = Image.open(img_path)
        draw = ImageDraw.Draw(image)
        draw_circle(p_xy, (0, 255, 0, 0))
        draw_circle(j1_xy, (0, 0, 255, 0))
        draw_circle(j2_xy, (0, 0, 255, 0))

        # 68
        draw_circle(p_data[f_data[fid][103]][fid], (255, 0, 0, 0))
        draw_circle(p_data[f_data[fid][108]][fid], (255, 0, 0, 0))

        image.save(save_path)

    def get_affinity_scatter(self, model_path):
        self.load_model(model_path)
        self.load_data()

        Attn_list = []
        loc_Attn_list = []
        '''location: (1, p_num, 2), hidden_location: (1, p_num. h)'''
        for t, location in enumerate(self.location_list):
            p_num = location.size(1)

            hidden_location = self.net1(location, p_num)
            Attn, _ = self.attn(hidden_location, hidden_location)
            loc_Attn = self.loc_attn(location, location)

            Attn_list.append(Attn.view(-1, 1))
            loc_Attn_list.append(loc_Attn.view(-1, 1))

        Affinity = torch.cat(Attn_list).data.cpu().numpy()
        Location = torch.cat(loc_Attn_list).data.cpu().numpy()

        sio.savemat('model/attn-l2.mat', {'Affinity': Affinity, 'Location': Location})


def main():
    model = Trace2Trace('data/GC/data_set.data')
    model.get_affinity_scatter('model/LSTM-NN')
    # model.main('model/LSTM-NN', 'img')
    # model.draw_point(444, 102, 88, 125, 'img/008880.jpg', 'img/save.jpg')


if __name__ == '__main__':
    main()
