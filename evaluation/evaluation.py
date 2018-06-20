# -*- coding: utf-8 -*-
# @Time    : 2017/9/3 下午1:18
# @Author  : Zhixin Piao 
# @Email   : piaozhx@shanghaitech.edu.cn

import scipy.io as sio
from base.base_network import *


class Trace2Trace:
    def __init__(self, data_path):
        self.data_path = data_path
        self.pedestrian_num = 20
        self.hidden_size = 40

        # input
        self.input_frame = 5
        self.input_size = 2 * self.input_frame

        # target
        self.target_frame = 5
        self.target_size = 2
        self.window_size = 1

        self.min_target_frame = 5
        self.max_target_frame = 25

    def load_data(self):
        # load data
        data = np.load(self.data_path)
        train_X, train_Y = data['train_X'], data['train_Y']
        test_X, test_Y = data['test_X'], data['test_Y']

        if self.use_gpu:
            self.test_input_traces = Variable(torch.from_numpy(test_X).cuda())
            self.test_target_traces = Variable(torch.from_numpy(test_Y).cuda())

            self.train_input_traces = Variable(torch.from_numpy(train_X).cuda())
            self.train_target_traces = Variable(torch.from_numpy(train_Y).cuda())
        else:
            self.test_input_traces = Variable(torch.from_numpy(test_X))
            self.test_target_traces = Variable(torch.from_numpy(test_Y))

            self.train_input_traces = Variable(torch.from_numpy(train_X))
            self.train_target_traces = Variable(torch.from_numpy(train_Y))

    def load_multi_frame_data(self):
        with open('data/multi_frame_GC.data', 'r') as f:
            data = pickle.load(f)

        train_X, train_Y = data['train_X'], data['train_Y']
        test_X, test_Y = data['test_X'], data['test_Y']

        if self.use_gpu:
            self.train_input_traces_frame_list = [Variable(torch.FloatTensor(f_train_X).cuda()) for f_train_X in train_X]
            self.train_target_traces_frame_list = [Variable(torch.FloatTensor(f_train_Y).cuda()) for f_train_Y in train_Y]

            self.test_input_traces_frame_list = [Variable(torch.FloatTensor(f_test_X).cuda()) for f_test_X in test_X]
            self.test_target_traces_frame_list = [Variable(torch.FloatTensor(f_test_Y).cuda()) for f_test_Y in test_Y]
        else:
            self.train_input_traces_frame_list = [Variable(torch.FloatTensor(f_train_X)) for f_train_X in train_X]
            self.train_target_traces_frame_list = [Variable(torch.FloatTensor(f_train_Y)) for f_train_Y in train_Y]

            self.test_input_traces_frame_list = [Variable(torch.FloatTensor(f_test_X)) for f_test_X in test_X]
            self.test_target_traces_frame_list = [Variable(torch.FloatTensor(f_test_Y)) for f_test_Y in test_Y]

    def load_model(self, model_path):
        self.encoder_net = EncoderNetWithLSTM(self.pedestrian_num, self.input_size, self.hidden_size, self.use_gpu)
        self.decoder_net = DecoderNet(self.pedestrian_num, self.target_size, self.hidden_size, self.window_size)
        self.regression_net = RegressionNet(self.pedestrian_num, self.target_size, self.hidden_size)
        self.attn = Attention()

        self.encoder_net.load_state_dict(torch.load('%s/encoder_net.pkl' % model_path))
        self.decoder_net.load_state_dict(torch.load('%s/decoder_net.pkl' % model_path))
        self.regression_net.load_state_dict(torch.load('%s/regression_net.pkl' % model_path))

        if self.use_gpu:
            self.encoder_net.cuda()
            self.decoder_net.cuda()
            self.regression_net.cuda()

    def main_compute_step(self, batch_input_traces, batch_target_traces):
        batch_size = batch_input_traces.size(0)
        target_traces = batch_input_traces[:, :, self.input_frame - 1]
        encoder_hidden = self.encoder_net.init_hidden(batch_size)

        # run LSTM in observation frame
        for i in xrange(self.input_frame - 1):
            encoder_output, encoder_hidden = self.encoder_net(batch_input_traces[:, :, i], encoder_hidden)

        regression_list = []
        Attn_list = []
        inner_Attn_list = []
        current_batch_input_traces = batch_input_traces[:, :, self.input_frame - 1]

        for i in xrange(self.target_frame):
            # NN with Attention
            target_hidden_traces = self.decoder_net(target_traces)
            inner_Attn, Attn_nn = self.attn(target_hidden_traces, target_hidden_traces)
            Attn_list.append(Attn_nn)
            inner_Attn_list.append(inner_Attn)

            # target_attn_hidden_traces_nn = torch.bmm(Attn_nn, target_hidden_traces)

            # LSTM with Attention
            input_hidden_traces, encoder_hidden = self.encoder_net(current_batch_input_traces, encoder_hidden)
            # Attn_lstm = self.attn(input_hidden_traces, input_hidden_traces)
            input_attn_hidden_traces_lstm = torch.bmm(Attn_nn, input_hidden_traces)

            # concat LSTM hidden with Attention and NN hidden with Attention
            # input_attn_hidden_traces = torch.cat((input_attn_hidden_traces_lstm, target_attn_hidden_traces_nn), 2)
            input_attn_hidden_traces = input_attn_hidden_traces_lstm

            # predict next frame traces
            regression_traces = self.regression_net(input_attn_hidden_traces, target_hidden_traces, target_traces)
            target_traces = regression_traces
            regression_list.append(regression_traces)

            current_batch_input_traces = regression_traces

        regression_traces = torch.stack(regression_list, 2)
        Attn_cube = torch.stack(Attn_list, 1)
        inner_Attn_cube = torch.stack(inner_Attn_list, 1)

        # compute loss

        L2_square_loss = ((batch_target_traces - regression_traces) ** 2).sum() / self.pedestrian_num
        MSE_loss = ((batch_target_traces - regression_traces) ** 2).sum(3).sqrt().mean()

        self.loss = L2_square_loss

        return L2_square_loss.data[0], MSE_loss.data[0], regression_traces, Attn_cube, inner_Attn_cube

    def get_data(self, data_type, batch_size):
        if data_type == 'test':
            input_traces = self.test_input_traces[:batch_size]
            target_traces = self.test_target_traces[:batch_size]
        elif data_type == 'train':
            input_traces = self.train_input_traces[:batch_size]
            target_traces = self.train_target_traces[:batch_size]

        else:
            return None, None

        if batch_size <= 0:
            self.batch_size = input_traces.size(0)
            return input_traces, target_traces
        else:
            self.batch_size = batch_size
            return input_traces[:batch_size], target_traces[:batch_size]

    def get_multi_data(self, target_frame, data_type, batch_size):
        if data_type == 'test':
            input_traces = self.test_input_traces_frame_list[target_frame][:batch_size]
            target_traces = self.test_target_traces_frame_list[target_frame][:batch_size]
        elif data_type == 'train':
            input_traces = self.train_input_traces_frame_list[target_frame][:batch_size]
            target_traces = self.train_target_traces_frame_list[target_frame][:batch_size]

        else:
            return None, None

        if batch_size <= 0:
            return input_traces, target_traces
        else:
            return input_traces[:batch_size], target_traces[:batch_size]

    # input: (B, pedestrian_num, input_size(input_frame * 2)) -> (B, pedestrian_num, input_frame, 2)
    def reshape_input_traces(self, input_traces):
        input_traces_x = input_traces[:, :, :5]
        input_traces_y = input_traces[:, :, 5:]

        return torch.stack([input_traces_x, input_traces_y], 3)

    def draw_predict_image(self, input_traces, target_traces, predict_traces, dir_path):
        format_input_traces = input_traces.data.cpu().numpy()
        format_target_traces = target_traces.data.cpu().numpy()
        format_predict_traces = predict_traces.data.cpu().numpy()
        batch_size = input_traces.size(0)

        for k in xrange(batch_size):
            k_dir_path = '%s/%s' % (dir_path, k)
            if not os.path.exists(k_dir_path):
                os.system('mkdir -p %s' % k_dir_path)

            fig = plt.figure(figsize=(10, 10))

            for f in xrange(1, 6):
                for i in xrange(20):
                    plt.scatter(format_input_traces[k, i, :, 1], format_input_traces[k, i, :, 0], color='r')
                    plt.text(format_input_traces[k, i, 0, 1], format_input_traces[k, i, 0, 0], str(i), ha='center', va='bottom', fontsize=12)
                    plt.scatter(format_target_traces[k, i, :f, 1], format_target_traces[k, i, :f, 0], color='b')
                    plt.scatter(format_predict_traces[k, i, :f, 1], format_predict_traces[k, i, :f, 0], color='g')

                fig.savefig('%s/%d.png' % (k_dir_path, f - 1))

    def draw_predict_attention(self, Attn_cube, dir_path):
        batch_size = Attn_cube.size(0)
        Attn_cube = Attn_cube.data.cpu().numpy()
        row, col = Attn_cube.shape[2], Attn_cube.shape[3]
        locator = MultipleLocator(1)

        for k in xrange(batch_size):
            k_dir_path = '%s/%s' % (dir_path, k)
            if not os.path.exists(k_dir_path):
                os.system('mkdir -p %s' % k_dir_path)

            for i in xrange(5):
                fig = plt.figure(figsize=(15, 15))
                ax = fig.add_subplot(111)
                ax.set_title('Attention')

                ax.xaxis.set_major_locator(locator)
                ax.yaxis.set_major_locator(locator)
                plt.imshow(Attn_cube[k, i])
                for r in xrange(row):
                    for c in xrange(col):
                        if Attn_cube[k, i, r, c] > 1e-2:
                            ax.text(c, r, '(%d,%d)' % (r, c), ha='center', va='bottom', fontsize=12, color='white')

                ax.set_aspect('equal')

                fig.savefig('%s/attn_%d.png' % (k_dir_path, i))

    def main(self, model_path, use_gpu=True):
        self.use_gpu = use_gpu
        self.load_model(model_path)

    def evaluate(self, algorithm_version, data_type='test', batch_size=-1, draw_predict_image=False):
        self.load_data()
        self.input_traces, self.ground_truth_traces = self.get_data(data_type, batch_size)
        self.dir_path = 'model/%s/predict_%s' % (algorithm_version, data_type)

        start_time = time.time()
        L2_square_loss, MSE_loss, self.regression_traces, self.Attn_cube, self.inner_Attn_cube = self.main_compute_step(self.input_traces,
                                                                                                                        self.ground_truth_traces)
        print (time.time() - start_time) / self.batch_size
        print ' L2_square_loss:%s,  MSE_loss: %s' % (L2_square_loss, MSE_loss)

        if draw_predict_image:
            print 'draw predict images...'
            self.draw_predict_image(self.input_traces, self.ground_truth_traces, self.regression_traces, self.dir_path)
            self.draw_predict_attention(self.Attn_cube, 'model/%s/predict_%s' % (algorithm_version, data_type))

    def evaluate_frame(self, data_type='test', batch_size=-1):
        self.load_multi_frame_data()
        for target_frame in xrange(self.min_target_frame, self.max_target_frame + 1):
            self.target_frame = target_frame
            input_traces, ground_truth_traces = self.get_multi_data(target_frame - self.min_target_frame, data_type, batch_size)
            L2_square_loss, MSE_loss, regression_traces, Attn_cube = self.main_compute_step(input_traces, ground_truth_traces)

            print 'target_frame: %d L2_square_loss:%s,  MSE_loss: %s' % (target_frame, L2_square_loss, MSE_loss)

    def save_data_by_idx(self, idx):
        format_input_traces = self.input_traces.data.cpu().numpy()
        format_target_traces = self.ground_truth_traces.data.cpu().numpy()
        format_predict_traces = self.regression_traces.data.cpu().numpy()
        Attn_cube = self.Attn_cube.data.cpu().numpy()
        batch_size = self.input_traces.size(0)

        k_dir_path = '%s/%s' % (self.dir_path, idx)
        if not os.path.exists(k_dir_path):
            os.system('mkdir -p %s' % k_dir_path)

        sio.savemat('%s/data.mat' % k_dir_path, {'input_traces': format_input_traces[idx],
                                                 'target_traces': format_target_traces[idx],
                                                 'predict_traces': format_predict_traces[idx],
                                                 'Attn': Attn_cube[idx]})
        print 'save %s/data.mat successfully!' % k_dir_path

    def draw_scatter(self):
        def compute_L2_Cube(input_hidden_traces, target_hidden_traces):
            standard_size = (input_hidden_traces.size(0), input_hidden_traces.size(1), input_hidden_traces.size(1))

            # L2 distance
            target_hidden_traces_square = (target_hidden_traces ** 2).sum(2).unsqueeze(2).expand(standard_size)
            input_hidden_traces_square = (input_hidden_traces ** 2).transpose(1, 2).sum(1).unsqueeze(1).expand(standard_size)
            input_target_mm = torch.bmm(target_hidden_traces, input_hidden_traces.transpose(1, 2))
            return (target_hidden_traces_square + input_hidden_traces_square - 2 * input_target_mm)

        batch_size = self.regression_traces.size(0)
        scatter_list = []
        for i in xrange(self.target_frame):
            L2_Cube = compute_L2_Cube(self.regression_traces[:, :, i, :], self.regression_traces[:, :, i, :]).contiguous()
            Attn_Cube = self.inner_Attn_cube[:, i].contiguous()

            scatter_list.append(torch.cat([L2_Cube.view(-1, 1), Attn_Cube.view(-1, 1)], 1))
        scatters = torch.cat(scatter_list, 0).data.cpu().numpy()

        sio.savemat('model/attn-l2.mat', {'scatters': scatters})

        fig = plt.figure(figsize=(10, 10))
        plt.scatter(scatters[:, 0], scatters[:, 1], color='r')
        fig.savefig('model/attn-l2.png')



def main():
    model = Trace2Trace('data/xy_data_set.npz')
    model.main(model_path='model/LSTM-NN', use_gpu=True)

    model.evaluate(algorithm_version='LSTM-NN', batch_size=-1, draw_predict_image=False)
    # model.draw_scatter()

    idxs = range(model.input_traces.size(0))
    for idx in idxs:
        model.save_data_by_idx(idx)


if __name__ == '__main__':
    main()
