import torch
import numpy as np
from dm_control import suite
from networks.lstm_autoencoder import LSTMAutoEncoder
import scipy.linalg as linalg


def lqr_pendulum():

    env = suite.load(domain_name="pendulum", task_name="swingup", task_kwargs={'time_limit': 5.0})
    action_spec = env.action_spec()
    env.physics.model.dof_damping[0] = 0.0
    time_step = env.reset()

    checkpoint_path = './checkpoints/lstm_auto_encoder/checkpoint_5k.pth'
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model = LSTMAutoEncoder(input_size=3, action_size=1, hidden_size=16, num_layers=1)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    w_a = state_dict['f_hidden.weight']
    w_b = state_dict['f_action.weight']

    A = w_a.numpy()
    B = w_b.numpy()

    Q = np.eye(16) * 100.0
    R = np.eye(1) * 0.1

    print('A, B shape: {} {}'.format(A.shape, B.shape))

    x_init = torch.Tensor([time_step.observation['orientation'][0], time_step.observation['orientation'][1],
                           time_step.observation['velocity'][0]]).view(1, 1, 3)

    x_des = torch.Tensor([-1.0, 0.0, 0.0]).view(1, 1, 3)
    print('Initial state: {}'.format(x_init.numpy().ravel()))
    print('Desired state: {}'.format(x_des.numpy().ravel()))

    with torch.no_grad():
        x_des_hidden, _ = model.lstm(x_des)
        x_init_hidden, _ = model.lstm(x_init)

    x_cur_hidden = x_init_hidden.view(16, 1).numpy()
    x_des_hidden = x_des_hidden.view(16, 1).numpy()

    def solve_lqr(A, B, Q, R):
        print('Solving Ricatti...')
        # discrete
        P = linalg.solve_discrete_are(A, B, Q, R)
        # K = np.linalg.solve(B.T.dot(P).dot(B) + R, B.T.dot(P).dot(A))
        # print(np.allclose(A.T.dot(P).dot(A) - P - A.T.dot(P).dot(B).dot(K), -Q))
        K = np.dot(linalg.inv(R + np.dot(B.T, np.dot(P, B))), np.dot(B.T, np.dot(P, A)))

        # continous
        # P = linalg.solve_continuous_are(A, B, Q, R)
        # K = - linalg.inv(R).dot(B.T.dot(P))
        return K

    success = False
    for i in range(100):
        K = - solve_lqr(A, B, Q, R)
        # u = -K * x_cur_hidden
        u = np.dot(K, (x_cur_hidden- x_des_hidden))
        # u = np.dot(K, x_cur_hidden)

        x_cur_hidden = np.dot(A, x_cur_hidden) + np.dot(B, u)
        # x_cur_hidden = np.dot(A, x_cur_hidden) - np.dot(np.dot(B, K), (x_cur_hidden - x_des_hidden))

        with torch.no_grad():
            x_cur_hidden_torch = torch.from_numpy(x_cur_hidden).float()
            x_cur_hidden_torch = x_cur_hidden_torch.view(1, 1, 16)
            decoded = model.f_decoder(x_cur_hidden_torch)

        # if np.sum(x_des_hidden - x_cur_hidden) < 0.001:
        #     break
        print(decoded)
        # print(np.sum(x_des_hidden - x_cur_hidden))
        # print(np.vstack((x_des_hidden.transpose(), x_cur_hidden.transpose())))
        # error = x_cur_hidden - x_des_hidden
        # print('error: {}' .format((x_des_hidden - x_cur_hidden).transpose()))
        # break


if __name__ == '__main__':
    lqr_pendulum()