# def forward(self, x):
    #     # h_t = torch.zeros(x.size(0), self.hidden_size, device='cuda')
    #     # c_t = torch.zeros(x.size(0), self.hidden_size, device='cuda')
    #
    #     S_hat = []
    #     for x_i, x_t in enumerate(x.chunk(x.size(1), dim=1)):
    #         x_t = x_t.view(x_t.size(0), x_t.size(2))
    #         # print(x_t.shape)
    #         s_t = x_t[:, 0:2]
    #         a_t = x_t[:, 3:4]
    #         # print(s_t.shape)
    #         s_t_enc = self.f_states_enc(s_t)
    #         # print(s_t_enc.shape)
    #         s_a_t = self.f_dec(self.f_enc(h_t) * self.f_actions(a_t))
    #
    #         h_t, c_t = self.lstm_cell(s_a_t, (h_t, c_t))
    #
    #
    #
    #         # print(s_a_t.shape)
    #         s_hat = self.f_states_dec(h_t)
    #
    #         S_hat += [s_hat]
    #
    #     S_hat = torch.stack(S_hat, 1)
    #     # print('Shat shape: {}'.format(S_hat.shape))
    #     return S_hat