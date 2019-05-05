# learning_dynamics

TODO:
- [ ] Nonlinear encoder and decoder (try combination)
- [ ] Train curriculum learning with more steps (1to3 3to5)
- [ ] Update forward function for curriculum learning
- [ ] LQR!!!!!!!!!!
- [ ] Better naming for checkpoints and losses


Notes:
- **Nonlinear vs linear encoding:** when trained on the same number of epochs, the linear encoding performed better, however, the loss of the last epoch for linear encoding was smaller than the nonlinear encoding. This might mean that nonlinear encoding needs more training.
- **Hidden size:** For the nonlinear encoding, increasing the hidden size significantly improved the performance.
- **Early stopping:**
