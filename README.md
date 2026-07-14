# Memory-alignment-loss-for-constrained-RNNs-on-tiny-devices
Public Domain/CC0 Memory alignment loss for constrained RNNs on tiny devices lstm, coherence, tinyML, embedded, sequence-stability, custom-loss

\mathcal{L}(t) = 1 - \frac{\mathbf{z}_{seq}(t) \cdot \left( \frac{\mathbf{h}_{t-1} + \sigma\Big( \mathbf{W}_f \mathbf{x}_t + \mathbf{U}_f \mathbf{h}_{t-1} + \mathbf{b}_f \Big) \odot \tanh\Big( \mathbf{W}_i \mathbf{x}_t + \mathbf{U}_i \mathbf{h}_{t-1} + \mathbf{b}_i \Big)}{\left\| \mathbf{h}_{t-1} + \sigma\Big( \mathbf{W}_f \mathbf{x}_t + \mathbf{U}_f \mathbf{h}_{t-1} + \mathbf{b}_f \Big) \odot \tanh\Big( \mathbf{W}_i \mathbf{x}_t + \mathbf{U}_i \mathbf{h}_{t-1} + \mathbf{b}_i \Big) \right\|_2} \right)}{\|\mathbf{z}_{seq}(t)\|_2}

No Copyright
The person who associated a work with this deed has dedicated the work to the public domain by waiving all of his or her rights to the work worldwide under copyright law, including all related and neighboring rights, to the extent allowed by law.
You can copy, modify, distribute and perform the work, even for commercial purposes, all without asking permission.
