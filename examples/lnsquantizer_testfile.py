import os
import sys
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '../../..'))
sys.path.insert(1, os.path.join(sys.path[0], '..'))

# from codegen.RPAG_graph_parser import generate_rpag_mcm, parse_graph, run_vivado_synthesis
# from codegen.MUXman import quant_weights
from mquat.QuantizerBase import DEFAULT_DATATYPE
from training.TrainingTools import QTraining

import mquat as mq  # noqa: E402
import models.LeNet5_Like as ln5  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow_datasets as tfds  # noqa: E402
import tensorflow as tf  # noqa: E402

# First train the model without any quantization. Just a normal floating-point 32bit training!
# -tt float -desc test5

# Then, train the model with some quantization (QAT - quantization aware training)
# Replace this fixed-point training with your LNS training.
# -tt lns -qd layer-wise -b-bits 10 -w-bits 8 -a-bits 10 -desc test5 --checkpoint "data/training/LeNetLike/float/test5/model_best_q.npz" -no-skip

internal_quant = mq.FlexPointQuantizer("internal",8)
quant = mq.LNSQuantizer("test",internal_quant)
data = np.array([-3.4,-1.1,-1.0,-0.9,-0.5,-0.1, 0.0, 0.1, 0.5, 0.6, 1.0, 1.1, 5.5])
data = np.array([-9.0, -0.7, -0.3, -0.25, -0.22, -0.14, 0.1, 0.1, 0.12, 0.03])
q_data = quant(data)
for x, quant_x in zip(data, q_data):
    print(x, quant_x)
print(internal_quant.get_sfix_settings())

