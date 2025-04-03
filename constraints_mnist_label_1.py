import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.contrib.layers import flatten
from PIL import Image

##############################################################################
# 1. Parse command-line arg for data type (posit8, etc.)
##############################################################################
if len(sys.argv) > 1:
    data_t = sys.argv[1]
    if data_t == 'posit8':
        posit = np.posit8
        tf_type = tf.posit8
    elif data_t == 'posit16':
        posit = np.posit16
        tf_type = tf.posit16
    elif data_t == 'posit32':
        posit = np.posit32
        tf_type = tf.posit32
    elif data_t == 'float16':
        posit = np.float16
        tf_type = tf.float16
    elif data_t == 'float32':
        posit = np.float32
        tf_type = tf.float32
    else:
        print("Unrecognized data type, defaulting to float32.")
        data_t = 'float32'
        posit = np.float32
        tf_type = tf.float32
else:
    data_t = 'float32'
    posit = np.float32
    tf_type = tf.float32

print(f"Selected data type: {tf_type}")

##############################################################################
# 2. Define LeNet (same as training)
##############################################################################
def LeNet(x):
    mu = 0
    sigma = 0.1

    conv1_W = tf.Variable(tf.truncated_normal(
        shape=(5, 5, 1, 6), mean=mu, stddev=sigma, dtype=tf_type), name='Variable')
    conv1_b = tf.Variable(tf.zeros(6, dtype=tf_type), name='Variable_1')
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1,1,1,1], padding='VALID') + conv1_b
    conv1 = tf.nn.relu(conv1)
    pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    conv2_W = tf.Variable(tf.truncated_normal(
        shape=(5, 5, 6, 16), mean=mu, stddev=sigma, dtype=tf_type), name='Variable_2')
    conv2_b = tf.Variable(tf.zeros(16, dtype=tf_type), name='Variable_3')
    conv2 = tf.nn.conv2d(pool1, conv2_W, strides=[1,1,1,1], padding='VALID') + conv2_b
    conv2 = tf.nn.relu(conv2)
    pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    fc0 = flatten(pool2)

    fc1_W = tf.Variable(tf.truncated_normal(
        shape=(400,120), mean=mu, stddev=sigma, dtype=tf_type), name='Variable_4')
    fc1_b = tf.Variable(tf.zeros(120, dtype=tf_type), name='Variable_5')
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b
    fc1 = tf.nn.relu(fc1)

    fc2_W = tf.Variable(tf.truncated_normal(
        shape=(120,84), mean=mu, stddev=sigma, dtype=tf_type), name='Variable_6')
    fc2_b = tf.Variable(tf.zeros(84, dtype=tf_type), name='Variable_7')
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b
    fc2 = tf.nn.relu(fc2)

    fc3_W = tf.Variable(tf.truncated_normal(
        shape=(84,10), mean=mu, stddev=sigma, dtype=tf_type), name='Variable_8')
    fc3_b = tf.Variable(tf.zeros(10, dtype=tf_type), name='Variable_9')
    # final logits, no ReLU
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits

##############################################################################
# 3. Restore from 'posit8.ckpt'
##############################################################################
x_ph = tf.placeholder(tf_type, (None,32,32,1), name='inputs')
y_ph = tf.placeholder(tf.int32, (None,), name='labels')
logits = LeNet(x_ph)

saver = tf.train.Saver()
model_checkpoint_path = './posit8.ckpt'

with tf.Session() as sess:
    saver.restore(sess, model_checkpoint_path)
    weights_and_biases = {}
    print("\nLoaded Variables:")
    for var in tf.trainable_variables():
        arr = sess.run(var)
        weights_and_biases[var.name] = arr
        print(f"Loaded {var.name}, shape={arr.shape}, dtype={var.dtype}")

##############################################################################
# 4. float -> 8-bit posit bits
##############################################################################
def float_to_posit8_bits(val: float) -> str:
    p = posit(val)
    return np.binary_repr(p.view(np.uint8), width=8)

##############################################################################
# 5. Load MNIST test, pick first sample labeled '1'
##############################################################################
(_, _), (X_test, y_test) = mnist.load_data()

def find_label_1(x_data, y_data):
    """
    Return first test idx labeled '1', or -1 if none found.
    """
    for idx in range(len(y_data)):
        if y_data[idx] == 1:
            return idx
    return -1

img_idx = find_label_1(X_test, y_test)
if img_idx == -1:
    print("No test image labeled 1 found; defaulting to idx=0.")
    img_idx = 0

print(f"Using MNIST test image index={img_idx} with label={y_test[img_idx]}")
img = X_test[img_idx]

# Resize (28->32), normalize to [-1,1]
pil_img = Image.fromarray(img)
pil_resized = pil_img.resize((32,32), Image.LANCZOS)
img_resized = np.array(pil_resized, dtype=np.float32)
img_resized = (img_resized - 127.5) / 127.5

# Build a dict of pixel -> bitstring
input_posit_map = {}
for i in range(32):
    for j in range(32):
        val_float = float(img_resized[i,j])
        bits = float_to_posit8_bits(val_float)
        input_posit_map[f"x_{i}_{j}_0"] = bits

##############################################################################
# 6. Convert all W/B arrays => 8-bit posit bits
##############################################################################
encoded_params = {}
for var_name, arr in weights_and_biases.items():
    shape = arr.shape
    encoded_arr = np.empty(shape, dtype=object)
    it = np.nditer(arr, flags=['multi_index'])
    for val in it:
        idx2 = it.multi_index
        encoded_arr[idx2] = float_to_posit8_bits(val)
    encoded_params[var_name] = encoded_arr

##############################################################################
# 7. Constraint generators
##############################################################################
def generate_conv_constraints(w_4d, b_1d, in_shape, layer_name, index_start, prev_layer=None):
    constraints = []
    node_count = 0
    fh, fw, in_d, out_d = w_4d.shape
    idx = index_start
    for out_c in range(out_d):
        bias_bits = b_1d[out_c]
        for i in range(in_shape[0] - fh + 1):
            for j in range(in_shape[1] - fw + 1):
                expr = f"{bias_bits}"
                for fh_ in range(fh):
                    for fw_ in range(fw):
                        for d_ in range(in_d):
                            w_bits = w_4d[fh_, fw_, d_, out_c]
                            if prev_layer is not None:
                                expr += f" + {w_bits}*m_{prev_layer}_{i+fh_}_{j+fw_}_{d_}"
                            else:
                                expr += f" + {w_bits}*{input_posit_map[f'x_{i+fh_}_{j+fw_}_{d_}']}"
                constraints.append(f"{idx}: y_{layer_name}_{i}_{j}_{out_c} = {expr}")
                idx += 1
                constraints.append(
                    f"{idx}: z_{layer_name}_{i}_{j}_{out_c} = if y_{layer_name}_{i}_{j}_{out_c} >= 00000000 then y_{layer_name}_{i}_{j}_{out_c} else 00000000"
                )
                idx += 1
                node_count += 1
    return constraints, node_count, idx

def generate_maxpool_constraints(in_shape, prev_layer_name, out_layer_name, index_start):
    constraints = []
    node_count = 0
    stride = 2
    out_h = in_shape[0] // 2
    out_w = in_shape[1] // 2
    out_d = in_shape[2]
    idx = index_start
    for d in range(out_d):
        for i in range(out_h):
            for j in range(out_w):
                expr = (
                    f"max("
                    f"z_{prev_layer_name}_{i*stride}_{j*stride}_{d}, "
                    f"z_{prev_layer_name}_{i*stride+1}_{j*stride}_{d}, "
                    f"z_{prev_layer_name}_{i*stride}_{j*stride+1}_{d}, "
                    f"z_{prev_layer_name}_{i*stride+1}_{j*stride+1}_{d})"
                )
                constraints.append(f"{idx}: m_{out_layer_name}_{i}_{j}_{d} = {expr}")
                idx += 1
                node_count += 1
    return constraints, node_count, idx

def generate_flatten_constraints(prev_layer_name, in_shape, out_layer_name, index_start):
    constraints = []
    node_count = 0
    idx = index_start
    (H, W, D) = in_shape
    k = 0
    for i in range(H):
        for j in range(W):
            for d in range(D):
                constraints.append(
                    f"{idx}: m_{out_layer_name}_{k} = m_{prev_layer_name}_{i}_{j}_{d}"
                )
                idx += 1
                node_count += 1
                k += 1
    return constraints, node_count, idx

def generate_fc_constraints(w_2d, b_1d, in_shape, layer_name, index_start,
                            prev_layer, prev_is_pool=False, final_layer=False):
    constraints = []
    node_count = 0
    in_size, out_size = w_2d.shape
    idx = index_start
    prefix = "m_" if prev_is_pool else "z_"
    for out_i in range(out_size):
        bias_bits = b_1d[out_i]
        expr = f"{bias_bits}"
        for in_j in range(in_size):
            w_bits = w_2d[in_j, out_i]
            expr += f" + {w_bits}*{prefix}{prev_layer}_{in_j}"
        constraints.append(f"{idx}: y_{layer_name}_{out_i} = {expr}")
        idx += 1
        if not final_layer:
            constraints.append(
                f"{idx}: z_{layer_name}_{out_i} = if y_{layer_name}_{out_i} >= 00000000 then y_{layer_name}_{out_i} else 00000000"
            )
            idx += 1
        node_count += 1
    return constraints, node_count, idx

##############################################################################
# 8. Build the constraints
##############################################################################
all_constraints = []
idx = 1
total_nodes = 0

# conv1 => shape(28,28,6)
c, n, idx = generate_conv_constraints(
    encoded_params["Variable:0"],
    encoded_params["Variable_1:0"],
    (32,32,1),
    "conv1",
    idx,
    prev_layer=None
)
all_constraints.extend(c)
total_nodes += n

# max-pool => (14,14,6)
mc, mn, idx = generate_maxpool_constraints((28,28,6), "conv1", "pool1", idx)
all_constraints.extend(mc)
total_nodes += mn

# conv2 => (10,10,16)
c, n, idx = generate_conv_constraints(
    encoded_params["Variable_2:0"],
    encoded_params["Variable_3:0"],
    (14,14,6),
    "conv2",
    idx,
    prev_layer="pool1"
)
all_constraints.extend(c)
total_nodes += n

# max-pool => (5,5,16)
mc, mn, idx = generate_maxpool_constraints((10,10,16), "conv2", "pool2", idx)
all_constraints.extend(mc)
total_nodes += mn

# flatten => shape=400
flat_c, flat_n, idx = generate_flatten_constraints("pool2", (5,5,16), "flatpool2", idx)
all_constraints.extend(flat_c)
total_nodes += flat_n

# fc1 => shape(400,120)
c, n, idx = generate_fc_constraints(
    encoded_params["Variable_4:0"],
    encoded_params["Variable_5:0"],
    (400,),
    "fc1",
    idx,
    prev_layer="flatpool2",
    prev_is_pool=True,
    final_layer=False
)
all_constraints.extend(c)
total_nodes += n

# fc2 => shape(120,84)
c, n, idx = generate_fc_constraints(
    encoded_params["Variable_6:0"],
    encoded_params["Variable_7:0"],
    (120,),
    "fc2",
    idx,
    prev_layer="fc1",
    prev_is_pool=False,
    final_layer=False
)
all_constraints.extend(c)
total_nodes += n

# fc3 => shape(84,10)
c, n, idx = generate_fc_constraints(
    encoded_params["Variable_8:0"],
    encoded_params["Variable_9:0"],
    (84,),
    "fc3",
    idx,
    prev_layer="fc2",
    prev_is_pool=False,
    final_layer=True
)
all_constraints.extend(c)
total_nodes += n

##############################################################################
# 9. Write constraints to file (single line, no trailing spaces)
##############################################################################
output_file = "fully_posit8_constraints_bitstring.txt"
with open(output_file, "w") as f:
    for line in all_constraints:
        line = line.rstrip()   # remove any trailing spaces
        f.write(line + "\n")   # exactly one newline, no blank lines

print(f"\nDone. Wrote {len(all_constraints)} constraints, total nodes={total_nodes}")
print(f"Constraints saved to {output_file}")
