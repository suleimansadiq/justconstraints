# ------------------------------------------------------------------------------
# This script performs a fully concretized export of a LeNet model's constraints,
# using 8-bit posit arithmetic for both weights/biases and image inputs.
#
# IMPORTANT FIXES:
# 1) The input is normalized to [-1,1], matching the training code.
# 2) We skip the ReLU on the final layer (fc3) so that layer is raw logits.
# 3) We output 8-bit posit bitstrings (binary) for all parameters & inputs.
# ------------------------------------------------------------------------------

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
# 2. Define LeNet with variable names (same as training)
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
    logits = tf.matmul(fc2, fc3_W) + fc3_b  # raw logits, no ReLU

    return logits

##############################################################################
# 3. Build graph & restore from 'posit8.ckpt'
##############################################################################
x_ph = tf.placeholder(tf_type, (None,32,32,1), name='inputs')
y_ph = tf.placeholder(tf.int32, (None,), name='labels')
logits = LeNet(x_ph)

saver = tf.train.Saver()
model_checkpoint_path = './posit8.ckpt'

with tf.Session() as sess:
    saver.restore(sess, model_checkpoint_path)

    # Extract all weights/biases
    weights_and_biases = {}
    print("\nLoaded Variables:")
    for var in tf.trainable_variables():
        arr = sess.run(var)
        weights_and_biases[var.name] = arr
        print(f"Loaded {var.name}, shape={arr.shape}, dtype={var.dtype}")

##############################################################################
# 4. Helper function: float -> 8-bit posit -> binary string
##############################################################################
def float_to_posit8_bits(val: float) -> str:
    """
    Cast 'val' to 8-bit posit, then return the 8-bit binary representation
    (like "10110110").
    """
    p = posit(val)  # e.g. np.posit8
    return np.binary_repr(p.view(np.uint8), width=8)

##############################################################################
# 5. Load one "non-black" MNIST image, resize to 32×32, but now normalized to [-1,1]
##############################################################################
(_, _), (X_test, y_test) = mnist.load_data()

def find_non_black_image(images, threshold=50):
    for idx in range(len(images)):
        if np.count_nonzero(images[idx]) >= threshold:
            return idx
    return -1

img_idx = find_non_black_image(X_test, 50)
if img_idx == -1:
    print("No suitable non-black image found; using idx=0.")
    img_idx = 0

print(f"Using MNIST test image index={img_idx} with label={y_test[img_idx]}")
img = X_test[img_idx]

# Resize from 28x28 -> 32x32
pil_img = Image.fromarray(img)
pil_resized = pil_img.resize((32,32), Image.LANCZOS)

# Convert to float
img_resized = np.array(pil_resized, dtype=np.float32)
# SHIFT & SCALE to [-1,1], matching training code
img_resized = (img_resized - 127.5) / 127.5

# For each pixel, store the 8-bit posit bitstring
input_posit_map = {}
for i in range(32):
    for j in range(32):
        val_float = float(img_resized[i,j])
        bits = float_to_posit8_bits(val_float)
        input_posit_map[f"x_{i}_{j}_0"] = bits

##############################################################################
# 6. Convert W/B arrays to 8-bit posit bitstrings for each element
##############################################################################
encoded_params = {}
for var_name, arr in weights_and_biases.items():
    shape = arr.shape
    encoded_arr = np.empty(shape, dtype=object)
    it = np.nditer(arr, flags=['multi_index'])
    for val in it:
        idx = it.multi_index
        encoded_arr[idx] = float_to_posit8_bits(val)
    encoded_params[var_name] = encoded_arr

##############################################################################
# 7. Constraint generators
##############################################################################

def generate_conv_constraints(w_4d, b_1d, in_shape, layer_name, index_start, prev_layer=None):
    """
    w_4d: shape=(fh,fw,in_d,out_d), each is 8-bit posit bitstring
    b_1d: shape=(out_d,), 8-bit posit bitstrings
    in_shape: (H,W,Depth)
    """
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
                # y_ is pre-activation
                constraints.append(
                    f"{idx}: y_{layer_name}_{i}_{j}_{out_c} = {expr}"
                )
                idx += 1
                # z_ is ReLU output
                constraints.append(
                    f"{idx}: z_{layer_name}_{i}_{j}_{out_c} = if y_{layer_name}_{i}_{j}_{out_c} >= 00000000 then y_{layer_name}_{i}_{j}_{out_c} else 00000000"
                )
                idx += 1
                node_count += 1

    return constraints, node_count, idx

def generate_maxpool_constraints(in_shape, prev_layer_name, out_layer_name, index_start):
    """
    Standard 2x2 max-pool
    in_shape: e.g. (28,28,6)
    out => (14,14,6)
    """
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
                expr = (f"max("
                        f"z_{prev_layer_name}_{i*stride}_{j*stride}_{d}, "
                        f"z_{prev_layer_name}_{i*stride+1}_{j*stride}_{d}, "
                        f"z_{prev_layer_name}_{i*stride}_{j*stride+1}_{d}, "
                        f"z_{prev_layer_name}_{i*stride+1}_{j*stride+1}_{d})")
                constraints.append(
                    f"{idx}: m_{out_layer_name}_{i}_{j}_{d} = {expr}"
                )
                idx += 1
                node_count += 1
    return constraints, node_count, idx

def generate_flatten_constraints(prev_layer_name, in_shape, out_layer_name, index_start):
    """
    Flatten 3D => 1D indexing: m_{out_layer_name}_k = m_{prev_layer_name}_{i}_{j}_{d}
    """
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
    """
    w_2d: shape=(in_size,out_size), 8-bit posit bits
    b_1d: shape=(out_size,), 8-bit posit bits
    in_shape: e.g. (120,) => in_size=120
    final_layer=True => skip the ReLU
    """
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

        # y_ is pre-activation
        constraints.append(
            f"{idx}: y_{layer_name}_{out_i} = {expr}"
        )
        idx += 1
        # If final_layer, skip ReLU
        if not final_layer:
            constraints.append(
                f"{idx}: z_{layer_name}_{out_i} = if y_{layer_name}_{out_i} >= 00000000 then y_{layer_name}_{out_i} else 00000000"
            )
            idx += 1
        node_count += 1

    return constraints, node_count, idx

##############################################################################
# 8. Build constraints layer-by-layer
##############################################################################
all_constraints = []
idx = 1
total_nodes = 0

# Layer 1 (conv1 => 28x28x6)
c, n, idx = generate_conv_constraints(
    encoded_params["Variable:0"],
    encoded_params["Variable_1:0"],
    (32,32,1),  # input shape
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

# Layer 2 (conv2 => 10,10,16)
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

# fc1 => 400->120 (ReLU)
c, n, idx = generate_fc_constraints(
    encoded_params["Variable_4:0"],
    encoded_params["Variable_5:0"],
    (400,),
    "fc1",
    idx,
    prev_layer="flatpool2",
    prev_is_pool=True,
    final_layer=False  # has ReLU
)
all_constraints.extend(c)
total_nodes += n

# fc2 => 120->84 (ReLU)
c, n, idx = generate_fc_constraints(
    encoded_params["Variable_6:0"],
    encoded_params["Variable_7:0"],
    (120,),
    "fc2",
    idx,
    prev_layer="fc1",
    prev_is_pool=False,
    final_layer=False  # has ReLU
)
all_constraints.extend(c)
total_nodes += n

# fc3 => 84->10 (NO ReLU => final logits)
c, n, idx = generate_fc_constraints(
    encoded_params["Variable_8:0"],
    encoded_params["Variable_9:0"],
    (84,),
    "fc3",
    idx,
    prev_layer="fc2",
    prev_is_pool=False,
    final_layer=True  # skip ReLU here
)
all_constraints.extend(c)
total_nodes += n

##############################################################################
# 9. Write constraints to file
##############################################################################
output_file = "fully_posit8_constraints_bitstring.txt"
with open(output_file, "w") as f:
    for line in all_constraints:
        f.write(line + "\n\n")

print(f"\nDone. Wrote {len(all_constraints)} constraints, total nodes={total_nodes}")
print(f"Constraints saved to {output_file}")
