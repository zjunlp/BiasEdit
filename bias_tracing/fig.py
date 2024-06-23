import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str,
                    help='the path of results',
                    default='results/ns3_r0_roberta-large_race/causal_trace/cases')
parser.add_argument("--num_layer", type=int,
                    help="The num of model layers.",
                    default=24)
parser.add_argument("--model_name", type=str,
                    help="The model name.",
                    default="roberta-large")
parser.add_argument("--bias", type=str,
                    help="The bias type.",
                    choices=["gender", "race"],
                    default="gender")
parser.add_argument("--num_sample", type=int, 
                    help="The num of samples",
                    default=500)
args = parser.parse_args()

all_path = sorted(os.listdir(args.root))
attname = []
mlpname = []
singlename = []
for name in all_path[:args.num_sample*3]:
    if "attn" in name:
        attname.append(name)
    elif "mlp" in name or "intermediate" in name:
        mlpname.append(name)
    else:
        singlename.append(name)


root = args.root
bias_word = []
pre_blank = []
blank = []
for filename in tqdm(singlename):
    # if "mlp" in filename or "attn" in filename:
    #     continue
    results = np.load(os.path.join(root, filename))
    for b,e in results['corrupt_range_anti']:
        bias_word.append(results['scores'][b:e])   # all layers
    pre_blank.append(results['scores'][results['blank_idxs_anti'][0]-1][np.newaxis, :]) 
    blank.append(results['scores'][results['blank_idxs_anti'][0]:results['blank_idxs_anti'][1]])


attn_bias_word = []
attn_pre_blank = []
attn_blank = []
for filename in tqdm(attname):
    # if "attn" in filename:
    results = np.load(os.path.join(root, filename))
    for b,e in results['corrupt_range_anti']:
        attn_bias_word.append(results['scores'][b:e])   # all layers
    attn_pre_blank.append(results['scores'][results['blank_idxs_anti'][0]-1][np.newaxis,:]) 
    attn_blank.append(results['scores'][results['blank_idxs_anti'][0]:results['blank_idxs_anti'][1]])

mlp_bias_word = []
mlp_pre_blank = []
mlp_blank = []
for filename in tqdm(mlpname):
    # if "mlp" in filename:
    results = np.load(os.path.join(root, filename))
    for b,e in results['corrupt_range_anti']:
        mlp_bias_word.append(results['scores'][b:e])   # all layers
    mlp_pre_blank.append(results['scores'][results['blank_idxs_anti'][0]-1][np.newaxis,:]) 
    mlp_blank.append(results['scores'][results['blank_idxs_anti'][0]:results['blank_idxs_anti'][1]])

bias_word = np.concatenate(bias_word, axis=0)
pre_blank = np.concatenate(pre_blank, axis=0)
blank = np.concatenate(blank, axis=0)
bias_mean = np.mean(bias_word, axis=0)
pre_blank_mean = np.mean(pre_blank, axis=0)
blank_mean = np.mean(blank, axis=0)

attn_bias_word = np.concatenate(attn_bias_word, axis=0)
attn_pre_blank = np.concatenate(attn_pre_blank, axis=0)
attn_blank = np.concatenate(attn_blank, axis=0)
attn_bias_mean = np.mean(attn_bias_word, axis=0)
attn_pre_blank_mean = np.mean(attn_pre_blank, axis=0)
attn_blank_mean = np.mean(attn_blank, axis=0)

mlp_bias_word = np.concatenate(mlp_bias_word, axis=0)
mlp_pre_blank = np.concatenate(mlp_pre_blank, axis=0)
mlp_blank = np.concatenate(mlp_blank, axis=0)
mlp_bias_mean = np.mean(mlp_bias_word, axis=0)
mlp_pre_blank_mean = np.mean(mlp_pre_blank, axis=0)
mlp_blank_mean = np.mean(mlp_blank, axis=0)


# Sample data for demonstration
layers = np.arange(args.num_layer)  # Layers from 0 to 40
effect_single_state = bias_mean
effect_attn_severed = attn_bias_mean
effect_mlp_severed = mlp_bias_mean

# Plotting the bar chart
plt.figure(figsize=(10, 5))
bar_width = 0.25  # Set the width of the bars

# Set position of bar on X axis
r1 = np.arange(len(effect_single_state))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Make the plot
plt.bar(r1, effect_single_state, color='blue', width=bar_width, edgecolor="gray",label='Effect of single state')
plt.bar(r2, effect_attn_severed, color='red', width=bar_width, edgecolor="gray",label='Effect with Attn severed')
plt.bar(r3, effect_mlp_severed, color='green', width=bar_width, edgecolor="gray",label='Effect with MLP severed')

# Add xticks on the middle of the group bars
plt.xlabel('Layer', fontweight='bold')
plt.xticks(np.arange(0, max(layers)+1, 10))
plt.ylim(2.2, 3.0)
# Create legend & Show graphic
plt.legend()
plt.ylabel('Absolute log probability difference')
plt.title(f'{args.bias.title()} bias effect of states ({args.model_name})')
plt.show()
plt.savefig(f'results/{args.model_name}-{args.bias}-state-mlp-attn.pdf', format='pdf')


# Sample data for demonstration
layers = np.arange(24)  # Layers from 0 to 40
effect_single_state = bias_mean
effect_attn_severed = pre_blank_mean
effect_mlp_severed = blank_mean

# Plotting the bar chart
plt.figure(figsize=(10, 5))
bar_width = 0.25  # Set the width of the bars

# Set position of bar on X axis
r1 = np.arange(len(effect_single_state))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Make the plot
plt.bar(r1, effect_single_state, color='blue', width=bar_width, edgecolor="gray", label='Effect of bias attribute words')
plt.bar(r2, effect_attn_severed, color='red', width=bar_width, edgecolor="gray", label='Effect of the token before attribute terms')
plt.bar(r3, effect_mlp_severed, color='green', width=bar_width, edgecolor="gray",label='Effect of attribute terms')

# Add xticks on the middle of the group bars
plt.xlabel('Layer', fontweight='bold')
plt.xticks(np.arange(0, max(layers)+1, 10))
plt.ylim(2.1, 3.1)
# Create legend & Show graphic
plt.legend()
plt.ylabel('Absolute log probability difference')
plt.title(f'{args.bias.title()} bias effect of different words ({args.model_name})')
plt.show()
plt.savefig(f'results/{args.model_name}-{args.bias}-words.pdf', format="pdf")
