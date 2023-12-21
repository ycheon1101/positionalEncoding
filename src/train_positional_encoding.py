from pathlib import Path
import sys
# path
modules_path = Path(__file__).parent
src_path = modules_path.parent

# append path
sys.path.append(src_path)
# sys.path.append(str(src_path) + '/modules')
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import numpy as np
from positional_encoding import GaussianFourier
import table_images
from set_device import device
from mlp import MLP
from PIL import Image
import numpy as np

# params
learning_rate = 5e-3
num_epochs = 300
hidden_feature = 128
hidden_layers = 8
max_pixel = 1.0

# calc loss
criterion = nn.MSELoss() 

# Model MLP instanciation
model = MLP(in_feature=128, hidden_feature=hidden_feature, hidden_layers=hidden_layers, out_feature=3).to(device)

# get img_data
img_df, crop_size = table_images.make_table()

# optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

# train model
def train_model(target, model_arg):
    for epoch in range(num_epochs):

        # model
        generated = model(model_arg)

        # loss = criterion(generated, target)
        loss = criterion(generated, target)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# plot image
def plot_img(model_arg):
    # reshape with final generated
    generated_reshape = model(model_arg)
    generated_reshape *= 255.
    generated_reshape = torch.reshape(generated_reshape, (crop_size, crop_size, 3))

    generated_reshape = generated_reshape.cpu().detach().numpy()

    # save image
    save_img = Image.fromarray(generated_reshape.astype(np.uint8))
    save_img_path = Path(src_path) / 'images' / 'tested_images'
    save_img_filename = save_img_path / 'positional_mlp_test_img1.jpg'
    save_img.save(save_img_filename)

    # generated_reshape = generated_reshape.detach().numpy()


    # show image
    # plt.imshow(generated_reshape)
    # plt.show()

# main
def main():
    for image in range(1, len(img_df.index) + 1):
        target = img_df['img_flatten'][image].to(device)
        xy_flatten = img_df['xy_flatten'][image].to(device)
        fourier_result = GaussianFourier(num_input_channels=2, mapping_size = 64, scale=4)(xy_flatten).to(device)
        
        train_model(target, fourier_result)
        plot_img(fourier_result)

main()












