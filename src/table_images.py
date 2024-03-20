import sys
from pathlib import Path
# path
modules_path = Path(__file__).parent
src_path = modules_path.parent

sys.path.append(src_path)
import img_dataset
import os
import matplotlib.pyplot as plt
import pandas as pd

# initialize list
height_list = []
width_list = []
table_list = []

dir_path = str(src_path) + '/images/cat'
file_list = os.listdir(dir_path)

def make_table():

    # get all images
    for img_file in file_list:
        _, crop_size, img_flatten, xy_flatten, _ = img_dataset.img_data(dir_path + '/', img_file)

        img_data = {
            'img_file' : img_file,
            'img_flatten': img_flatten,
            'xy_flatten' : xy_flatten,
        }

        table_list.append(img_data)

    # create table
    img_df = pd.DataFrame(table_list)
    img_df.index += 1 

    return img_df, crop_size
# img_df, _ = make_table()
# print(img_df['img_flatten'])




    


