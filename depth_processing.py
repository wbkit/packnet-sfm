import matplotlib.pyplot as plt
import numpy as np
import torch


class DepthFilter:

    def __init__(self, batch, filter_settings=None):
        depth_map = batch['input_depth'][0][0].numpy()
        self.x_map = np.zeros(depth_map.shape)
        self.y_map = np.zeros(depth_map.shape)

        x0_intr = batch['intrinsics_full'].numpy()[0][0,2]
        y0_intr = batch['intrinsics_full'].numpy()[0][1,2]
        k_f     = batch['intrinsics_full'].numpy()[0][0,0]


        for i in range(0, depth_map.shape[1]):
            self.x_map[:, i] = i  - x0_intr
        for i in range(0, depth_map.shape[0]):
            self.y_map[i, :] = -i  + y0_intr

        self.x_map = self.x_map / k_f
        self.y_map = self.y_map / k_f

        self.filter_settings = filter_settings
        if filter_settings is not None:
            self.filter_length = len(filter_settings)
            self.batch_size = batch['input_depth'].numpy().shape[0]

    def filter_ch_from_to(self, batch, from_ch=0, to_ch=63):
        depth_map = batch['input_depth'].numpy()

        z_map = depth_map / np.sqrt(1. + self.x_map**2 + self.y_map**2)
        #x_map = self.x_map * z_map
        y_map = self.y_map * z_map

        # prel_mask = (depth_map > 0)
        theta = np.arcsin(y_map / (depth_map + 1e-5) ) # * prel_mask

        # Get bounds from data
        theta_min = -0.30 #np.min(theta)
        theta_max = 0.04 #np.max(theta)
        # Transform and normalise data
        # Value range will be from -0.5 to 63.49
        theta_norm = theta - theta_min
        theta_max_n = np.max(theta_norm)
        theta_norm = theta_norm * (63.99 / theta_max_n) - 0.5
        # Classify by rounding
        lidar_channel_arr = np.around(theta_norm) 

        # Filter
        mask_arr = ((depth_map <= 0) | (lidar_channel_arr < from_ch) | (lidar_channel_arr > to_ch))
        depth_map[mask_arr] = 0

        #lidar_channel_arr[mask_arr] = -1
        #fig, ax = plt.subplots()
        #ax.imshow(np.mod(lidar_channel_arr[0][0], 5))
        #plt.show()
        #plt.savefig('depth_plot_chfilt2.png', dpi=800)

        return depth_map

    def filter_ch_modulo(self, batch, modulo=2):
        depth_map = batch['input_depth'].numpy()

        z_map = depth_map / np.sqrt(1. + self.x_map**2 + self.y_map**2)
        #x_map = self.x_map * z_map
        y_map = self.y_map * z_map

        # prel_mask = (depth_map > 0)
        theta = np.arcsin(y_map / (depth_map + 1e-5) ) # * prel_mask

        # Get bounds from data
        theta_min = -0.30 #np.min(theta)
        theta_max = 0.04 #np.max(theta)
        # Transform and normalise data
        # Value range will be from -0.5 to 63.49
        theta_norm = theta - theta_min
        theta_max_n = np.max(theta_norm)
        theta_norm = theta_norm * (63.99 / theta_max_n) - 0.5
        # Classify by rounding
        lidar_channel_arr = np.around(theta_norm) 

        # Filter
        mask_arr = ((depth_map <= 0) | (np.mod(lidar_channel_arr, modulo) != 0))
        depth_map[mask_arr] = 0

        #lidar_channel_arr[mask_arr] = -1
        #fig, ax = plt.subplots()
        #ax.imshow(np.mod(lidar_channel_arr[0][0], 5))
        #plt.show()
        #plt.savefig('depth_plot_chfilt2.png', dpi=800)

        return depth_map

    def filter_batch_modulo(self, batch):
        # Random sampling to determine filter that is to be applied
        index = np.random.randint(0, self.filter_length)
        task_vect = np.zeros((1,self.filter_length), dtype=np.float32)
        task_vect[:,index] = 1

        filter_mod_value = self.filter_settings[index]

        # Apply filter
        output_batch = self.filter_ch_modulo(batch, modulo=filter_mod_value)

        return output_batch, task_vect



def plot_depth_map(batch, index=0):
    #Get depthmap as numpy array from batch
    depth_map = batch['depth'][index][0].numpy()

    # Plot and save
    fig, ax = plt.subplots()
    ax.imshow(depth_map)
    #plt.colorbar()
    plt.show()
    plt.savefig('depth_plot.png')

def filter_depth_channels(batch, index=0):
    depth_map = batch['depth'][index][0].numpy()
    x_map = np.zeros(depth_map.shape)
    y_map = np.zeros(depth_map.shape)

    x0_intr = batch['intrinsics_full'].numpy()[0][0,2]
    y0_intr = batch['intrinsics_full'].numpy()[0][1,2]
    k_f     = batch['intrinsics_full'].numpy()[0][0,0]


    for i in range(0, depth_map.shape[1]):
        x_map[:, i] = i  - x0_intr
    for i in range(0, depth_map.shape[0]):
        y_map[i, :] = -i  + y0_intr

    x_map = x_map / k_f
    y_map = y_map / k_f
    z_map = depth_map / np.sqrt(1. + x_map**2 + y_map**2)
    x_map = x_map * z_map
    y_map = y_map * z_map

    mask_arr = (depth_map < 0)

    # distance_norm = np.sqrt(x_map**2 + y_map**2 + depth_map**2)
    theta = np.arcsin(y_map / depth_map ) 

    # Get bounds from data
    theta_min =  -0.4#np.min(theta)
    theta_max = 0.03 #np.max(theta)

    # Transform and normalise data
    # Value range will be from -0.5 to 63.49
    theta_norm = theta - theta_min
    theta_max_n = np.max(theta_norm)
    theta_norm = theta_norm * (63.99 / theta_max_n) - 0.5

    # Classify by rounding
    lidar_channel_arr = np.around(theta_norm) 
    lidar_channel_arr[mask_arr] = -1

    # assert(np.max(lidar_channel_arr) == 63)
    # assert(np.min(lidar_channel_arr) == 0)
    fig, ax = plt.subplots()
    ax.imshow(np.mod(lidar_channel_arr, 5))
    plt.show()
    plt.savefig('depth_plot_ch.png', dpi=800)
