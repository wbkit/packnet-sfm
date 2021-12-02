import matplotlib.pyplot as plt
import numpy as np
import torch

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

    mask_arr = (depth_map >= 0)

    #### Plot 3D data
    # fig, ax = plt.subplots()
    # ax.scatter(x_map, y_map, depth_map, c=depth_map)
    # plt.show()
    # plt.savefig('depth_plot3D.png')

    # distance_norm = np.sqrt(x_map**2 + y_map**2 + depth_map**2)
    theta = np.arcsin(y_map / depth_map ) * mask_arr

    # Get bounds from data
    theta_min = np.min(theta)
    theta_max = np.max(theta)

    # Transform and normalise data
    # Value range will be from -0.5 to 63.49
    theta_norm = theta - theta_min
    theta_max_n = np.max(theta_norm)
    theta_norm = theta_norm * (63.99 / theta_max_n) - 0.5

    # Classify by rounding
    lidar_channel_arr = np.around(theta_norm) 

    # assert(np.max(lidar_channel_arr) == 63)
    # assert(np.min(lidar_channel_arr) == 0)
    fig, ax = plt.subplots()
    ax.imshow(np.mod(lidar_channel_arr, 5))
    #plt.colorbar()
    plt.show()
    plt.savefig('depth_plot_ch.png', dpi=800)

def classify_lidar_channels(velodyne):
    # Get horizontal angle of each datapoint in the LIDAR coord. system
    x = velodyne[:,0]
    y = velodyne[:,1]
    z = velodyne[:,2]

    norm = np.sqrt(x**2 + y**2 + z**2)
    lidar_theta = np.arccos(z / norm)
    

    # Plot the vertical angle distribution
    lidar_theta = (-lidar_theta + np.pi/2)
    plt.figure(figsize=(12, 4))
    plt.hist((360 * lidar_theta /(2*np.pi)), bins=256)
    plt.xlabel("theta in °")

    # Get bounds from data
    theta_min = np.min(lidar_theta)
    theta_max = np.max(lidar_theta)

    # Transform and normalise data
    # Value range will be from -0.5 to 63.49
    lidar_theta_norm = lidar_theta - theta_min
    theta_max_n = np.max(lidar_theta_norm)
    lidar_theta_norm = lidar_theta_norm * (63.99 / theta_max_n) - 0.5

    # Classify by rounding
    lidar_channel_arr = np.around(lidar_theta_norm)

    assert(np.max(lidar_channel_arr) == 63)
    assert(np.min(lidar_channel_arr) == 0)

    return lidar_channel_arr