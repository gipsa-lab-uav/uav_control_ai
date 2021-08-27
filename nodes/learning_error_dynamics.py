#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import Tkinter as tk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# import warnings
# warnings.filterwarnings("ignore")
import numpy as np
import tensorflow as tf
import math
import time
import rosbag
import os
import glob

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras import callbacks
import tensorflow.keras.optimizers as kopt

from os.path import dirname
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import display_results as dr
import rotations_tools as rt
import rdn_ros_funct as rrf
from PIL import ImageTk, Image
from shutil import copyfile
import rospy

class Application(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.creer_widgets()

    def creer_widgets(self):
        # Figure for time response
        figure = plt.figure()
        self.a1 = figure.add_subplot(311)
        self.a1.set_xlabel('Sample')
        self.a1.set_ylabel('X (m)')
        self.a2 = figure.add_subplot(312)
        self.a2.set_xlabel('Sample')
        self.a2.set_ylabel('Y (m)')
        self.a3 = figure.add_subplot(313)
        self.a3.set_xlabel('Sample')
        self.a3.set_ylabel('Z (m)')
        self.fig_canvas = FigureCanvasTkAgg(figure, self)
        self.fig_canvas.get_tk_widget().grid(column=0, row=7, columnspan=4, rowspan=6)

        # Titles
        tk.Label(self, text='TRAINING DATASET', font='Helvetica 14 bold').grid(column=1, row=0, sticky='NEWS', pady=10)
        tk.Label(self, text='NEURAL NETWORK TRAINING', font='Helvetica 14 bold').grid(column=6, row=0, sticky='NEWS', pady=10, columnspan=2)

        # Image
        img = ImageTk.PhotoImage(Image.open(dirname(os.path.abspath(__file__)) +"/net.jpg").resize((300, 250), Image.ANTIALIAS))
        self.image_net = tk.Label(self, image = img)
        self.image_net.image = img
        self.image_net.grid(column=6, row=6, rowspan=3, columnspan=2, padx=40)

        ## Second Grid Row
        # RosBag Folder Label
        tk.Label(self, text="Enter rosbag folder path :").grid(column=0, row=1, pady=20)

        # RosBag Folder Path Entry
        self.path_net = tk.Entry(self, width = 50)
        self.path_net.grid(column=1, row=1)
        path = dirname(dirname(os.path.abspath(__file__)))+'/rosbag/selected'
        self.path_net.insert(0, path)

        # Read Button
        self.button_read = tk.Button(self, text="Read", command=self.rosbag_read)
        self.button_read.grid(column=3, row=1, padx=40)

        # Exit Button
        self.exit_button = tk.Button(self, text="Exit", command=self.quit)
        self.exit_button.grid(column=11, row=3)

        # Train Button
        self.button_train = tk.Button(self, text="Train", command=self.learning)
        self.button_train.grid(column=6, row=9, columnspan=1, sticky='E')
        self.button_train["state"] = "disabled"

        ## Third Grid Row
        # Lower Bound
        tk.Label(self, text="Lower bound :").grid(column=0, row=2)
        self.bound_min = tk.Entry(self, width = 10)
        self.bound_min.insert(0, 0)
        self.bound_min.grid(column=1, row=2, sticky='W')

        # Crop Button
        self.button_data_range = tk.Button(self, text="Crop", command=self.switch_b1)
        self.button_data_range.grid(column=1, row=2)
        self.button_data_range["state"] = "disabled"

        ## Fourth Grid Row
        tk.Label(self, text="Upper bound :").grid(column=0, row=3)
        self.bound_max = tk.Entry(self, width = 10)
        self.bound_max.insert(0, 0)
        self.bound_max.grid(column=1, row=3, sticky='W')
        self.button_data_range_val = tk.Button(self, text="Validate Range", command=self.switch_b2)
        self.button_data_range_val.grid(column=1, row=3, pady=20)
        self.button_data_range_val["state"] = "disabled"

        # Variables
        self.bag_dic = None
        self.var_crop = tk.IntVar()
        self.var_validate = tk.IntVar()

        ## Neural Network Options
        tk.Label(self, text='Learning Rate').grid(column=6, row=1, sticky='W')
        self.lr = tk.Entry(self, width = 7)
        self.lr.insert(0, 0.01)
        self.lr.grid(column=7, row=1, sticky='W')

        tk.Label(self, text='Epoch').grid(column=6, row=2, sticky='W')
        self.epochs = tk.Entry(self, width = 7)
        self.epochs.insert(0, 500)
        self.epochs.grid(column=7, row=2, sticky='W')

        tk.Label(self, text='Batch Size').grid(column=6, row=3, sticky='W')
        self.batch_size = tk.Entry(self, width = 7)
        self.batch_size.insert(0, 256)
        self.batch_size.grid(column=7, row=3, sticky='W')

        self.label_epoch = tk.Label(self, text="Select Model around Epoch : ")
        self.num_epoch  = tk.Entry(self, width = 7)
        self.num_epoch.insert(0, 50)

        # Save Model
        self.save_button = tk.Button(self, text="Save Model", command=self.save)
        self.label_epoch.grid(column=10, row=2)
        self.num_epoch.grid(column=11, row=2)
        self.save_button.grid(column=12, row=2)
        tk.Label(self, text='MODEL SELECTION', font='Helvetica 14 bold').grid(column=11, row=0, sticky='NEWS', pady=10)

    def switch_b1(self):
        # Button Command For Data Crop
        self.button_data_range["state"] = "disabled"
        self.button_data_range_val["state"] = "active"
        self.var_crop.set(1)

    def switch_b2(self):
        # Button Command For Data Validation
        self.button_data_range_val["state"] = "disabled"
        self.button_data_range["state"] = "active"
        self.var_validate.set(1)


    def save(self):
        # Saving DNN Model after training
        tab = os.listdir(dirname(os.path.abspath(__file__)) + '/temp_dnn/')
        tab = [ int(l.replace('.h5', '')) for l in tab]
        value = int(self.num_epoch.get())
        idx = np.searchsorted(tab, value, side="left")

        # Look for nearest value of "value" in "tab"
        if idx > 0 and (idx == len(tab) or math.fabs(value - tab[idx-1]) < math.fabs(value - tab[idx])):
            value =  tab[idx-1]
        else:
            value =  tab[idx]

        # Copy DNN in the good directory
        copyfile(dirname(os.path.abspath(__file__)) + '/temp_dnn/'+"{:05d}".format(value)+'.h5', dirname(os.path.abspath(__file__)) + '/dnn_model/model_lrn_'+"{:05d}".format(value)+'.h5')

        # Show Save Message Info
        msg_str = "Model Selected at epoch: "+"{:05d}".format(value)+"\n\n Saved under:  ./dnn_model/model_lrn_"+"{:05d}".format(value)+".h5"
        self.option_add('*Dialog.msg.font', 'Helvetica 12')
        tk.messagebox.showinfo(title="Model Saved", message=msg_str)

    def rosbag_read(self):

        # Set Button State
        self.button_read["state"] = "disabled"
        self.button_data_range["state"] = "active"
        self.button_train["state"] = "disabled"

        # Bag Dictionary
        bag_dic = {}
        i = 0

        # Get Folder Path of Selected Bags
        bag_path = self.path_net.get()

        files = os.listdir(bag_path)

        for name in files:
            if os.path.isfile(bag_path+'/'+name):
                bag_dic["bag{0}".format(i)] = rosbag.Bag(bag_path+'/'+name)
                positions = rrf.get_positions_data(bag_dic.get('bag'+format(i)))

                self.bound_max.delete(0, 'end')
                self.bound_max.insert(0, len(positions[0]))
                self.bound_min.delete(0, 'end')
                self.bound_min.insert(0, 0)

                # Plot Positions Graphs
                self.a1.plot(positions[0])
                self.a1.plot(positions[3])
                self.a2.plot(positions[1])
                self.a2.plot(positions[4])
                self.a3.plot(positions[2])
                self.a3.plot(positions[5])

                # Set Legends
                self.a1.set_xlabel('Sample')
                self.a1.set_ylabel('X (m)')
                self.a2.set_xlabel('Sample')
                self.a2.set_ylabel('Y (m)')
                self.a3.set_xlabel('Sample')
                self.a3.set_ylabel('Z (m)')
                self.a1.set_title('Position Data From Bag'+str(i))
                self.fig_canvas.draw()

                # Wait for Crop Validation
                self.button_data_range.wait_variable(self.var_crop)

                # Get Crop Bounds
                bmin = int(self.bound_min.get())
                bmax = int(self.bound_max.get())
                bag_dic["bag_range{0}".format(i)] = [bmin, bmax]

                # Plot New Positions Graphs
                self.a1.clear()
                self.a2.clear()
                self.a3.clear()
                self.a1.plot(positions[0, bmin:bmax])
                self.a1.plot(positions[3, bmin:bmax])
                self.a2.plot(positions[1, bmin:bmax])
                self.a2.plot(positions[4, bmin:bmax])
                self.a3.plot(positions[2, bmin:bmax])
                self.a3.plot(positions[5, bmin:bmax])

                # Set Legends
                self.a1.set_xlabel('Sample')
                self.a1.set_ylabel('X (m)')
                self.a2.set_xlabel('Sample')
                self.a2.set_ylabel('Y (m)')
                self.a3.set_xlabel('Sample')
                self.a3.set_ylabel('Z (m)')
                self.a1.set_title('Position Data From Bag'+str(i)+' Cropped')
                self.fig_canvas.draw()

                # Wait for Data Validation
                self.button_data_range_val.wait_variable(self.var_validate)

                # Restart var Values
                self.var_crop.set(0)
                self.var_validate.set(0)

                # Clear for Next Plot
                self.a1.clear()
                self.a2.clear()
                self.a3.clear()
                i+=1

        # Store Bag Dictionary
        self.bag_dic = bag_dic

        # Set Button State
        self.button_data_range_val["state"] = "disabled"
        self.button_data_range["state"] = "disabled"
        self.button_train["state"] = "active"
        self.button_read["state"] = "active"

    def learning(self):

        # Get x_train & y_train from Bags
        x_train, y_train, infos = rrf.get_data(self.bag_dic.get('bag0'), data_range=self.bag_dic.get('bag_range0'), filter_imu=2, battery_v=1)

        for i in range(1, int(len(self.bag_dic)/2)):
            # print(self.bag_dic.get('bag_range'+format(i)))
            x_train_add, y_train_add, infos_add = rrf.get_data(self.bag_dic.get('bag'+format(i)), data_range=self.bag_dic.get('bag_range'+format(i)), filter_imu=2, battery_v=1)
            x_train = np.concatenate((x_train, x_train_add))
            y_train = np.concatenate((y_train, y_train_add))
            infos = np.concatenate((infos, infos_add))

        # DNN Initialization
        model_dnn = Sequential()
        model_dnn.add(Dense(64, activation='relu', input_dim=12))
        model_dnn.add(Dense(64, activation='relu'))
        model_dnn.add(Dense(3))

        # Optimizer
        opt = kopt.Nadam(learning_rate=float(self.lr.get()), beta_1=0.9, beta_2=0.999)

        # Path to save DNN models
        filepath = dirname(os.path.abspath(__file__)) + '/temp_dnn/{epoch:05d}.h5'

        # Remove Previous Models
        for f in os.listdir(dirname(os.path.abspath(__file__)) + '/temp_dnn/'):
            os.remove(dirname(os.path.abspath(__file__)) + '/temp_dnn/'+f)

        # Callbacks
        # 1 - Save Models
        # 2 - Learning rate reduction

        callbacks_list = [
            # Saving Model
            callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True,
                                          save_weights_only=False, mode='auto', save_freq='epoch'),
            # Learning rate reduction
            callbacks.ReduceLROnPlateau(monitor='loss', factor=9/10, patience=20, verbose=1,
                                        mode='auto', min_delta=0.0001, cooldown=0, min_lr=1e-10),

        ]

        # Compile
        model_dnn.compile(loss='mean_squared_error', optimizer=opt)

        # Get Validation data
        [x_train, x_val, y_train, y_val] = train_test_split(x_train, y_train, test_size=1/3, shuffle=False)

        history = model_dnn.fit(x_train, y_train, validation_data=(x_val, y_val),
                                  epochs=int(self.epochs.get()), batch_size=int(self.batch_size.get()), shuffle = True,
                                  callbacks=callbacks_list)

        # Save Info
        self.save_info(infos)

        # Figure For Loss
        figure2 = plt.figure()
        self.loss_fig = figure2.add_subplot(111)
        self.loss_fig.plot(history.history['loss'], label='loss')
        self.loss_fig.plot(history.history['val_loss'], label='validation loss')
        self.loss_fig.set_xlabel('Epoch')
        self.loss_fig.set_ylabel('Mean Squared Error (MSE)')
        self.fig_canvas2 = FigureCanvasTkAgg(figure2, self)
        self.fig_canvas2.get_tk_widget().grid(column=10, row=7, columnspan=4, rowspan=6)

    def save_info(self, infos):
        # Create a .txt file with general info on learning process
        total = np.sum(infos, axis=0)
        f = open(dirname(os.path.abspath(__file__)) + "/learning_info.txt","w+")
        f.write("### General learning info ###")
        f.write("\n\n--- Neural Network ---")
        f.write("\nNumber of layers: 2")
        f.write("\nNumber of units per layer: 64")
        f.write("\nActivation Function: ReLU")
        f.write("\n\n--- Learning ---")
        f.write("\nNumber of flight scenario: "+ str(infos.shape[0]))
        f.write("\nTotal time flight (s): "+ str(round(total[0], 2)))
        f.write("\nNumber of data: "+ str(total[1]))
        f.write("\nNumber of epochs: " + self.epochs.get())
        f.write("\nNumber of batch size: "+ self.batch_size.get())
        f.write("\nOptimizer: Nadam")
        f.write("\nLearning Rate: "+ self.lr.get())
        f.close()

if __name__ == "__main__":
    app = Application()
    app.title("Learning Errors Dynamics From Ros Bags")
    app.mainloop()
