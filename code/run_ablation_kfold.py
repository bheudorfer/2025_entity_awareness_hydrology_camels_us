# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 13:20:59 2024

@author: gw3013
"""

#%%

# pth0 = "D:/07b_GRL_spinoff/"
pth0 = "/pfs/work7/workspace/scratch/gw3013-GRL/"


#Import necessary packages
import os
import sys
import pandas as pd
import numpy as np
import time
import pickle
import random
import torch
from torch.utils.data import DataLoader

# Import classes and functions from other files
sys.path.append(pth0+"code/aux_functions")
from functions_training import nse_basin_averaged
from functions_evaluation import nse
from functions_aux import create_folder, set_random_seed, write_report

# Import model
from cudalstm import CudaLSTM


#%% Part 1.1 Initialize information


# ----- k-fold experiment  ----------


# Import dataset to use
from camelsus import CAMELS_US

# do iteration
seeds = [227, 310, 325, 514, 550]
folds = np.unique([x[:5] for x in os.listdir(pth0+"data/folds")]).tolist()

for seedi in seeds: 
    for fold in folds:
        
        # Define experiment name
        experiment_name = "ablation_"+fold+"_"+str(seedi)
        
        # paths to access the information
        path_entities = pth0+"data/folds/"+fold+".txt"
        path_entities_test = pth0+"data/folds/"+fold+"_test.txt"
        path_data = pth0+"data/CAMELS_US"
        
        # dynamic forcings and target
        dynamic_input = ['prcp_daymet', 'srad_daymet', 'tmax_daymet', 'tmin_daymet', 'vp_daymet',
                         'prcp_maurer', 'srad_maurer', 'tmax_maurer', 'tmin_maurer', 'vp_maurer',
                         'prcp_nldas', 'srad_nldas', 'tmax_nldas', 'tmin_nldas', 'vp_nldas']
        forcings = ["all"]
        target = ["QObs(mm/d)"]
        
        # static attributes that will be used
        static_input = []
        
        
        #%% Part 1.2 Initialize information
        
        # time periods
        training_period = ["1999-10-01","2008-09-30"]
        validation_period = ["1980-10-01","1989-09-30"]
        testing_period = ["1989-10-01","1999-10-30"]
        
        model_hyper_parameters = {
            "input_size_lstm": len(dynamic_input) + len(static_input),
            "no_of_layers":1,  
            "seq_length": 365,
            "hidden_size": 256,
            "batch_size_training":256,
            "batch_size_evaluation":1024,
            "no_of_epochs": 20,
            "drop_out_rate": 0.4, 
            "learning_rate": 0.001,
            "adapt_learning_rate_epoch": 5,
            "adapt_gamma_learning_rate": 0.8,
            "set_forget_gate":3,
            "validate_every": 1,
            "validate_n_random_basins": -1
        }
        
        # device to train the model
        running_device = "gpu" #cpu or gpu
        
        # define random seed
        seed = seedi
        
        # colorblind friendly palette
        color_palette = {"observed": "#1f78b4","simulated": "#ff7f00"}
        
        # Create folder to store the results
        path_save_folder = pth0+"results/"+experiment_name
        create_folder(folder_path=path_save_folder)
        
        # check if model will be run in gpu or cpu and define device
        if running_device == "gpu":
            print(torch.cuda.get_device_name(0))
            device= f'cuda:0'
        elif running_device == "cpu":
            device = "cpu"
        
        
        
        #%% Part 2. Class to create the dataset object used to manage the information
        
        
        # Dataset training
        training_dataset = CAMELS_US(dynamic_input= dynamic_input,
                                     forcing= forcings,
                                     target= target, 
                                     sequence_length= model_hyper_parameters["seq_length"],
                                     time_period= training_period,
                                     path_data= path_data,
                                     path_entities= path_entities,
                                     static_input= static_input,
                                     check_NaN= True)
        
        training_dataset.calculate_basin_std()
        training_dataset.calculate_global_statistics(path_save_scaler=path_save_folder)
        training_dataset.standardize_data()
        
        # Dataloader training
        train_loader = DataLoader(dataset = training_dataset, 
                                  batch_size = model_hyper_parameters["batch_size_training"],
                                  shuffle = True,
                                  drop_last = True)
        
        print("Batches in training: ", len(train_loader))
        sample = next(iter(train_loader))
        print(f'x_lstm: {sample["x_lstm"].shape} | y_obs: {sample["y_obs"].shape} | basin_std: {sample["basin_std"].shape}')
        
        # xx = sample["x_lstm"][0]
        # xx = pd.DataFrame(xx.numpy())
        # xx
        
        # yy = sample["y_obs"][0]
        # yy = pd.DataFrame(yy.numpy())
        # ID = 25
        
        
        #%% Part 3. Create dataset for validation
        
        
        # We will create an individual dataset per basin. This will give us more flexibility
        entities_ids = np.loadtxt(path_entities, dtype="str").tolist()
        validation_dataset = {}
        
        for entity in entities_ids:
            dataset = CAMELS_US(dynamic_input= dynamic_input,
                                forcing= forcings,
                                target= target, 
                                sequence_length= model_hyper_parameters["seq_length"],
                                time_period= validation_period,
                                path_data= path_data,
                                entity= entity,
                                static_input= static_input,
                                check_NaN= False)
            
            dataset.scaler = training_dataset.scaler
            dataset.standardize_data(standardize_output=False)
            validation_dataset[entity]= dataset
        
        
        #%% Part 4. Train LSTM
        
        
        # construct model
        set_random_seed(seed=seed)
        lstm_model = CudaLSTM(hyperparameters=model_hyper_parameters).to(device)
        
        # optimizer
        optimizer = torch.optim.Adam(lstm_model.parameters(),
                                     lr=model_hyper_parameters["learning_rate"])
                                      # eps=0.0001)
            
        # define learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=model_hyper_parameters["adapt_learning_rate_epoch"],
                                                    gamma=model_hyper_parameters["adapt_gamma_learning_rate"])
        
        # set forget gate to 3 to ensure that the model is capable to learn long term dependencies
        lstm_model.lstm.bias_hh_l0.data[model_hyper_parameters["hidden_size"]:2 * model_hyper_parameters["hidden_size"]]=\
            model_hyper_parameters["set_forget_gate"]
        
        training_time = time.time()
        # Loop through the different epochs
        for epoch in range(1, model_hyper_parameters["no_of_epochs"]+1):
            
            epoch_start_time = time.time()
            total_loss = []
            # Training -------------------------------------------------------------------------------------------------------
            lstm_model.train()
            
            for sample in train_loader:
                optimizer.zero_grad() # sets gradients of weigths and bias to zero
                pred  = lstm_model(sample["x_lstm"].to(device)) # forward call
                
                loss = nse_basin_averaged(y_sim=pred["y_hat"], 
                                          y_obs=sample["y_obs"].to(device), 
                                          per_basin_target_std=sample["basin_std"].to(device))
                
                loss.backward() # backpropagates
                optimizer.step() #update weights
                total_loss.append(loss.item())
                
                # remove from cuda
                del sample["x_lstm"], sample["y_obs"], sample["basin_std"], pred
                torch.cuda.empty_cache()
                
            #training report  
            report = f'Epoch: {epoch:<2} | Loss training: {"%.3f "% (np.mean(total_loss))}'
            
            # Validation -----------------------------------------------------------------------------------------------------
            if epoch % model_hyper_parameters["validate_every"] == 0:
                lstm_model.eval()
                validation_results = {}
                with torch.no_grad():
                    # If we define validate_n_random_basins as 0 or negative, we take all the basins
                    if model_hyper_parameters["validate_n_random_basins"] <= 0:
                        validation_basin_ids = validation_dataset.keys()
                    else:
                        keys = list(validation_dataset.keys())
                        validation_basin_ids = random.sample(keys, model_hyper_parameters["validate_n_random_basins"])
                    
                    # go through each basin that will be used for validation
                    for basin in validation_basin_ids:
                        loader = DataLoader(dataset=validation_dataset[basin], 
                                            batch_size=model_hyper_parameters["batch_size_evaluation"], 
                                            shuffle=False, 
                                            drop_last = False)
                        
                        df_ts = pd.DataFrame()
                        for sample in loader:
                            pred  = lstm_model(sample["x_lstm"].to(device)) 
                            # backtransformed information
                            y_sim = pred["y_hat"]* validation_dataset[basin].scaler["y_std"].to(device) +\
                                validation_dataset[basin].scaler["y_mean"].to(device)
        
                            # join results in a dataframe and store them in a dictionary (is easier to plot later)
                            df = pd.DataFrame({"y_obs": sample["y_obs"].flatten().cpu().detach(), 
                                               "y_sim": y_sim.flatten().cpu().detach()}, 
                                              index=pd.to_datetime(sample["date"]))
        
                            df_ts = pd.concat([df_ts, df], axis=0)
        
                            # remove from cuda
                            del pred, y_sim
                            torch.cuda.empty_cache()       
                        
                        validation_results[basin] = df_ts
                         
                    #average loss validation
                    loss_validation = nse(df_results=validation_results)
                    report += f'| Loss validation: {"%.3f "% (loss_validation)}'
        
            
            # save model after every epoch
            path_saved_model = path_save_folder+"/epoch_" + str(epoch)
            torch.save(lstm_model.state_dict(), path_saved_model)
                    
            # print epoch report
            report += f'| Epoch time: {"%.1f "% (time.time()-epoch_start_time)} s | LR:{"%.5f "% (optimizer.param_groups[0]["lr"])}'
            print(report)
            write_report(file_path=path_save_folder+"/run_progress.txt", text=report)
            # modify learning rate
            scheduler.step()
        
        # print final report
        report = f'Total training time: {"%.1f "% (time.time()-training_time)} s'
        print(report)
        write_report(file_path=path_save_folder+"/run_progress.txt", text=report)   
        
        
        #%% Part 5. Test LSTM
        
        
        # In case I already trained an LSTM I can re-construct the model
        #lstm_model = CudaLSTM(hyperparameters=model_hyper_parameters).to(device)
        #lstm_model.load_state_dict(torch.load(path_save_folder + "/epoch_20", map_location=device))
        
        # We will create an individual dataset per basin. This will give us more flexibility
        entities_ids = np.loadtxt(path_entities_test, dtype="str").tolist()
        testing_dataset = {}
        
        # We can read a previously generated scaler or use the one from before
        scaler = training_dataset.scaler
        #with open(path_save_folder + "/scaler.pickle", "rb") as file:
        #    scaler = pickle.load(file)
        
        for entity in entities_ids:
            dataset = CAMELS_US(dynamic_input= dynamic_input,
                                forcing= forcings,
                                target= target, 
                                sequence_length= model_hyper_parameters["seq_length"],
                                time_period= testing_period,
                                path_data= path_data,
                                entity= entity,
                                static_input= static_input,
                                check_NaN= False)
            
            dataset.scaler = scaler
            dataset.standardize_data(standardize_output=False)
            testing_dataset[entity]= dataset
        
        
        lstm_model.eval()
        test_results = {}
        with torch.no_grad():
            for basin, dataset in testing_dataset.items():
                loader = DataLoader(dataset = dataset, 
                                    batch_size = model_hyper_parameters["batch_size_evaluation"], 
                                    shuffle = False, 
                                    drop_last = False) 
                df_ts = pd.DataFrame()
                for sample in loader:
                    pred  = lstm_model(sample["x_lstm"].to(device)) 
                    # backtransformed information
                    y_sim = pred["y_hat"]* dataset.scaler["y_std"].to(device) + dataset.scaler["y_mean"].to(device)
        
                    # join results in a dataframe and store them in a dictionary (is easier to plot later)
                    df = pd.DataFrame({"y_obs": sample["y_obs"].flatten().cpu().detach(), 
                                        "y_sim": y_sim.flatten().cpu().detach()}, 
                                        index=pd.to_datetime(sample["date"]))
        
                    df_ts = pd.concat([df_ts, df], axis=0)
        
                    # remove from cuda
                    del pred, y_sim
                    torch.cuda.empty_cache()       
                
                test_results[basin] = df_ts
        
        # Save results as a pickle file
        with open(path_save_folder+"/test_results.pickle", "wb") as f:
            pickle.dump(test_results, f)
        
        
        #%% Part 6. Initial analysis
        
        
        # Loss testing
        loss_testing = nse(df_results=test_results, average=False)
        df_NSE = pd.DataFrame(data={"basin_id": testing_dataset.keys(), "NSE": np.round(loss_testing,3)})
        df_NSE = df_NSE.set_index("basin_id")
        df_NSE.to_csv(path_save_folder+"/NSE.csv", index=True, header=True)
        
        