import sys
import os
import numpy as np
import h5py
import pandas as pd
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, project_root)

# from constants import PROJ_ADDR
PROJ_ADDR = os.path.dirname(script_dir)

from SACArgs import SACArgs
from sac import SAC
from smartfunc import sel_ue
from custom_mimo_env import MimoEnv

# Action Steering Imports
from Action_Steering.action_steering_utils import process_buffer, transform_action
from Action_Steering.symbolic_representation import QuantileManager, Symbolizer
from Action_Steering.experiment_constants import KPI_LIST, USERS


def generate_plot_data(dataset_path, model_path, output_csv_path, is_dqn=False):
    print(f"Loading Dataset: {dataset_path}")
    H_file = h5py.File(dataset_path, 'r')
    H = np.array(H_file.get('H'))
    se_max_ur = np.array(H_file.get('se_max'))
    
    # Using the exact same test split logic as SAC_main.py
    train_ratio = 0.8
    num_samples = H.shape[0]
    num_train = int(train_ratio * num_samples)
    
    H_test = H[num_train:]
    se_max_test = se_max_ur[num_train:]
    print(f"Testing samples: {H_test.shape[0]}")

    env = MimoEnv(H_test, se_max_test)
    
    num_states = env.observation_space.shape[0]
    num_actions = len([env.action_space.sample()])
    max_actions = env.action_space.n

    # ======================== Initialize SAC agent ========================
    args = SACArgs(H_test, max_episode=1)
    agent = SAC(num_states, num_actions, max_actions, args, args.lr, args.alpha_lr)
    
    print(f"Loading Model: {model_path}")
    if is_dqn:
        pass # Handle DQN case if needed later. Assuming SAC primarily for now based on SAC_main.py 
    else:
        agent.load_checkpoint(model_path)
    
    # Symbolic tools Instantiation
    kpis = KPI_LIST
    quantile_manager = QuantileManager(kpis + ['scheduled_user'])
    quantile_manager.reset()
    quantile_manager.partial_fit("scheduled_user", [0])
    quantile_manager.partial_fit("scheduled_user", [7])

    symbolic_df = pd.DataFrame()
    symbolizer = Symbolizer(quantile_manager=quantile_manager, kpi_list=kpis, users=USERS)

    observation, info = env.reset()
    done = False
    
    print("Starting generation...")
    # Episode loop
    while not done:
        action, final_action = agent.select_action(observation)
        ue_select, idx = sel_ue(final_action[0])
        
        buff_ac = [(observation, action)]
        curr_states_df, curr_actions_rewards_df = process_buffer(buff_ac, transform_action, sel_ue, mode=None, timestep=info['current_step'])

        state_t_df = curr_states_df[curr_states_df['timestep'] == info['current_step']]
        decision_t_df = curr_actions_rewards_df[curr_actions_rewards_df['timestep'] == info['current_step']]

        symbolic_form = symbolizer.create_symbolic_form(state_t_df, decision_t_df)
        
        next_obs, reward, done, _, info = env.step(final_action[0])
        
        if not symbolic_form.empty:
            symbolic_form['reward'] = [reward] * symbolic_form.shape[0]
            # Maintain a timestep counter valid for the data
            if symbolic_df.empty:
                current_max_ts = 0
            else:
                current_max_ts = symbolic_df['timestep'].max()
            symbolic_form['timestep'] = [current_max_ts + 1] * symbolic_form.shape[0]
            symbolic_df = pd.concat([symbolic_df, symbolic_form], ignore_index=True)

        symbolizer.step()
        observation = next_obs
        print(f'Step: {info["current_step"]} / {env.total_step - 1}', end='\r')
        
    symbolic_df = symbolic_df.reset_index(drop=True)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    symbolic_df.to_csv(output_csv_path, index=False)
    print(f"\nSaved symbolic data to {output_csv_path}")


if __name__ == "__main__":
    # Example runs based on notebook expected plots
    experiments = [
        {
            "dataset": f"{PROJ_ADDR}/Datasets/LOS_highspeed2_64_7.hdf5",
            "model": "models/SACG_484.87_300_dtLOS_HS2_final.pth",
            "output": f"{PROJ_ADDR}/SAC/Agents_Numeric_Symbolic_Raw_Data/DS_LOS_HS2-Agent-SACG_HS2_1000-AS_No.csv"
        },
        # You can add the LS / NLOS combinations here
    ]
    
    # Note: We need the user to run training to get the specific model files or point to existing ones.
    # The plot notebook hardcodes specific CSV filenames like DS_LOS_LS-Agent-SACG_LS_80-AS_No.csv
    # This script provides the framework to generate them. 
    # Let's generate a sample to prove it works if the model exists.

    models_dir = os.path.join(script_dir, "models")
    if os.path.exists(models_dir):
        model_files = sorted([f for f in os.listdir(models_dir) if f.endswith(".pth")])
        if model_files:
            chosen_model = os.path.join(models_dir, model_files[-1]) # Use latest model 
            
            # Decide dataset based on model name if possible, else default to HS2
            dataset = f"{PROJ_ADDR}/Datasets/LOS_highspeed2_64_7.hdf5"
            if "LOS_HS2" in chosen_model or "LOS_highspeed" in chosen_model:
                output_name = "DS_LOS_HS2-Agent-SACG_HS2_1000-AS_No.csv"
            else:
                output_name = "DS_Custom-Agent-SACG-AS_No.csv"
                
            output = f"{PROJ_ADDR}/SAC/Agents_Numeric_Symbolic_Raw_Data/{output_name}"
            generate_plot_data(dataset, chosen_model, output)
        else:
            print(f"No models found in the '{models_dir}' directory. Please run SAC_main.py first.")
    else:
        print(f"The '{models_dir}' directory does not exist. Please run SAC_main.py first.")
