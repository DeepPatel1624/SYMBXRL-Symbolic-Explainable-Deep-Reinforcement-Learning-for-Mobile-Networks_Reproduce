#================ IMPORTS =====================

from Action_Steering.p_square_quantile_approximator import PSquareQuantileApproximator
import pandas as pd
import numpy as np
import ast


KPI_CHANGE_THRESHOLD_PERCENT = 5

#=============== SYMBOLIZER DEFINATION =========
"""
# - Symbolizer that receives one or 2 timestep of data
# - returns the symbolic representation
"""

class QuantileManager:
   
    '''
        A class to manage quantile approximations 
        for multiple KPIs 
        using the PSquareQuantileApproximator.
    '''

    def __init__(self):
        pass

    def fit():
        pass 

    def partial_fit():
        pass

    def get_markers():
        pass

    def reset():
        pass


    def represnt_markers():
        pass


class Symbolizer:

    def __init__():
        pass

    def create_symbolic_form():
        pass

    def step():
        pass

    def _clean_member_state_according_to_scheduling(self, members_list, decision):
        pass

    def _calculate_decision_symbolic_state(self, current_decision_df, previous_decision, group_num, group_users):
        pass

    def _calculate_kpi_symbolic_state(self, curr_state_df:pd.DataFrame, prev_state_df:pd.DataFrame, members:list):
        pass

    def _define_MSE_or_DTU_symbolic_state(self, curr_value, prev_value, kpi_column, kpi_name):
        pass

    def _find_change_percentage(self, curr_value, prev_value):
        pass


    def _get_predicate():
        pass

    def _get_kpi_quantile():
        pass

    def _add_timestep_kpi_data_to_approximator(self, timestep_df):
        pass

    def _get_list_of_kpi_column_for_users(self, kpi_name, user_list):
        pass

    def _get_list_of_existing_groups_in_timestep(self, data):
        pass

    def _get_actions_full_represetntation(self, agent_action_tuple):
        pass

    




