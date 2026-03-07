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
    """
    QuantileManager
    ----------------
    This class maintains quantile estimators for multiple KPIs (Key Performance Indicators).
    
    In SymbXRL, numerical KPI values (e.g., spectral efficiency, fairness, etc.)
    are converted into symbolic categories such as Q1, Q2, Q3, Q4.

    Instead of storing all historical KPI data to compute quartiles,
    this class uses the P2 (P-Square) online quantile approximation algorithm.

    Advantages:
    - Memory efficient
    - Works in streaming data settings
    - Updates quantiles incrementally as new data arrives
    """

    def __init__(self, kpi_list, p=50):
        """
        Initialize the quantile manager.

        Parameters
        ----------
        kpi_list : list
            List of KPI names that require quantile estimation.
            Example: ["MSE", "DTU", "scheduled_user"]

        p : int
            Target percentile for the quantile approximation.
            Default = 50 (median). The P² algorithm internally
            estimates the quartile markers.
        """

        # Create one quantile approximator per KPI
        # Dictionary structure:
        # {
        #   "MSE": PSquareQuantileApproximator,
        #   "DTU": PSquareQuantileApproximator,
        #   ...
        # }
        self.quantile_approximators = {
            kpi: PSquareQuantileApproximator(p) for kpi in kpi_list
        }

    def fit(self):
        """
        Initialize all quantile approximators.

        The P2 algorithm requires an initial fitting stage.
        Here we initialize them with an empty dataset.

        In practice, they will quickly adapt as new values
        are streamed through `partial_fit`.
        """

        for approximator in self.quantile_approximators.values():
            approximator.fit([])

    def partial_fit(self, kpi_name, value):
        """
        Update the quantile estimator for a specific KPI.

        This function is called at every timestep to incorporate
        new observations of KPI values.

        Parameters
        ----------
        kpi_name : str
            Name of the KPI being updated.

        value : list or numpy array
            New KPI values observed at the current timestep.

        Example
        -------
        partial_fit("MSE", [3.2, 2.8, 4.1])
        """

        # Only update if the KPI exists
        if kpi_name in self.quantile_approximators:
            self.quantile_approximators[kpi_name].partial_fit(value)

    def get_markers(self, kpi_name):
        """
        Retrieve quantile markers for a specific KPI.

        The P² algorithm maintains 5 markers representing
        key quantile boundaries:

        markers = [q0, q1, q2, q3, q4]

        where:
        q0 = minimum
        q1 = first quartile (25%)
        q2 = median (50%)
        q3 = third quartile (75%)
        q4 = maximum

        Returns
        -------
        list
            List of quantile markers if available.
            Returns empty list if KPI does not exist.
        """

        if kpi_name in self.quantile_approximators:
            return self.quantile_approximators[kpi_name].get_markers()
        else:
            return []

    def reset(self):
        """
        Reset all quantile estimators.

        Useful when restarting an experiment or beginning
        a new dataset.

        NOTE:
        Current implementation resets markers to
        [1, 2, 3, 4, 5], which may not be ideal.
        The TODO comment in the original code suggests
        improving this initialization.
        """

        for kpi in self.quantile_approximators:
            self.quantile_approximators[kpi].reset()

    def represent_markers(self):
        """
        Represent the quantile markers for all KPIs
        as a pandas DataFrame.

        This is mainly used for debugging, visualization,
        or analysis.

        Output format example
        ---------------------

        KPI      q0    q1    q2    q3    q4
        -----------------------------------
        MSE     0.5   1.2   2.3   3.4   4.8
        DTU     0.3   0.9   1.8   2.7   3.9

        Returns
        -------
        pandas.DataFrame
            Table of quantile markers for each KPI.
        """

        markers_data = []

        # Iterate through each KPI's quantile estimator
        for kpi in self.quantile_approximators:

            # Retrieve marker values
            markers = self.get_markers(kpi)

            # Valid marker list should contain 5 values
            if len(markers) == 5:

                markers_data.append({
                    "kpi": kpi,
                    "q0": markers[0],  # minimum
                    "q1": markers[1],  # 25th percentile
                    "q2": markers[2],  # median
                    "q3": markers[3],  # 75th percentile
                    "q4": markers[4],  # maximum
                })

        # Convert list of dictionaries to DataFrame
        return pd.DataFrame(markers_data)


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






