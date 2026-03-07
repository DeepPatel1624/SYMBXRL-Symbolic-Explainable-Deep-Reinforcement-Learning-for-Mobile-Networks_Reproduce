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

    """
    Symbolizer
    ----------
    This class converts raw RL environment data (states + actions)
    into symbolic predicates used by the SymbXRL explanation layer.

    The symbolic representation describes:
        - KPI changes (increase / decrease / constant)
        - Scheduling decisions
        - Group-level behaviour

    Example symbolic output:
        inc(MSE, Q3)
        dec(DTU, Q1)
        inc(G1, Q2, 50)

    where:
        inc / dec / const  → change direction
        Q1..Q4             → KPI quartile
        G1                 → user group
        50                 → percent scheduled
    """

    def __init__(self, quantile_manager: QuantileManager, kpi_list, users):
        """
        Initialize the Symbolizer.

        Parameters
        ----------
        quantile_manager : QuantileManager
            Object responsible for computing KPI quartiles.

        kpi_list : list
            List of KPI names (example: ["MSE", "DTU"])

        users : list
            List of user indices in the system (example: [0..6])
        """

        # Quantile manager used to categorize KPI values
        self.quantile_manager = quantile_manager

        # List of KPI names
        self.kpis_name_list = kpi_list

        # User IDs in the system
        self.users = users

        # Previous timestep state per group
        self.prev_state_df = {}

        # Previous timestep decisions per group
        self.prev_decision_df = {}

        # Temporary storage for current timestep
        self.prev_state_candid_df = {}
        self.prev_decision_candid_df = {}



    def create_symbolic_form(self, state_t_df, decision_t_df):
        """
        Convert a timestep state + decision into symbolic predicates.

        Parameters
        ----------
        state_t_df : pandas.DataFrame
            Environment state at timestep t

        decision_t_df : pandas.DataFrame
            RL agent action at timestep t

        Returns
        -------
        pandas.DataFrame
            Symbolic representation of effects
        """

        effects_symbolic_representation = []

        # Extract groups and their members
        groups = self._get_list_of_existing_groups_in_timestep(state_t_df)

        # Convert agent action to full representation
        agent_complete_decision = self._get_actions_full_represetntation(
            decision_t_df['action'].iloc[0]
        )

        # Process each group separately
        for group, group_members in groups.items():

            members_refined = None

            # Build column names for the KPIs belonging to this group
            kpi_columns = list(np.concatenate([
                self._get_list_of_kpi_column_for_users(kpi_name, group_members)
                for kpi_name in self.kpis_name_list
            ]))

            # If previous state exists → compute symbolic transition
            if group in self.prev_state_df:

                # Compute symbolic KPI change
                group_symbolic_effect = self._calculate_kpi_symbolic_state(
                    state_t_df,
                    self.prev_state_df[group],
                    group_members
                )

                # Compute symbolic scheduling change
                group_symbolic_decision = self._calculate_decision_symbolic_state(
                    decision_t_df,
                    self.prev_decision_df[group],
                    group,
                    group_members
                )

                # Split users into scheduled / unscheduled
                members_refined = self._clean_member_state_according_to_scheduling(
                    group_members,
                    decision_t_df['action'].iloc[0]
                )

                # Store symbolic record
                effects_symbolic_representation.append({
                    "timestep": state_t_df['timestep'].iloc[0],
                    "group": group,
                    "group_members": str(group_members),

                    # symbolic KPI predicates
                    **group_symbolic_effect,

                    # scheduled user list
                    "sched_members": str(members_refined),

                    # full action representation
                    "sched_members_complete": str(agent_complete_decision),

                    # symbolic decision predicate
                    "decision": group_symbolic_decision
                })

            else:
                # If this group appears for first time
                members_refined = self._clean_member_state_according_to_scheduling(
                    group_members,
                    decision_t_df['action'].iloc[0]
                )

            # Save decision for next timestep comparison
            decision_to_be_rememeberd = decision_t_df.copy()
            decision_to_be_rememeberd.at[
                decision_to_be_rememeberd.index[0],
                'action'
            ] = members_refined[0]

            # Store current state for next timestep
            self.prev_state_candid_df[group] = state_t_df[kpi_columns]

            self.prev_decision_candid_df[group] = decision_to_be_rememeberd

            # Update quantile estimator for scheduled user count
            self.quantile_manager.partial_fit(
                'scheduled_user',
                [len(members_refined[0])]
            )

        # Update KPI quartile estimators
        self._add_timestep_kpi_data_to_approximator(state_t_df)

        return pd.DataFrame(effects_symbolic_representation)



    def step(self):
        """
        Move the symbolizer to next timestep.

        Copies candidate state/decision to previous state/decision.
        """

        self.prev_state_df = self.prev_state_candid_df.copy()
        self.prev_decision_df = self.prev_decision_candid_df.copy()



    def _clean_member_state_according_to_scheduling(self, members_list, decision):
        """
        Separate scheduled and unscheduled users.

        Returns
        -------
        [scheduled_users, unscheduled_users]
        """

        decision = set(ast.literal_eval(decision)
                       if not isinstance(decision, tuple)
                       else decision)

        scheduled_members = [
            member for member in members_list if member in decision
        ]

        unscheduled_members = [
            member for member in members_list if member not in decision
        ]

        return [scheduled_members, unscheduled_members]



    def _calculate_decision_symbolic_state(
            self,
            current_decision_df,
            previous_decision,
            group_num,
            group_users
    ):
        """
        Convert scheduling decision change into symbolic predicate.

        Example output:
            inc(G1, Q2, 50)
        """

        current_decision = self._clean_member_state_according_to_scheduling(
            group_users,
            current_decision_df['action'].iloc[0]
        )

        previous_decision = ast.literal_eval(
            previous_decision['action'].iloc[0]
        ) if not isinstance(previous_decision['action'].iloc[0], list) \
            else previous_decision['action'].iloc[0]

        scheduled_users_count = len(current_decision[0])

        total_users_count = (
                len(current_decision[0]) +
                len(current_decision[1])
        )

        # Determine predicate direction
        predicate = "const"

        if scheduled_users_count > len(previous_decision):
            predicate = "inc"
        elif scheduled_users_count < len(previous_decision):
            predicate = "dec"

        # Group name
        group_name = f"G{group_num}"

        # Quartile category
        quartile = self._get_kpi_quantile(
            "scheduled_user",
            scheduled_users_count
        )

        # Percent of users scheduled
        scheduled_percentage = round(
            (scheduled_users_count / total_users_count) * 100 / 25
        ) * 25

        return f"{predicate}({group_name}, {quartile}, {scheduled_percentage})"



    def _calculate_kpi_symbolic_state(self,
                                      curr_state_df,
                                      prev_state_df,
                                      members):
        """
        Generate symbolic predicates for KPI changes.

        Example output:

            inc(MSE, Q3)
            dec(DTU, Q1)
        """

        kpi_symbolic_representatino = {}

        for kpi_group in self.kpis_name_list:

            curr_mean = round(
                curr_state_df[
                    self._get_list_of_kpi_column_for_users(
                        kpi_group,
                        members
                    )
                ].iloc[0].mean(),
                4
            )

            prev_mean = round(
                prev_state_df.filter(regex=f"^{kpi_group}")
                .iloc[0].mean(),
                4
            )

            kpi_symbolic_representatino[f'{kpi_group}'] = \
                self._define_MSE_or_DTU_symbolic_state(
                    curr_mean,
                    prev_mean,
                    f'{kpi_group}',
                    kpi_group
                )

        return kpi_symbolic_representatino



    def _define_MSE_or_DTU_symbolic_state(
            self,
            curr_value,
            prev_value,
            kpi_column,
            kpi_name
    ):
        """
        Convert KPI numeric change into symbolic predicate.
        """

        change_percentage = self._find_change_percentage(
            curr_value,
            prev_value
        )

        predicate = self._get_predicate(change_percentage)

        return f'{predicate}({kpi_column}, {self._get_kpi_quantile(kpi_name, curr_value)})'



    def _find_change_percentage(self, curr_value, prev_value):
        """
        Compute percentage change between timesteps.
        """

        if prev_value == 0:

            if curr_value == 0:
                return 0

            else:
                return 'inf'

        else:

            return int(
                100 * (curr_value - prev_value) / prev_value
            )



    def _get_predicate(self, change_percentage):
        """
        Convert percentage change into predicate.
        """

        if change_percentage == 'inf':
            return "inc"

        elif change_percentage > KPI_CHANGE_THRESHOLD_PERCENT:
            return "inc"

        elif change_percentage < -KPI_CHANGE_THRESHOLD_PERCENT:
            return "dec"

        else:
            return "const"



    def _get_kpi_quantile(self, kpi_name, kpi_value):
        """
        Map KPI value to quartile category.
        """

        markers = self.quantile_manager.get_markers(kpi_name)

        if len(markers) < 5:
            return "NaN"

        if kpi_value <= markers[1]:
            return "Q1"

        elif kpi_value <= markers[2]:
            return "Q2"

        elif kpi_value <= markers[3]:
            return "Q3"

        elif kpi_value <= 0.999 * markers[4]:
            return "Q4"

        else:
            return "MAX"



    def _add_timestep_kpi_data_to_approximator(self, timestep_df):
        """
        Feed KPI values into quantile estimators.
        """

        for kpi_name in self.kpis_name_list:

            kpi_columns = self._get_list_of_kpi_column_for_users(
                kpi_name,
                self.users
            )

            self.quantile_manager.partial_fit(
                kpi_name,
                timestep_df[kpi_columns].iloc[0].to_numpy()
            )



    def _get_list_of_kpi_column_for_users(self, kpi_name, user_list):
        """
        Generate KPI column names for given users.

        Example:
            kpi_name = "MSE"
            users = [0,1,2]

        Returns:
            ["MSE0","MSE1","MSE2"]
        """

        return [f'{kpi_name}{user}' for user in user_list]



    def _get_list_of_existing_groups_in_timestep(self, data):
        """
        Extract group membership of users.

        Returns
        -------
        dict

        Example:

            {0:[1,3], 1:[0,2,4]}
        """

        groups = {}

        for i in self.users:

            group_number = int(data[f'UGUr{i}'].iloc[0])

            if group_number in groups:
                groups[group_number].append(i)

            else:
                groups[group_number] = [i]

        return groups



    def _get_actions_full_represetntation(self, agent_action_tuple):
        """
        Convert scheduled users into
        [scheduled, unscheduled]
        """

        members = list(agent_action_tuple)

        full_set = set(range(7))

        input_set = set(agent_action_tuple)

        missing_numbers = list(full_set - input_set)

        return [members, missing_numbers]