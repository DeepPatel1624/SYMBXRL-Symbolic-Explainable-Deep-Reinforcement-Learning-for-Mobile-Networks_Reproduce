"""
====================================================================
Action Steering Utilities
====================================================================

--------------------------------------------------------------------
Purpose
--------------------------------------------------------------------

Deep Reinforcement Learning agents learn policies through interaction
with an environment. However, these policies may sometimes produce
suboptimal actions due to exploration noise or imperfect convergence.

Action Steering provides a mechanism to guide the agent toward
better actions using historical experience and symbolic knowledge.

Instead of blindly following the RL policy, the algorithm:
    1. Analyzes the current symbolic state
    2. Searches historical transitions with similar states
    3. Uses the Decision Graph to identify promising actions
    4. Selects actions with higher expected reward

This improves stability and interpretability of RL decisions.

--------------------------------------------------------------------
2. Conceptual Pipeline
--------------------------------------------------------------------

The Action Steering pipeline works as follows:

    Current State
          ↓
    Symbolic Representation
          ↓
    Decision Graph Lookup
          ↓
    Historical Filtering
          ↓
    Action Selection
          ↓
    Steered Action

--------------------------------------------------------------------
3. Mathematical Idea
--------------------------------------------------------------------

Action steering uses the expected reward of transitions.

For a decision transition:
        d_i → d_j

We estimate:
        mean_reward(d_i → d_j)

where
        mean_reward = total_reward / occurrence

Top transitions are selected as candidate actions.

--------------------------------------------------------------------
4. Randomized Steering
--------------------------------------------------------------------

The randomized version uses Softmax weighting.

Softmax converts rewards into probabilities:

        P(a_i) = exp(r_i) / Σ exp(r_k)

This ensures:
    • High reward actions more likely
    • Low reward actions still possible

This balances exploitation and exploration.

--------------------------------------------------------------------
5. Data Structures
--------------------------------------------------------------------

Main inputs used by the algorithm:

curr_state_df
    Current symbolic state dataframe

history_df
    Historical symbolic state/action dataset

rt_decision_graph
    Decision graphs built during runtime

--------------------------------------------------------------------
6. Output
--------------------------------------------------------------------

The algorithm returns:

    scheduled users
    associated reward

If no suitable action is found:

    return False, None

--------------------------------------------------------------------
#######################################################################
--------------------------------------------------------------------

This module enables:
    • RL policy correction
    • symbolic reasoning integration
    • explainable scheduling decisions
    • improved reward performance

It bridges symbolic reasoning and reinforcement learning.

====================================================================
"""

# ================= IMPORTS =================
# ast: safely converts string representations of Python objects into actual objects
# random: used for probabilistic action selection
# numpy: numerical operations (used for softmax)
# pandas: dataframe operations for state/history processing
import ast
import random
import numpy as np
import pandas as pd


# ================================================================
# Extract Final Decision From Symbolic Suggestions
# ================================================================
def extract_decision_from_suggested(suggested_decision):
    """
    Extracts and sorts decisions from a list of suggested decisions.

    suggested_decision may contain:
        - string representations of lists
        - actual lists

    Each item typically has structure:
        [scheduled_users, unscheduled_users]

    We extract scheduled users and merge them.

    Example:
        input:
            ["([1,2],[3,4])", "([0],[1,2,3])"]

        output:
            (0,1,2)

    Returns:
        tuple of sorted scheduled users
    """

    # list that will collect all scheduled users
    extracted_decision = []

    for item in suggested_decision:

        # convert string → python object safely
        # ast.literal_eval avoids security risks of eval()
        converted_decision = ast.literal_eval(item) if type(item) == str else item

        # converted_decision[0] contains scheduled users
        extracted_decision.extend(converted_decision[0])

    # return sorted tuple for deterministic representation
    return tuple(sorted(extracted_decision))


# ================================================================
# Deterministic Action Steering
# ================================================================
def do_action_steering_this_timestep(curr_state_df, history_df, rt_decision_graph):
    """
    Performs action steering based on:
        - current symbolic state
        - historical symbolic data
        - decision graph knowledge

    Idea:
        Instead of trusting the RL agent blindly, we search
        past timesteps with similar states and choose the action
        that historically produced the best reward.

    Returns:
        scheduled members
        reward
    """

    # --------------------------------------------------------------
    # get the previous timestep state from history
    # --------------------------------------------------------------
    prev_state_df = history_df[
        history_df['timestep'] == history_df['timestep'].tail(1).iloc[0]
    ]

    # groups present in current state
    groups = curr_state_df['group'].unique()

    # initialize candidate timesteps as all history
    common_timesteps = set(history_df['timestep'])


    # --------------------------------------------------------------
    # process each user group independently
    # --------------------------------------------------------------
    for group in groups:

        # if previous state contains this group
        if not prev_state_df[prev_state_df['group'] == group].empty:

            # retrieve decision graph for the group
            G = rt_decision_graph[group].get_graph(mode="networkX")

            # previous decision node
            node_id = prev_state_df[prev_state_df['group'] == group]['decision'].iloc[0]

            # possible next decisions
            neighbors = list(G.neighbors(node_id))


            # ------------------------------------------------------
            # evaluate neighbor decisions using mean reward
            # ------------------------------------------------------
            neighbors_with_mean_rewards = []

            for neighbor in neighbors:

                # retrieve edge statistics
                edge_data = G.get_edge_data(node_id, neighbor)

                # expected reward for that transition
                mean_reward = edge_data.get('mean_reward', 0)

                neighbors_with_mean_rewards.append((neighbor, mean_reward))


            # ------------------------------------------------------
            # sort candidate decisions by expected reward
            # ------------------------------------------------------
            neighbors_with_mean_rewards.sort(
                key=lambda x: x[1],
                reverse=True
            )


            # take top 3 promising decisions
            top_neighbors = neighbors_with_mean_rewards[:3]


            # ------------------------------------------------------
            # candidate actions include:
            #   best transitions + current agent decision
            # ------------------------------------------------------
            actions = [x[0] for x in top_neighbors] + [
                curr_state_df[curr_state_df['group'] == group]['decision'].iloc[0]
            ]


            # ------------------------------------------------------
            # filter history to find matching states
            # ------------------------------------------------------
            conditioned_timesteps = set(
                history_df[
                    (
                        (history_df['timestep'].isin(common_timesteps)) &
                        (history_df['group'] == group) &
                        (history_df['group_members'] ==
                         curr_state_df[curr_state_df['group'] == group]['group_members'].iloc[0]) &
                        (history_df['MSEUr'] ==
                         curr_state_df[curr_state_df['group'] == group]['MSEUr'].iloc[0]) &
                        (history_df['DTUr'] ==
                         curr_state_df[curr_state_df['group'] == group]['DTUr'].iloc[0]) &
                        (history_df['decision'].isin(actions))
                    )
                ]['timestep']
            )

            # keep only timesteps satisfying all groups
            common_timesteps &= conditioned_timesteps

        else:
            # ------------------------------------------------------
            # case where group has no previous state
            # ------------------------------------------------------
            group_timesteps = set(
                history_df[
                    (
                        (history_df['timestep'].isin(common_timesteps)) &
                        (history_df['group'] == group) &
                        (history_df['group_members'] ==
                         curr_state_df[curr_state_df['group'] == group]['group_members'].iloc[0]) &
                        (history_df['MSEUr'] ==
                         curr_state_df[curr_state_df['group'] == group]['MSEUr'].iloc[0]) &
                        (history_df['DTUr'] ==
                         curr_state_df[curr_state_df['group'] == group]['DTUr'].iloc[0])
                    )
                ]['timestep']
            )

            common_timesteps &= group_timesteps


        # if no valid historical state remains
        if len(common_timesteps) == 0:
            return False, None


    # --------------------------------------------------------------
    # choose best timestep based on reward
    # --------------------------------------------------------------
    action_steered_timestep = (
        history_df[history_df['timestep'].isin(common_timesteps)]
        .groupby('timestep')
        .first()
        .reset_index()
        .nlargest(1, 'reward')['timestep']
        .iloc[0]
    )

    # return scheduled members and reward
    return (
        history_df[history_df['timestep'] == action_steered_timestep]['sched_members'],
        history_df[history_df['timestep'] == action_steered_timestep]['reward'].iloc[0]
    )


# ================================================================
# Softmax Function
# ================================================================
def softmax(x):
    """
    Converts rewards into probabilities.

    Mathematical formula:

        softmax(x_i) = exp(x_i) / sum(exp(x_j))

    Used to randomly select actions with probability
    proportional to their reward.
    """

    e_x = np.exp(x - np.max(x))  # numerical stability trick
    return e_x / e_x.sum()


# ================================================================
# Randomized Action Steering
# ================================================================
def do_action_steering_this_timestep_randomized(
        curr_state_df,
        history_df,
        rt_decision_graph,
        agent_expected_reward):
    """
    Same logic as deterministic steering but introduces randomness.

    Steps:
        1. find matching historical states
        2. filter actions with reward >= agent expectation
        3. apply softmax on reward
        4. randomly select action

    This balances:
        exploration vs exploitation.
    """

    prev_state_df = history_df[
        history_df['timestep'] == history_df['timestep'].tail(1).iloc[0]
    ]

    groups = curr_state_df['group'].unique()
    common_timesteps = set(history_df['timestep'])

    # group filtering identical to deterministic version
    for group in groups:

        if not prev_state_df[prev_state_df['group'] == group].empty:

            G = rt_decision_graph[group].get_graph(mode="networkX")
            node_id = prev_state_df[prev_state_df['group'] == group]['decision'].iloc[0]
            neighbors = list(G.neighbors(node_id))

            neighbors_with_mean_rewards = []

            for neighbor in neighbors:
                edge_data = G.get_edge_data(node_id, neighbor)
                mean_reward = edge_data.get('mean_reward', 0)
                neighbors_with_mean_rewards.append((neighbor, mean_reward))

            neighbors_with_mean_rewards.sort(key=lambda x: x[1], reverse=True)

            top_neighbors = neighbors_with_mean_rewards[:3]

            actions = [x[0] for x in top_neighbors] + [
                curr_state_df[curr_state_df['group'] == group]['decision'].iloc[0]
            ]

            conditioned_timesteps = set(
                history_df[
                    (
                        (history_df['timestep'].isin(common_timesteps)) &
                        (history_df['group'] == group) &
                        (history_df['group_members'] ==
                         curr_state_df[curr_state_df['group'] == group]['group_members'].iloc[0]) &
                        (history_df['MSEUr'] ==
                         curr_state_df[curr_state_df['group'] == group]['MSEUr'].iloc[0]) &
                        (history_df['DTUr'] ==
                         curr_state_df[curr_state_df['group'] == group]['DTUr'].iloc[0]) &
                        (history_df['decision'].isin(actions))
                    )
                ]['timestep']
            )

            common_timesteps &= conditioned_timesteps

        else:

            group_timesteps = set(
                history_df[
                    (
                        (history_df['timestep'].isin(common_timesteps)) &
                        (history_df['group'] == group) &
                        (history_df['group_members'] ==
                         curr_state_df[curr_state_df['group'] == group]['group_members'].iloc[0]) &
                        (history_df['MSEUr'] ==
                         curr_state_df[curr_state_df['group'] == group]['MSEUr'].iloc[0]) &
                        (history_df['DTUr'] ==
                         curr_state_df[curr_state_df['group'] == group]['DTUr'].iloc[0])
                    )
                ]['timestep']
            )

            common_timesteps &= group_timesteps

        if len(common_timesteps) == 0:
            return False, None


    # keep timesteps with reward >= expected reward
    better_timesteps = history_df[
        (history_df['timestep'].isin(common_timesteps)) &
        (history_df['reward'] >= agent_expected_reward)
    ]

    if better_timesteps.empty:
        return False, None


    # compute average reward per decision
    action_rewards = better_timesteps.groupby(
        'decision',
        as_index=False)['reward'].mean()


    # convert reward → probability
    action_rewards['weight'] = softmax(action_rewards['reward'])


    # randomly sample action
    chosen_action = random.choices(
        population=action_rewards['decision'].tolist(),
        weights=action_rewards['weight'].tolist(),
        k=1
    )[0]


    # find best timestep corresponding to that action
    action_steered_timestep = (
        better_timesteps[
            better_timesteps['decision'] == chosen_action
        ]
        .sort_values(by='reward', ascending=False)
        .iloc[0]['timestep']
    )


    return (
        history_df[history_df['timestep'] == action_steered_timestep]['sched_members'],
        history_df[history_df['timestep'] == action_steered_timestep]['reward'].iloc[0]
    )


# ================================================================
# Transform Continuous SAC Action → Discrete Action
# ================================================================
def transform_action(action, high=1, low=-1, tot_act=127):
    """
    SAC outputs continuous actions in [-1,1].

    Scheduler actions are discrete:
        {0,1,...,126}

    Linear transformation:

        k = (high-low)/(tot_act-1)

        discrete_action =
        round((action - low) / k)
    """

    k = (high - low) / (tot_act - 1)
    return round((action - low) / k)


# ================================================================
# Convert Replay Buffer → DataFrames
# ================================================================
def process_buffer(buff, transform_action, sel_ue, mode, timestep=0, agent_type='SAC'):
    """
    Processes a buffer of transitions and returns two DataFrames: one for states and one for actions and rewards.
    Args:
        buff (list): A list of transitions, where each transition is a tuple (state, action).
        transform_action (function): A function to transform the action if the agent type is 'SAC'.
        sel_ue (function): A function to select the user equipment (UE) from the action.
        mode (str): The mode of processing, either 'buffer' or another mode.
        timestep (int, optional): The timestep to use if mode is not 'buffer'. Defaults to 0.
        agent_type (str, optional): The type of agent, defaults to 'SAC'.
    Returns:
        tuple: A tuple containing two DataFrames:
            - states_df (pd.DataFrame): DataFrame containing the processed states.
            - actions_rewards_df (pd.DataFrame): DataFrame containing the processed actions and rewards.
    """

    buff_state_columns = ["MSEUr0", "MSEUr1", "MSEUr2", "MSEUr3", "MSEUr4", "MSEUr5", "MSEUr6",
                 "DTUr0", "DTUr1", "DTUr2", "DTUr3", "DTUr4", "DTUr5", "DTUr6",
                 "UGUr0", "UGUr1", "UGUr2", "UGUr3", "UGUr4", "UGUr5", "UGUr6"]
    buff_states = []
    buff_actions_rewards = []

    for transition in buff:
        state, action = transition

        state_1d = state.flatten()
        buff_states.append(state_1d)

        action_reward = [action[0]]
        buff_actions_rewards.append(action_reward)

    states_df = pd.DataFrame(buff_states, columns=buff_state_columns)
 
    if mode == 'buffer':
        states_df['timestep'] = states_df.index + 1
    else:
        states_df['timestep'] = timestep
    cols = ['timestep'] + [col for col in states_df.columns if col != 'timestep']
    states_df = states_df[cols]
    actions_rewards_df = pd.DataFrame(buff_actions_rewards, columns=["action"])
    if agent_type == 'SAC':
        actions_rewards_df["action"] = actions_rewards_df["action"].apply(transform_action)
    actions_rewards_df["action"] = actions_rewards_df["action"].apply(lambda x: sel_ue(x)[0])
    if mode == 'buffer':
        actions_rewards_df['timestep'] = actions_rewards_df.index + 1
    else:
        actions_rewards_df['timestep'] = timestep
    cols = ['timestep'] + [col for col in actions_rewards_df.columns if col != 'timestep']
    actions_rewards_df = actions_rewards_df[cols]

    return states_df, actions_rewards_df