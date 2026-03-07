"""
====================================================================
DecisionGraph Module
====================================================================
Purpose: Construct an explainable decision graph from symbolic RL data.

--------------------------------------------------------------------
1. Motivation
--------------------------------------------------------------------

Deep Reinforcement Learning models (such as SAC or DQN) often behave
as black boxes. While they can learn effective policies, it is difficult
to understand *why* a particular decision is taken.

The SymbXRL framework introduces an explainability layer by:

    RL Agent → Symbolic Representation → Decision Graph

This module implements the **Decision Graph**, which represents the
policy learned by the RL agent in the form of a directed graph.

Nodes represent symbolic decisions and edges represent transitions
between decisions over time.

--------------------------------------------------------------------
2. Graph Representation
--------------------------------------------------------------------

The decision graph is represented using a directed graph:

        G = (V, E)

where:

    V = set of decision nodes
    E = set of directed transitions between decisions

Example:

    decision_t   →   decision_t+1

Each node represents a symbolic decision such as:

    inc(G1, Q3, 50)

Meaning:

    inc = increase
    G1  = user group 1
    Q3  = KPI quartile
    50  = percentage of users scheduled


--------------------------------------------------------------------
3. Node Statistics
--------------------------------------------------------------------

Each node stores statistics describing the RL agent behavior.

Attributes stored for each node:

    occurrence     = number of times this decision was taken

    total_reward   = cumulative reward obtained when taking
                     this decision

    mean_reward    = average reward for this decision


Mathematically:

    mean_reward(d) = total_reward(d) / occurrence(d)


--------------------------------------------------------------------
4. Node Probability
--------------------------------------------------------------------

Node probability measures how frequently the RL agent takes
a particular decision.

Formula:

    P(d) = occurrence(d) / Σ occurrence(all decisions)


This tells us:

    "How likely is the agent to take decision d?"

Example:

    decision A occurred 20 times
    total decisions = 100

    P(A) = 0.20


--------------------------------------------------------------------
5. Edge Statistics (Decision Transitions)
--------------------------------------------------------------------

Edges represent transitions between consecutive decisions.

Example:

    d1 → d2

Meaning:

    after decision d1 the agent moves to decision d2.


Each edge stores:

    occurrence
    total_reward
    mean_reward


Edge mean reward:

    mean_reward(d1 → d2) =
        total_reward(d1 → d2) / occurrence(d1 → d2)


--------------------------------------------------------------------
6. Edge Probability
--------------------------------------------------------------------

Edge probability represents the likelihood of moving from one
decision to another.

This is a conditional probability:

        P(d_j | d_i)

Formula:

    P(d_j | d_i) =
        occurrence(d_i → d_j)
        ----------------------
        Σ occurrence(d_i → d_k)

where:

    d_i = source decision
    d_j = next decision


Example:

    A → B : 30
    A → C : 10

Then:

    P(B | A) = 30 / 40 = 0.75
    P(C | A) = 10 / 40 = 0.25


--------------------------------------------------------------------
7. Node Size Scaling
--------------------------------------------------------------------

Node sizes in the visualization are scaled according to
occurrence frequency.

To avoid extremely large nodes dominating the graph,
exponential scaling is used:

    size = min_size + exp(scale_factor * log(occurrence))


This keeps the visualization balanced.


--------------------------------------------------------------------
8. Visualization
--------------------------------------------------------------------

The graph is visualized using PyVis, which provides an
interactive HTML network visualization.

Nodes show:

    decision predicate
    occurrence count
    mean reward
    probability

Edges show:

    transition probability
    transition reward
    transition frequency


--------------------------------------------------------------------
9. Relationship to SymbXRL Pipeline
--------------------------------------------------------------------

This module is the final stage of the explainability pipeline.

Pipeline:

    Environment (Massive MIMO Scheduler)
            ↓
    RL Agent (SAC / DQN)
            ↓
    State + Action Logs
            ↓
    Symbolizer
            ↓
    Symbolic Predicates
            ↓
    DecisionGraph   ← THIS MODULE
            ↓
    Explainable Policy Graph


--------------------------------------------------------------------
10. Example Graph Interpretation
--------------------------------------------------------------------

Node:

    inc(G1, Q3, 50)

Meaning:

    The scheduler increases allocation to group 1 when KPI
    is in quartile 3 and schedules 50% of users.


Edge:

    inc(G1,Q3,50) → const(G2,Q2,50)

Probability:

    0.73

Interpretation:

    After increasing scheduling in group 1, the agent
    usually moves to maintaining scheduling in group 2.


###########################################################################

This graph provides a symbolic and interpretable representation
of the policy learned by the RL agent.

Advantages:

    • Explainable AI for wireless scheduling
    • Policy analysis
    • Strategy discovery
    • Debugging RL behavior
    • Action steering

###########################################################################
    

    
====================================================================
"""

# ========================= IMPORTS =========================
# pandas → used for handling the symbolic dataframe
# numpy → used for mathematical operations
# networkx → used to build and manipulate the decision graph
# pyvis → used to visualize the graph interactively in HTML

import pandas as pd
import numpy as np
import networkx as nx
import pyvis.network as network


# ========================= DECISION GRAPH CLASS =========================
# This class builds a directed graph representing the agent's symbolic decisions
# and transitions between those decisions.

class DecisionGraph:

    def __init__(self, column_name) -> None:
        """
        Constructor for the DecisionGraph class.

        Parameters
        ----------
        column_name : str
            Name of the column in the symbolic dataframe that contains
            the decision predicate (example: "decision").

        Example decision string:
            inc(G1, Q3, 50)
        """

        # Column name containing the decision predicate
        self.column_name = column_name

        # Optional storage of decision history (not heavily used here)
        self.decision_df = []

        # Directed graph object
        # DiGraph means edges have direction: decision_t → decision_t+1
        self.G = nx.DiGraph()

        # PyVis network object for visualization
        self.net = None

        # Store the previous decision to build transitions
        self.previous_decision = None

        return


    # ========================= UPDATE GRAPH =========================

    def update_graph(self, symbolic_form_df: pd.DataFrame):
        """
        Update the decision graph using a new symbolic decision.

        Parameters
        ----------
        symbolic_form_df : pandas.DataFrame

        Expected columns include:
            decision
            reward
            timestep

        This function performs three major tasks:

        1. Create/update node statistics
        2. Create/update edge transitions
        3. Update probabilities and visualization sizes
        """

        # Extract the current decision predicate
        current_decision = symbolic_form_df[self.column_name].iloc[0]

        # Extract the reward obtained at this timestep
        current_reward = symbolic_form_df['reward'].iloc[0]


        # ---------------- NODE CREATION ----------------
        # If the decision node does not exist, create it
        if current_decision not in self.G.nodes:

            self.G.add_node(
                current_decision,

                # title used by pyvis hover tooltip
                title=current_decision,

                # number of times this decision occurred
                occurrence=0,

                # sum of rewards obtained when taking this decision
                total_reward=0,

                # average reward of this decision
                mean_reward=0
            )


        # ---------------- NODE STATISTICS UPDATE ----------------

        # Increase occurrence count
        self.G.nodes[current_decision]['occurrence'] += 1

        # Add reward
        self.G.nodes[current_decision]['total_reward'] += current_reward

        # Update mean reward
        self.G.nodes[current_decision]['mean_reward'] = (
            self.G.nodes[current_decision]['total_reward']
            /
            self.G.nodes[current_decision]['occurrence']
        )


        # ---------------- EDGE TRANSITION UPDATE ----------------

        # If this is not the first decision in the episode
        if self.previous_decision is not None:

            # If transition already exists
            if self.G.has_edge(self.previous_decision, current_decision):

                # Update occurrence
                self.G[self.previous_decision][current_decision]['occurrence'] += 1

                # Update accumulated reward
                self.G[self.previous_decision][current_decision]['total_reward'] += current_reward

            else:

                # Create new edge
                self.G.add_edge(
                    self.previous_decision,
                    current_decision,

                    # number of times this transition occurred
                    occurrence=1,

                    # reward accumulated on this transition
                    total_reward=current_reward
                )


            # Update mean reward for the transition
            self.G[self.previous_decision][current_decision]['mean_reward'] = (

                self.G[self.previous_decision][current_decision]['total_reward']
                /
                self.G[self.previous_decision][current_decision]['occurrence']
            )


        # Save current decision as previous for next step
        self.previous_decision = current_decision

        # Save reward as well
        self.previous_reward = current_reward


        # Update graph statistics
        self._update_probabilities_and_sizes()

        return



    # ========================= UPDATE PROBABILITIES =========================

    def _update_probabilities_and_sizes(self):
        """
        Update node probabilities, node sizes, and edge probabilities.

        Node probability represents:
            P(decision)

        Edge probability represents:
            P(next_decision | current_decision)
        """

        # -------- TOTAL NODE OCCURRENCES --------

        total_node_occurrence = sum(
            nx.get_node_attributes(self.G, 'occurrence').values()
        )


        # -------- NODE PROBABILITY --------

        node_probabilities = {}
        node_sizes = {}

        for node, data in self.G.nodes(data=True):

            # probability = occurrences / total decisions
            node_probabilities[node] = (
                data['occurrence'] / total_node_occurrence
            )

            # Node size scaling for visualization
            min_size = 10
            scale_factor = 0.5

            node_sizes[node] = (
                min_size
                +
                np.exp(scale_factor * np.log1p(data['occurrence'] - 1))
            )


        # Store attributes in graph
        nx.set_node_attributes(self.G, node_probabilities, 'probability')
        nx.set_node_attributes(self.G, node_sizes, 'size')


        # -------- EDGE TRANSITION PROBABILITIES --------

        edge_probabilities = {}

        for u, v, data in self.G.edges(data=True):

            # number of transitions starting from node u
            total_transitions_from_u = sum(
                self.G[u][nbr]['occurrence']
                for nbr in self.G.successors(u)
            )

            # conditional probability
            edge_probabilities[(u, v)] = (
                data['occurrence'] / total_transitions_from_u
                if total_transitions_from_u > 0
                else 0
            )


        # Store edge probability
        nx.set_edge_attributes(self.G, edge_probabilities, 'probability')



    # ========================= BUILD GRAPH =========================

    def build_graph(self):
        """
        Build the PyVis visualization of the decision graph.

        Returns
        -------
        self.net : pyvis.network.Network
        """

        # Update graph statistics before visualization
        self._update_probabilities_and_sizes()


        # Create PyVis network object
        self.net = network.Network(
            height="1500px",
            width="100%",
            bgcolor="#222222",
            font_color="white",
            directed=True,
            notebook=True,
            filter_menu=True,
            select_menu=True,
            cdn_resources="in_line"
        )


        # Convert networkX graph → PyVis graph
        self.net.from_nx(self.G)


        # -------- NODE TOOLTIP --------

        for node in self.net.nodes:

            occurrence = self.G.nodes[node['id']]['occurrence']
            probability = round(
                100 * self.G.nodes[node['id']]['probability'],
                1
            )

            mean_reward = self.G.nodes[node['id']]['mean_reward']

            node['title'] = (
                f"Node: {node['id']} \n"
                f"Occurrence: {occurrence} \n"
                f"Mean Reward: {mean_reward:.2f} \n"
                f"Probability: {probability}%"
            )


        # -------- EDGE TOOLTIP --------

        for edge in self.net.edges:

            u, v = edge['from'], edge['to']

            occurrence = self.G[u][v]['occurrence']
            probability = round(
                100 * self.G[u][v]['probability'],
                1
            )

            mean_reward = self.G[u][v]['mean_reward']

            edge['title'] = (
                f"Edge from {u} to {v} \n"
                f"Occurrence: {occurrence} \n"
                f"Mean Reward: {mean_reward:.2f} \n"
                f"Probability: {probability}%"
            )


        # -------- GRAPH SIZE INFO --------

        num_nodes = self.G.number_of_nodes()
        num_edges = self.G.number_of_edges()

        size_text = (
            f"Number of Nodes: {num_nodes}<br>"
            f"Number of Edges: {num_edges}"
        )


        # Add text node with graph statistics
        self.net.add_node(
            "size_info",
            label=size_text,
            shape="text",
            x='-95%',
            y=0,
            physics=False
        )


        # Layout physics model
        self.net.barnes_hut(overlap=1)

        # Add UI controls
        self.net.show_buttons(filter_=['physics'])

        return



    # ========================= GET GRAPH =========================

    def get_graph(self, mode="all"):
        """
        Retrieve graph objects.

        mode options
        ------------
        all       → return both networkX and pyvis graphs
        networkX  → return only networkX graph
        pyvis     → return only pyvis graph
        """

        if mode == "all":
            return self.G, self.net

        if mode == "networkX":
            return self.G

        if mode == "pyvis":
            return self.net