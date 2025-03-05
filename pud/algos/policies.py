import time
import torch
import logging
import numpy as np
import networkx as nx
from scipy.stats import truncnorm
from scipy.sparse.csgraph import floyd_warshall

from pud.mapf.cbs import CBSSolver


class BasePolicy:
    def __init__(self, agent, **kwargs):
        self.agent = agent
        self.constraints = (
            None if not hasattr(agent, "constraints") else agent.constraints
        )

    def select_action(self, state):
        return self.agent.select_action(state)


class GaussianPolicy(BasePolicy):
    def __init__(self, agent, noise_scale=1.0):
        super().__init__(agent)
        self.noise_scale = noise_scale

    def select_action(self, state):
        action = super().select_action(state)
        action += np.random.normal(
            0, self.agent.max_action * self.noise_scale, size=action.shape
        ).astype(action.dtype)
        action = action.clip(-self.agent.max_action, self.agent.max_action)
        return action


class VectorGaussianPolicy(BasePolicy):
    def __init__(self, agent, noise_scale=1.0):
        super().__init__(agent)
        self.noise_scale = noise_scale

    def select_action(self, state):
        action = super().select_action(state)
        noise_dist = truncnorm(
            -self.agent.max_action - action,
            self.agent.max_action - action,
            loc=np.zeros_like(action),
            scale=self.agent.max_action * self.noise_scale,
        )
        noise = noise_dist.rvs(size=action.shape).astype(action.dtype)

        action = action + noise
        action = action.clip(-self.agent.max_action, self.agent.max_action)
        return action


class SearchPolicy(BasePolicy):
    def __init__(
        self,
        agent,
        rb_vec,
        pdist=None,
        aggregate="min",
        open_loop=False,
        max_search_steps=7,
        no_waypoint_hopping=False,
        weighted_path_planning=False,
        waypoint_consistency_cutoff=5.0,
        **kwargs,
    ):
        """
        Args:
            rb_vec: a replay buffer vector storing the observations that will be used as nodes in the graph
            pdist: a matrix of dimension len(rb_vec) x len(rb_vec) where pdist[i,j] gives the distance going from
                   rb_vec[i] to rb_vec[j]
            max_search_steps: (int)
            open_loop: if True, only performs search once at the beginning of the episode
            weighted_path_planning: whether or not to use edge weights when planning a shortest path from start to goal
            no_waypoint_hopping: if True, will not try to proceed to goal until all waypoints have been reached
        """
        super().__init__(agent=agent, **kwargs)
        self.rb_vec = rb_vec
        self.pdist = pdist

        self.aggregate = aggregate
        self.open_loop = open_loop
        self.max_search_steps = max_search_steps
        self.weighted_path_planning = weighted_path_planning

        self.cleanup = False
        self.attempt_cutoff = 3 * max_search_steps
        self.no_waypoint_hopping = no_waypoint_hopping
        self.waypoint_consistency_cutoff = waypoint_consistency_cutoff

        self.build_rb_graph(self.rb_vec)
        if not self.open_loop:
            pdist2 = self.agent.get_pairwise_dist(
                self.rb_vec,
                aggregate=self.aggregate,
                max_search_steps=self.max_search_steps,
                masked=True,
            )
            self.rb_distances = floyd_warshall(pdist2, directed=True)
        self.reset_stats()

    def __str__(self):
        s = f"{self.__class__.__name__} (|V|={self.g.number_of_nodes()}, |E|={self.g.number_of_edges()})"
        return s

    def reset_stats(self):
        self.stats = dict(
            localization_fails=0,
            path_planning_fails=0,
            graph_search_time=0.0,
            path_planning_attempts=0,
        )

    def get_stats(self):
        return self.stats

    def set_cleanup(
        self, cleanup
    ):  # If True, will prune edges when fail to reach waypoint after `attempt_cutoff`
        self.cleanup = cleanup

    def build_rb_graph(self, rb_vec):
        g = nx.DiGraph()
        assert self.pdist is not None
        pdist_combined = np.max(self.pdist, axis=0)

        for i, _ in enumerate(rb_vec):
            for j, _ in enumerate(rb_vec):
                length = pdist_combined[i, j]
                if length < self.max_search_steps:
                    g.add_edge(i, j, weight=length)
        self.g = g

    def get_pairwise_dist_to_rb(self, state, masked=True):
        start_to_rb_dist = self.agent.get_pairwise_dist(
            [state["observation"]],
            self.rb_vec,
            aggregate=self.aggregate,
            max_search_steps=self.max_search_steps,
            masked=masked,
        )
        rb_to_goal_dist = self.agent.get_pairwise_dist(
            self.rb_vec,
            [state["goal"]],
            aggregate=self.aggregate,
            max_search_steps=self.max_search_steps,
            masked=masked,
        )
        return start_to_rb_dist, rb_to_goal_dist

    def get_closest_waypoint(self, state):
        """
        For closed loop replanning at each step. Uses the precomputed distances
        `rb_distances` b/w states in `rb_vec`
        """
        obs_to_rb_dist, rb_to_goal_dist = self.get_pairwise_dist_to_rb(state)
        # (B x A), (A x B)

        # The search_dist tensor should be (B x A x A)
        search_dist = sum(
            [
                np.expand_dims(obs_to_rb_dist, 2),
                np.expand_dims(self.rb_distances, 0),
                np.expand_dims(np.transpose(rb_to_goal_dist), 1),
            ]
        )  # Elementwise sum

        # We assume a batch size of 1.
        min_search_dist = np.min(search_dist)
        waypoint_index = np.argmin(np.min(search_dist, axis=2), axis=1)[0]
        waypoint = self.rb_vec[waypoint_index]

        return waypoint, waypoint_index, min_search_dist

    def construct_planning_graph(
        self, state, planning_graph=None, start_id="start", goal_id="goal"
    ):
        start_to_rb_dist, rb_to_goal_dist = self.get_pairwise_dist_to_rb(state)
        if planning_graph is None:
            planning_graph = self.g.copy()

        for i, (dist_from_start, dist_to_goal) in enumerate(
            zip(start_to_rb_dist.flatten(), rb_to_goal_dist.flatten())
        ):
            if dist_from_start < self.max_search_steps:
                planning_graph.add_edge(start_id, i, weight=dist_from_start)
            if dist_to_goal < self.max_search_steps:
                planning_graph.add_edge(i, goal_id, weight=dist_to_goal)

        if not np.any(start_to_rb_dist < self.max_search_steps) or not np.any(
            rb_to_goal_dist < self.max_search_steps
        ):
            self.stats["localization_fails"] += 1

        return planning_graph

    def get_path(self, state):
        g2 = self.construct_planning_graph(state)
        try:
            self.stats["path_planning_attempts"] += 1
            graph_search_start = time.perf_counter()

            if self.weighted_path_planning:
                path = nx.shortest_path(
                    g2, source="start", target="goal", weight="weight"
                )
            else:
                path = nx.shortest_path(g2, source="start", target="goal")
        except Exception as e:
            self.stats["path_planning_fails"] += 1
            raise RuntimeError(
                f"Failed to find path in graph (|V|={g2.number_of_nodes()}, |E|={g2.number_of_edges()})"
            ) from e
        finally:
            graph_search_end = time.perf_counter()
            self.stats["graph_search_time"] += graph_search_end - graph_search_start

        edge_lengths = []
        for i, j in zip(path[:-1], path[1:]):
            edge_lengths.append(g2[i][j]["weight"])

        waypoint_to_goal_dist = np.cumsum(edge_lengths[::-1])[::-1]  # Reverse CumSum
        waypoint_indices = list(path)[1:-1]
        return waypoint_indices, waypoint_to_goal_dist[1:]

    def initialize_path(self, state):
        self.waypoint_indices, self.waypoint_to_goal_dist_vec = self.get_path(state)
        self.waypoint_counter = 0
        self.waypoint_attempts = 0
        self.reached_final_waypoint = False

    def get_current_waypoint(self):
        waypoint_index = self.waypoint_indices[self.waypoint_counter]
        waypoint = self.rb_vec[waypoint_index]
        return waypoint, waypoint_index

    def get_waypoints(self):
        waypoints = [self.rb_vec[i] for i in self.waypoint_indices]
        return waypoints

    def reached_waypoint(self, dist_to_waypoint, state, waypoint_index):
        return dist_to_waypoint < 2.0

    def select_action(self, state):
        goal = state["goal"]
        dist_to_goal = self.agent.get_dist_to_goal({k: [v] for k, v in state.items()})[
            0
        ]

        if self.open_loop or self.cleanup:
            if state.get("first_step", False):
                self.initialize_path(state)

            if self.cleanup and (self.waypoint_attempts >= self.attempt_cutoff):
                # Prune edge and replan
                if self.waypoint_counter != 0 and not self.reached_final_waypoint:
                    src_node = self.waypoint_indices[self.waypoint_counter - 1]
                    dest_node = self.waypoint_indices[self.waypoint_counter]
                    self.g.remove_edge(src_node, dest_node)
                self.initialize_path(state)

            waypoint, waypoint_index = self.get_current_waypoint()
            state["goal"] = waypoint
            dist_to_waypoint = self.agent.get_dist_to_goal(
                {k: [v] for k, v in state.items()}
            )[0]

            if self.reached_waypoint(dist_to_waypoint, state, waypoint_index):
                if not self.reached_final_waypoint:
                    self.waypoint_attempts = 0

                self.waypoint_counter += 1
                if self.waypoint_counter >= len(self.waypoint_indices):
                    self.reached_final_waypoint = True
                    self.waypoint_counter = len(self.waypoint_indices) - 1

                waypoint, waypoint_index = self.get_current_waypoint()
                state["goal"] = waypoint
                dist_to_waypoint = self.agent.get_dist_to_goal(
                    {k: [v] for k, v in state.items()}
                )[0]

            dist_to_goal_via_waypoint = (
                dist_to_waypoint + self.waypoint_to_goal_dist_vec[self.waypoint_counter]
            )
        else:
            # Closed loop, replan waypoint at each step
            waypoint, waypoint_index, dist_to_goal_via_waypoint = (
                self.get_closest_waypoint(state)
            )

        if (
            (self.no_waypoint_hopping and not self.reached_final_waypoint)
            or (dist_to_goal_via_waypoint < dist_to_goal)
            or (dist_to_goal > self.max_search_steps)
        ):
            state["goal"] = waypoint
            if self.open_loop:
                self.waypoint_attempts += 1
        else:
            state["goal"] = goal
        return super().select_action(state)


class ConstrainedSearchPolicy(SearchPolicy):
    def __init__(
        self,
        agent,
        rb_vec,
        pdist=None,
        pcost=None,
        ckpts=None,
        open_loop=False,
        max_search_steps=7,
        max_cost_limit=1.0,
        dist_aggregate="min",
        cost_aggregate="max",
        no_waypoint_hopping=False,
        weighted_path_planning=False,
        waypoint_consistency_cutoff=5.0,
        **kwargs,
    ):
        """
        Args:
            pcost: a matrix of dimension len(rb_vec) x len(rb_vec) where pcost[i,j] gives the cost of going from
                   rb_vec[i] to rb_vec[j]
            max_cost_limit: (int)
            cost_aggregate: (str) aggregation function to use when computing cost from ensembles
        """
        self.pcost = pcost
        self.cost_aggregate = cost_aggregate
        self.max_cost_limit = max_cost_limit

        assert ckpts is not None
        self.ckpts = ckpts

        assert hasattr(agent, "constraints") and agent.constraints is not None
        self.constraints = agent.constraints

        super().__init__(
            agent=agent,
            pdist=pdist,
            rb_vec=rb_vec,
            open_loop=open_loop,
            aggregate=dist_aggregate,
            max_search_steps=max_search_steps,
            no_waypoint_hopping=no_waypoint_hopping,
            weighted_path_planning=weighted_path_planning,
            waypoint_consistency_cutoff=waypoint_consistency_cutoff,
            **kwargs,
        )

    def build_rb_graph(self, rb_vec):
        g = nx.DiGraph()
        assert self.pdist is not None
        assert self.pcost is not None
        pdist_combined = np.max(self.pdist, axis=0)
        pcost_combined = np.max(self.pcost, axis=0)

        for i, s_i in enumerate(rb_vec):
            for j, s_j in enumerate(rb_vec):
                cost = pcost_combined[i, j]
                length = pdist_combined[i, j]
                if length < self.max_search_steps and cost < self.max_cost_limit:
                    g.add_edge(i, j, weight=length)
        self.g = g

    def get_pairwise_cost_to_rb(self, state, masked=True):
        self.agent.load_state_dict(
            torch.load(self.ckpts["unconstrained"], map_location="cuda:0")
        )
        start_to_rb_cost = self.agent.get_pairwise_cost(
            [state["observation"]],
            self.rb_vec,
            aggregate=self.cost_aggregate,
        )
        rb_to_goal_cost = self.agent.get_pairwise_cost(
            self.rb_vec,
            [state["goal"]],
            aggregate=self.cost_aggregate,
        )
        self.agent.load_state_dict(
            torch.load(self.ckpts["constrained"], map_location="cuda:0")
        )
        return start_to_rb_cost, rb_to_goal_cost

    def get_closest_waypoint(self, state):
        """
        For closed loop replanning at each step. Uses the precomputed distances
        `rb_distances` b/w states in `rb_vec`
        """
        obs_to_rb_cost, _ = self.get_pairwise_cost_to_rb(state)
        obs_to_rb_dist, rb_to_goal_dist = self.get_pairwise_dist_to_rb(state)
        # (B x A), (A x B)

        # The search_dist tensor should be (B x A x A)
        search_dist = sum(
            [
                np.expand_dims(obs_to_rb_dist, 2),
                np.expand_dims(self.rb_distances, 0),
                np.expand_dims(np.transpose(rb_to_goal_dist), 1),
            ]
        )  # Elementwise sum

        # We assume a batch size of 1.
        not_safe = True
        while not_safe:
            min_search_dist = np.min(search_dist)
            waypoint_index = np.argmin(np.min(search_dist, axis=2), axis=1)[0]
            waypoint = self.rb_vec[waypoint_index]

            if obs_to_rb_cost[0, waypoint_index] < self.max_cost_limit:
                not_safe = False

        return waypoint, waypoint_index, min_search_dist

    def construct_planning_graph(
        self, state, planning_graph=None, start_id="start", goal_id="goal"
    ):
        start_to_rb_dist, rb_to_goal_dist = self.get_pairwise_dist_to_rb(state)
        start_to_rb_cost, rb_to_goal_cost = self.get_pairwise_cost_to_rb(state)
        if planning_graph is None:
            planning_graph = self.g.copy()

        for i, (from_start, to_goal) in enumerate(
            zip(
                zip(start_to_rb_dist.flatten(), start_to_rb_cost.flatten()),
                zip(rb_to_goal_dist.flatten(), rb_to_goal_cost.flatten()),
            )
        ):
            dist_from_start, cost_from_start = from_start
            dist_to_goal, cost_to_goal = to_goal
            if (
                dist_from_start < self.max_search_steps
                and cost_from_start < self.max_cost_limit
            ):
                planning_graph.add_edge(start_id, i, weight=dist_from_start)
            if (
                dist_to_goal < self.max_search_steps
                and cost_to_goal < self.max_cost_limit
            ):
                planning_graph.add_edge(i, goal_id, weight=dist_to_goal)

        if (
            not np.any(start_to_rb_dist < self.max_search_steps)
            or not np.any(rb_to_goal_dist < self.max_search_steps)
            or not np.any(start_to_rb_cost < self.max_cost_limit)
            or not np.any(rb_to_goal_cost < self.max_cost_limit)
        ):
            self.stats["localization_fails"] += 1

        return planning_graph


class VisualSearchPolicy(SearchPolicy):
    def __init__(
        self,
        agent,
        rb_vec,
        pdist=None,
        aggregate="min",
        open_loop=False,
        max_search_steps=7,
        no_waypoint_hopping=False,
        weighted_path_planning=False,
        waypoint_consistency_cutoff=5.0,
        **kwargs,
    ):
        super().__init__(
            agent=agent,
            rb_vec=rb_vec,
            pdist=pdist,
            aggregate=aggregate,
            open_loop=open_loop,
            max_search_steps=max_search_steps,
            no_waypoint_hopping=no_waypoint_hopping,
            weighted_path_planning=weighted_path_planning,
            waypoint_consistency_cutoff=waypoint_consistency_cutoff,
            **kwargs,
        )

        assert isinstance(rb_vec, tuple)
        self.rb_vec_grid, self.rb_vec = rb_vec

        self.build_rb_graph(self.rb_vec_grid)
        if not self.open_loop:
            pdist2 = self.agent.get_pairwise_dist(
                self.rb_vec,
                aggregate=self.aggregate,
                max_search_steps=self.max_search_steps,
                masked=True,
            )
            self.rb_distances = floyd_warshall(pdist2, directed=True)

    def get_waypoints(self):
        waypoints = [
            (self.rb_vec_grid[i], self.rb_vec[i]) for i in self.waypoint_indices
        ]
        return waypoints

    def select_action(self, state):
        goal = state["goal"]
        grid_goal = state["grid"]["goal"]
        dist_to_goal = self.agent.get_dist_to_goal({k: [v] for k, v in state.items()})[
            0
        ]

        if self.open_loop or self.cleanup:
            if state.get("first_step", False):
                self.initialize_path(state)

            if self.cleanup and (self.waypoint_attempts >= self.attempt_cutoff):
                # Prune edge and replan
                if self.waypoint_counter != 0 and not self.reached_final_waypoint:
                    src_node = self.waypoint_indices[self.waypoint_counter - 1]
                    dest_node = self.waypoint_indices[self.waypoint_counter]
                    self.g.remove_edge(src_node, dest_node)
                self.initialize_path(state)

            waypoint, waypoint_index = self.get_current_waypoint()
            state["goal"] = waypoint
            state["grid"]["goal"] = self.rb_vec_grid[waypoint_index]
            dist_to_waypoint = self.agent.get_dist_to_goal(
                {k: [v] for k, v in state.items()}
            )[0]

            if self.reached_waypoint(dist_to_waypoint, state, waypoint_index):
                if not self.reached_final_waypoint:
                    self.waypoint_attempts = 0

                self.waypoint_counter += 1
                if self.waypoint_counter >= len(self.waypoint_indices):
                    self.reached_final_waypoint = True
                    self.waypoint_counter = len(self.waypoint_indices) - 1

                waypoint, waypoint_index = self.get_current_waypoint()
                state["goal"] = waypoint
                state["grid"]["goal"] = self.rb_vec_grid[waypoint_index]
                dist_to_waypoint = self.agent.get_dist_to_goal(
                    {k: [v] for k, v in state.items()}
                )[0]

            dist_to_goal_via_waypoint = (
                dist_to_waypoint + self.waypoint_to_goal_dist_vec[self.waypoint_counter]
            )
        else:
            # Closed loop, replan waypoint at each step
            waypoint, waypoint_index, dist_to_goal_via_waypoint = (
                self.get_closest_waypoint(state)
            )

        if (
            (self.no_waypoint_hopping and not self.reached_final_waypoint)
            or (dist_to_goal_via_waypoint < dist_to_goal)
            or (dist_to_goal > self.max_search_steps)
        ):
            state["goal"] = waypoint
            state["grid"]["goal"] = self.rb_vec_grid[waypoint_index]
            if self.open_loop:
                self.waypoint_attempts += 1
        else:
            state["goal"] = goal
            state["grid"]["goal"] = grid_goal
        return super(SearchPolicy, self).select_action(state)


class VisualConstrainedSearchPolicy(ConstrainedSearchPolicy, VisualSearchPolicy):
    def __init__(
        self,
        agent,
        rb_vec,
        pdist=None,
        pcost=None,
        ckpts=None,
        open_loop=False,
        max_search_steps=7,
        max_cost_limit=1.0,
        dist_aggregate="min",
        cost_aggregate="max",
        no_waypoint_hopping=False,
        weighted_path_planning=False,
        waypoint_consistency_cutoff=5.0,
        **kwargs,
    ):
        super().__init__(
            agent=agent,
            pdist=pdist,
            pcost=pcost,
            ckpts=ckpts,
            rb_vec=rb_vec,
            open_loop=open_loop,
            dist_aggregate=dist_aggregate,
            cost_aggregate=cost_aggregate,
            max_cost_limit=max_cost_limit,
            max_search_steps=max_search_steps,
            no_waypoint_hopping=no_waypoint_hopping,
            weighted_path_planning=weighted_path_planning,
            waypoint_consistency_cutoff=waypoint_consistency_cutoff,
            **kwargs,
        )


class MultiAgentSearchPolicy(SearchPolicy):
    def __init__(
        self,
        agent,
        rb_vec,
        n_agents,
        pdist=None,
        radius=0.1,
        aggregate="min",
        open_loop=False,
        max_search_steps=7,
        no_waypoint_hopping=False,
        weighted_path_planning=False,
        waypoint_consistency_cutoff=5.0,
        **kwargs,
    ):
        super().__init__(
            agent=agent,
            rb_vec=rb_vec,
            pdist=pdist,
            aggregate=aggregate,
            open_loop=open_loop,
            max_search_steps=max_search_steps,
            no_waypoint_hopping=no_waypoint_hopping,
            weighted_path_planning=weighted_path_planning,
            waypoint_consistency_cutoff=waypoint_consistency_cutoff,
            **kwargs,
        )
        self.radius = radius
        self.n_agents = n_agents

    def get_closest_waypoints(self, state):

        assert "composite_goals" in state.keys()
        assert len(state["composite_goals"]) == self.n_agents

        augmented_waypoints = []
        augmented_waypoint_indices = []
        augmented_min_search_dists = []

        for agent_id in range(self.n_agents):

            state_copy = state.copy()
            state_copy["goal"] = state["composite_goals"][agent_id]
            state_copy["observation"] = state["agent_observations"][agent_id]

            waypoint, waypoint_index, min_search_dist = self.get_closest_waypoint(
                state_copy
            )

            augmented_waypoint_indices.append(waypoint_index)

            if waypoint in augmented_waypoints:
                augmented_min_search_dists.append(0)
                augmented_waypoints.append(state["agent_waypoints"][agent_id])
            else:
                augmented_waypoints.append(waypoint)
                augmented_min_search_dists.append(min_search_dist)
                state["agent_waypoints"][agent_id] = waypoint
                state["agent_waypoints_visual"][agent_id] = waypoint
        return (
            augmented_waypoints,
            augmented_waypoint_indices,
            augmented_min_search_dists,
        )

    def construct_augmented_planning_graph(self, starts, goals):
        planning_graph = self.g.copy()
        logging.debug("Initial graph size = ", self.g.number_of_nodes())

        num_nodes = self.rb_vec.shape[0] - 1

        nodes_to_agent_maps = {}
        for agent_id, (start, goal) in enumerate(zip(starts, goals)):

            start_id = num_nodes + 1
            nodes_to_agent_maps["start" + str(agent_id)] = start_id

            goal_id = num_nodes + 2
            nodes_to_agent_maps["goal" + str(agent_id)] = goal_id

            planning_graph = self.construct_planning_graph(
                {"observation": start, "goal": goal}, planning_graph, start_id, goal_id  # type: ignore
            )

            assert planning_graph.has_node(start_id), "Start node not added to graph"
            assert planning_graph.has_node(goal_id), "Goal node not added to graph"
            num_nodes += 2

        logging.debug("Final graph size = ", planning_graph.number_of_nodes())
        return planning_graph, nodes_to_agent_maps

    def initialize_paths(self, rb_vec, starts, goals):
        graph, nodes_to_agents_maps = self.construct_augmented_planning_graph(
            starts, goals
        )

        goal_ids = [
            nodes_to_agents_maps["goal" + str(agent_id)]
            for agent_id in range(self.n_agents)
        ]
        start_ids = [
            nodes_to_agents_maps["start" + str(agent_id)]
            for agent_id in range(self.n_agents)
        ]

        augmented_wps = rb_vec.copy()
        for _, (start, goal) in enumerate(zip(starts, goals)):
            augmented_wps = np.vstack([augmented_wps, start, goal])

        cbs_solver = CBSSolver(
            graph,
            augmented_wps,
            start_ids,
            goal_ids,
            weighted=self.weighted_path_planning,
            collision_radius=self.radius,
        )
        paths = cbs_solver.find_paths()

        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            print("Printing the paths")
            for agent_id, path in enumerate(paths):
                print("--" * 10)
                print("Path for agent ", agent_id)
                for vertex in path:
                    if vertex in start_ids or vertex in goal_ids:
                        (
                            print("Start: ", starts[agent_id])
                            if vertex in start_ids
                            else print("Goal: ", goals[agent_id])
                        )
                    else:
                        print("Vertex: ", rb_vec[vertex])

        self.augmented_waypoint_indices, self.augmented_waypoint_to_goal_dist_vec = (
            [],
            [],
        )
        for path in paths:
            edge_lengths = []
            for i, j in zip(path[:-1], path[1:]):
                edge_lengths.append(graph[i][j]["weight"])
            waypoint_to_goal_dist = np.cumsum(edge_lengths[::-1])[::-1]
            waypoint_indices = list(path)[1:-1]
            self.augmented_waypoint_indices.append(waypoint_indices)
            self.augmented_waypoint_to_goal_dist_vec.append(waypoint_to_goal_dist[1:])

        self.augmented_waypoint_stays = np.zeros(self.n_agents, dtype=bool)
        self.augmented_waypoint_counters = np.zeros(self.n_agents, dtype=int)
        self.augmented_waypoint_attempts = np.zeros(self.n_agents, dtype=int)
        self.augmented_reached_final_waypoints = np.zeros(self.n_agents, dtype=bool)

        self.goals = goals
        self.starts = starts
        self.goal_ids = goal_ids
        self.start_ids = start_ids
        self.nodes_to_agents_maps = nodes_to_agents_maps

    def get_current_waypoints(self):
        augmented_waypoints = []
        augmented_wp_indices = []

        for agent_id in range(self.n_agents):
            waypoint_index = self.augmented_waypoint_indices[agent_id][
                self.augmented_waypoint_counters[agent_id]
            ]
            if waypoint_index > self.rb_vec.shape[0]:
                waypoint = (
                    self.starts[agent_id]
                    if waypoint_index == self.start_ids[agent_id]
                    else self.goals[agent_id]
                )
            else:
                waypoint = self.rb_vec[waypoint_index]

            if agent_id > 0:
                waypoint_qvals = self.agent.get_pairwise_dist(
                    [waypoint], augmented_waypoints, aggregate=None
                )
                if np.any(waypoint_qvals < self.waypoint_consistency_cutoff):
                    waypoint_index = self.augmented_waypoint_indices[agent_id][
                        self.augmented_waypoint_counters[agent_id]
                    ]
                    waypoint = self.rb_vec[waypoint_index]

            augmented_waypoints.append(waypoint)
            augmented_wp_indices.append(waypoint_index)
        return augmented_waypoints, augmented_wp_indices

    def get_augmented_waypoints(self):
        augmented_waypoints = []
        for agent_id in range(self.n_agents):
            augmented_waypoints.append(
                [self.rb_vec[j] for j in self.augmented_waypoint_indices[agent_id]]
            )
        return augmented_waypoints

    def select_action(self, state):
        assert "composite_goals" in state.keys()
        assert len(state["composite_goals"]) == self.n_agents

        # Composite start and goals are the grid representation of the state
        composite_goals = state["composite_goals"]
        dist_to_composite_goals = []
        for agent_id in range(self.n_agents):
            c_goal = composite_goals[agent_id]
            state_copy = state.copy()
            state_copy["goal"] = c_goal
            state_copy["observation"] = state["agent_observations"][agent_id]
            dist_to_composite_goals.append(
                self.agent.get_dist_to_goal({k: [v] for k, v in state_copy.items()})[0]
            )

        if self.open_loop or self.cleanup:

            if state.get("first_step", False):

                self.initialize_paths(
                    self.rb_vec, state["composite_starts"], state["composite_goals"]
                )

            if self.cleanup:
                for idx, (
                    waypoint_counter,
                    waypoint_attempts,
                    reached_final_waypoint,
                ) in enumerate(
                    zip(
                        self.augmented_waypoint_counters,
                        self.augmented_waypoint_attempts,
                        self.augmented_reached_final_waypoints,
                    )
                ):
                    if (
                        waypoint_attempts >= self.attempt_cutoff
                        and waypoint_counter != 0
                        and not reached_final_waypoint
                    ):
                        src_node = self.augmented_waypoint_indices[idx][
                            waypoint_counter - 1
                        ]
                        dest_node = self.augmented_waypoint_indices[idx][
                            waypoint_counter
                        ]
                        self.g.remove_edge(src_node, dest_node)

                self.initialize_paths(
                    self.rb_vec, state["composite_starts"], state["composite_goals"]
                )

            waypoints, waypoint_indices = self.get_current_waypoints()

            dist_to_goal_via_waypoints = []
            for agent_id in range(self.n_agents):
                state_copy = state.copy()
                state_copy["goal"] = waypoints[agent_id]
                state_copy["observation"] = state["agent_observations"][agent_id]
                dist_to_waypoint = self.agent.get_dist_to_goal(
                    {k: [v] for k, v in state_copy.items()}
                )[0]

                if self.reached_waypoint(
                    dist_to_waypoint, state_copy, waypoint_indices[agent_id]
                ):
                    if not self.augmented_reached_final_waypoints[agent_id]:
                        self.augmented_waypoint_attempts[agent_id] = 0

                    self.augmented_waypoint_counters[agent_id] += 1
                    if self.augmented_waypoint_counters[agent_id] >= len(
                        self.augmented_waypoint_indices[agent_id]
                    ):
                        self.augmented_reached_final_waypoints[agent_id] = True
                        self.augmented_waypoint_counters[agent_id] = (
                            len(self.augmented_waypoint_indices[agent_id]) - 1
                        )

                    waypoints, waypoint_indices = self.get_current_waypoints()
                    state_copy["goal"] = waypoints[agent_id]
                    dist_to_waypoint = self.agent.get_dist_to_goal(
                        {k: [v] for k, v in state_copy.items()}
                    )[0]

                dist_to_goal_via_waypoint = (
                    dist_to_waypoint
                    + self.augmented_waypoint_to_goal_dist_vec[agent_id][
                        self.augmented_waypoint_counters[agent_id]
                    ]
                )
                dist_to_goal_via_waypoints.append(dist_to_goal_via_waypoint)
        else:
            # Closed loop, replan waypoint at each step
            waypoints, waypoint_indices, dist_to_goal_via_waypoints = (
                self.get_closest_waypoints(state)
            )

        # These variables are used by the "get_trajectories" function to update agent's goals with intermediate
        # waypoints
        agent_goals = []
        agent_actions = []
        for agent_id in range(self.n_agents):
            state_copy = state.copy()
            if (
                (
                    self.no_waypoint_hopping
                    and not self.augmented_reached_final_waypoints[agent_id]
                )
                or (
                    dist_to_goal_via_waypoints[agent_id]
                    < dist_to_composite_goals[agent_id]
                )
                or (dist_to_composite_goals[agent_id] > self.max_search_steps)
            ):
                state_copy["goal"] = waypoints[agent_id]
                if self.open_loop:
                    self.augmented_waypoint_attempts[agent_id] += 1
            else:
                state_copy["goal"] = composite_goals[agent_id]

            agent_goals.append(state_copy["goal"])
            state_copy["observation"] = state["agent_observations"][agent_id]

            agent_action = super(SearchPolicy, self).select_action(state_copy)
            agent_actions.append(agent_action)

        return agent_actions, agent_goals


class ConstrainedMultiAgentSearchPolicy(
    ConstrainedSearchPolicy, MultiAgentSearchPolicy
):
    def __init__(
        self,
        agent,
        rb_vec,
        n_agents,
        ckpts=None,
        pdist=None,
        pcost=None,
        radius=0.1,
        open_loop=False,
        max_search_steps=7,
        max_cost_limit=1.0,
        dist_aggregate="min",
        cost_aggregate="max",
        no_waypoint_hopping=False,
        weighted_path_planning=False,
        waypoint_consistency_cutoff=5.0,
        **kwargs,
    ):
        super().__init__(
            agent=agent,
            rb_vec=rb_vec,
            pdist=pdist,
            pcost=pcost,
            ckpts=ckpts,
            radius=radius,
            n_agents=n_agents,
            open_loop=open_loop,
            dist_aggregate=dist_aggregate,
            cost_aggregate=cost_aggregate,
            max_cost_limit=max_cost_limit,
            max_search_steps=max_search_steps,
            no_waypoint_hopping=no_waypoint_hopping,
            weighted_path_planning=weighted_path_planning,
            waypoint_consistency_cutoff=waypoint_consistency_cutoff,
            **kwargs,
        )


class VisualMultiAgentSearchPolicy(MultiAgentSearchPolicy):
    def __init__(
        self,
        agent,
        rb_vec,
        n_agents,
        pdist=None,
        radius=0.1,
        aggregate="min",
        open_loop=False,
        max_search_steps=7,
        no_waypoint_hopping=False,
        weighted_path_planning=False,
        waypoint_consistency_cutoff=5.0,
        **kwargs,
    ):
        super().__init__(
            pdist=pdist,
            agent=agent,
            rb_vec=rb_vec,
            radius=radius,
            n_agents=n_agents,
            aggregate=aggregate,
            open_loop=open_loop,
            max_search_steps=max_search_steps,
            no_waypoint_hopping=no_waypoint_hopping,
            weighted_path_planning=weighted_path_planning,
            waypoint_consistency_cutoff=waypoint_consistency_cutoff,
            **kwargs,
        )

        assert isinstance(rb_vec, tuple)
        self.rb_vec_grid, self.rb_vec = rb_vec

        self.build_rb_graph(self.rb_vec_grid)
        if not self.open_loop:
            pdist2 = self.agent.get_pairwise_dist(
                self.rb_vec,
                aggregate=self.aggregate,
                max_search_steps=self.max_search_steps,
                masked=True,
            )
            self.rb_distances = floyd_warshall(pdist2, directed=True)

    def get_closest_waypoints(self, state):

        augmented_waypoints = []
        augmented_waypoint_indices = []
        augmented_visual_waypoints = []
        augmented_min_search_dists = []

        for agent_id in range(self.n_agents):

            state_copy = state.copy()

            state_copy["goal"] = state["composite_goals"][agent_id][1]
            state_copy["grid"]["goal"] = state["composite_goals"][agent_id][0]
            state_copy["observation"] = state["agent_observations_visual"][agent_id][1]
            state_copy["grid"]["observation"] = state["agent_observations"][agent_id][0]

            waypoint, waypoint_index, min_search_dist = self.get_closest_waypoint(
                state_copy
            )
            waypoint_grid = self.rb_vec_grid[waypoint_index]

            augmented_waypoint_indices.append(waypoint_index)
            if waypoint_grid in augmented_waypoints:
                augmented_min_search_dists.append(0)
                augmented_waypoints.append(state["agent_waypoints"][agent_id][0])
                augmented_visual_waypoints.append(state["agent_waypoints"][agent_id][1])
            else:
                augmented_waypoints.append(waypoint_grid)
                augmented_visual_waypoints.append(waypoint)
                augmented_min_search_dists.append(min_search_dist)
                state["agent_waypoints"][agent_id] = (waypoint_grid, waypoint)

        return (
            (
                augmented_waypoints,
                augmented_visual_waypoints,
            ),
            augmented_waypoint_indices,
            augmented_min_search_dists,
        )

    def initialize_paths(self, rb_vec, starts, goals):

        goals_grid = [goal[0] for goal in goals]
        goals = [goal[1] for goal in goals]
        starts_grid = [start[0] for start in starts]
        starts = [start[1] for start in starts]
        graph, nodes_to_agents_maps = self.construct_augmented_planning_graph(
            starts, goals
        )

        goal_ids = [
            nodes_to_agents_maps["goal" + str(agent_id)]
            for agent_id in range(self.n_agents)
        ]
        start_ids = [
            nodes_to_agents_maps["start" + str(agent_id)]
            for agent_id in range(self.n_agents)
        ]

        augmented_wps = rb_vec.copy()
        for _, (start, goal) in enumerate(zip(starts_grid, goals_grid)):
            augmented_wps = np.vstack([augmented_wps, start, goal])

        cbs_solver = CBSSolver(
            graph,
            augmented_wps,
            start_ids,
            goal_ids,
            weighted=self.weighted_path_planning,
            collision_radius=self.radius,
        )
        paths = cbs_solver.find_paths()

        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            print("Printing the paths")
            for a_id, path in enumerate(paths):
                print("--" * 10)
                print("Path for agent ", a_id)
                for vertex in path:
                    if vertex in start_ids or vertex in goal_ids:
                        if vertex in start_ids:
                            print("Start: ", starts_grid[a_id])
                        else:
                            print("Goal: ", goals_grid[a_id])
                    else:
                        print("Vertex: ", rb_vec[vertex])

        self.augmented_waypoint_indices, self.augmented_waypoint_to_goal_dist_vec = (
            [],
            [],
        )

        for path in paths:
            edge_lengths = []
            for i, j in zip(path[:-1], path[1:]):
                edge_lengths.append(graph[i][j]["weight"])
            waypoint_to_goal_dist = np.cumsum(edge_lengths[::-1])[::-1]
            waypoint_indices = list(path)[1:-1]
            self.augmented_waypoint_indices.append(waypoint_indices)
            self.augmented_waypoint_to_goal_dist_vec.append(waypoint_to_goal_dist[1:])

        self.augmented_waypoint_stays = np.zeros(len(starts), dtype=bool)
        self.augmented_waypoint_counters = np.zeros(len(starts), dtype=int)
        self.augmented_waypoint_attempts = np.zeros(len(starts), dtype=int)
        self.augmented_reached_final_waypoints = np.zeros(len(starts), dtype=bool)

        self.goals = goals
        self.starts = starts
        self.goal_ids = goal_ids
        self.start_ids = start_ids
        self.nodes_to_agents_maps = nodes_to_agents_maps

    def get_current_waypoints(self):

        augmented_waypoints = []
        augmented_wp_indices = []
        augmented_grid_waypoints = []

        for agent_id in range(self.n_agents):
            waypoint_index = self.augmented_waypoint_indices[agent_id][
                self.augmented_waypoint_counters[agent_id]
            ]
            if waypoint_index > self.rb_vec.shape[0]:
                waypoint = (
                    self.starts[agent_id]
                    if waypoint_index == self.start_ids[agent_id]
                    else self.goals[agent_id]
                )
            else:
                waypoint = self.rb_vec[waypoint_index]

            if agent_id > 0:
                waypoint_qvals = self.agent.get_pairwise_dist(
                    [waypoint], augmented_waypoints, aggregate=None
                )
                if np.any(waypoint_qvals < self.waypoint_consistency_cutoff):
                    waypoint_index = self.augmented_waypoint_indices[agent_id][
                        self.augmented_waypoint_counters[agent_id]
                    ]
                    waypoint = self.rb_vec[waypoint_index]

            augmented_waypoints.append(waypoint)
            augmented_wp_indices.append(waypoint_index)
            augmented_grid_waypoints.append(self.rb_vec_grid[waypoint_index])
        return (augmented_grid_waypoints, augmented_waypoints), augmented_wp_indices

    def get_augmented_waypoints(self):
        augmented_waypoints = []
        for agent_id in range(self.n_agents):
            augmented_waypoints.append(
                [
                    (self.rb_vec_grid[j], self.rb_vec[j])
                    for j in self.augmented_waypoint_indices[agent_id]
                ]
            )
        return augmented_waypoints

    def select_action(self, state):
        assert "composite_goals" in state.keys()
        assert len(state["composite_goals"]) == self.n_agents

        # Composite start and goals are the grid representation of the state
        composite_goals = state["composite_goals"]
        composite_goals_grid = [goal[0] for goal in composite_goals]
        composite_goals = [goal[1] for goal in composite_goals]

        composite_starts_grid = [start[0] for start in state["composite_starts"]]
        dist_to_composite_goals = []
        for agent_id in range(self.n_agents):

            state_copy = state.copy()
            state_copy["goal"] = composite_goals[agent_id]
            state_copy["grid"]["goal"] = composite_goals_grid[agent_id]
            state_copy["observation"] = state["agent_observations"][agent_id][1]
            state_copy["grid"]["observation"] = state["agent_observations"][agent_id][0]
            dist_to_composite_goals.append(
                self.agent.get_dist_to_goal({k: [v] for k, v in state_copy.items()})[0]
            )

        if self.open_loop or self.cleanup:

            if state.get("first_step", False):

                logging.debug("Composite starts: ", composite_starts_grid)
                logging.debug("Composite goals: ", composite_goals_grid)
                self.initialize_paths(
                    self.rb_vec_grid,
                    state["composite_starts"],
                    state["composite_goals"],
                )

            if self.cleanup:
                for idx, (
                    waypoint_counter,
                    waypoint_attempts,
                    reached_final_waypoint,
                ) in enumerate(
                    zip(
                        self.augmented_waypoint_counters,
                        self.augmented_waypoint_attempts,
                        self.augmented_reached_final_waypoints,
                    )
                ):
                    if (
                        waypoint_attempts >= self.attempt_cutoff
                        and waypoint_counter != 0
                        and not reached_final_waypoint
                    ):
                        src_node = self.augmented_waypoint_indices[idx][
                            waypoint_counter - 1
                        ]
                        dest_node = self.augmented_waypoint_indices[idx][
                            waypoint_counter
                        ]
                        self.g.remove_edge(src_node, dest_node)

                self.initialize_paths(
                    self.rb_vec_grid,
                    state["composite_starts"],
                    state["composite_goals"],
                )

            waypoints, waypoint_indices = self.get_current_waypoints()
            waypoints_grid, waypoints = waypoints

            dist_to_goal_via_waypoints = []
            for agent_id in range(self.n_agents):
                state_copy = state.copy()

                state_copy["goal"] = waypoints[agent_id]
                state_copy["grid"]["goal"] = waypoints_grid[agent_id]
                state_copy["observation"] = state["agent_observations"][agent_id][1]
                state_copy["grid"]["observation"] = state["agent_observations"][
                    agent_id
                ][0]
                dist_to_waypoint = self.agent.get_dist_to_goal(
                    {k: [v] for k, v in state_copy.items()}
                )[0]

                if self.reached_waypoint(
                    dist_to_waypoint, state_copy, waypoint_indices[agent_id]
                ):
                    if not self.augmented_reached_final_waypoints[agent_id]:
                        self.augmented_waypoint_attempts[agent_id] = 0

                    self.augmented_waypoint_counters[agent_id] += 1
                    if self.augmented_waypoint_counters[agent_id] >= len(
                        self.augmented_waypoint_indices[agent_id]
                    ):
                        self.augmented_reached_final_waypoints[agent_id] = True
                        self.augmented_waypoint_counters[agent_id] = (
                            len(self.augmented_waypoint_indices[agent_id]) - 1
                        )

                    waypoints, waypoint_indices = self.get_current_waypoints()
                    waypoints_grid, waypoints = waypoints

                    state_copy["goal"] = waypoints[agent_id]
                    state_copy["grid"]["goal"] = waypoints_grid[agent_id]
                    dist_to_waypoint = self.agent.get_dist_to_goal(
                        {k: [v] for k, v in state_copy.items()}
                    )[0]

                dist_to_goal_via_waypoint = (
                    dist_to_waypoint
                    + self.augmented_waypoint_to_goal_dist_vec[agent_id][
                        self.augmented_waypoint_counters[agent_id]
                    ]
                )
                dist_to_goal_via_waypoints.append(dist_to_goal_via_waypoint)
        else:
            # Closed loop, replan waypoint at each step
            waypoints, waypoint_indices, dist_to_goal_via_waypoints = (
                self.get_closest_waypoints(state)
            )
            waypoints_grid, waypoints = waypoints

        # These variables are used by the "get_trajectories" function to update agent's goals with intermediate
        # waypoints
        agent_goals = []
        agent_actions = []
        for agent_id in range(self.n_agents):
            state_copy = state.copy()
            if (
                (
                    self.no_waypoint_hopping
                    and not self.augmented_reached_final_waypoints[agent_id]
                )
                or (
                    dist_to_goal_via_waypoints[agent_id]
                    < dist_to_composite_goals[agent_id]
                )
                or (dist_to_composite_goals[agent_id] > self.max_search_steps)
            ):
                state_copy["goal"] = waypoints[agent_id]
                state_copy["grid"]["goal"] = waypoints_grid[agent_id]
                if self.open_loop:
                    self.augmented_waypoint_attempts[agent_id] += 1
            else:
                state_copy["goal"] = composite_goals[agent_id]
                state_copy["grid"]["goal"] = composite_goals_grid[agent_id]

            agent_goals.append((state_copy["grid"]["goal"], state_copy["goal"]))
            state_copy["observation"] = state["agent_observations"][agent_id][1]
            state_copy["grid"]["observation"] = state["agent_observations"][agent_id][0]

            agent_action = super(SearchPolicy, self).select_action(state_copy)
            agent_actions.append(agent_action)

        return agent_actions, agent_goals


class VisualConstrainedMultiAgentSearchPolicy(
    ConstrainedSearchPolicy, VisualMultiAgentSearchPolicy
):
    def __init__(
        self,
        agent,
        rb_vec,
        n_agents,
        pdist=None,
        pcost=None,
        ckpts=None,
        radius=0.1,
        open_loop=False,
        max_search_steps=7,
        max_cost_limit=1.0,
        dist_aggregate="min",
        cost_aggregate="max",
        no_waypoint_hopping=False,
        weighted_path_planning=False,
        waypoint_consistency_cutoff=5.0,
        **kwargs,
    ):
        super().__init__(
            agent=agent,
            rb_vec=rb_vec,
            pdist=pdist,
            pcost=pcost,
            ckpts=ckpts,
            radius=radius,
            n_agents=n_agents,
            open_loop=open_loop,
            dist_aggregate=dist_aggregate,
            cost_aggregate=cost_aggregate,
            max_cost_limit=max_cost_limit,
            max_search_steps=max_search_steps,
            no_waypoint_hopping=no_waypoint_hopping,
            weighted_path_planning=weighted_path_planning,
            waypoint_consistency_cutoff=waypoint_consistency_cutoff,
            **kwargs,
        )
