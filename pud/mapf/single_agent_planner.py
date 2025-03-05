from __future__ import annotations
import time
import heapq
import logging
from networkx import Graph
from typing import Dict, List, Union


def compute_sum_of_costs(
    paths: List[List[int]], graph: Graph, weighted: bool = False
) -> float:
    """
    Compute the sum of costs of the paths
    """
    sum_of_costs = 0
    for path in paths:
        for i in range(len(path) - 1):
            sum_of_costs += graph[path[i]][path[i + 1]]["weight"] if weighted else 1
    return sum_of_costs


def compute_heuristics(
    graph: Graph, goal: int, weighted: bool = False
) -> Dict[int, float]:
    """
    Compute the heuristic for each node in the graph
    """
    open_list = []
    closed_list = {}
    heuristics = {}
    root = {"location": goal, "cost": 0}
    heapq.heappush(open_list, (root["cost"], goal, root))

    closed_list[goal] = root
    undirected_graph = graph.to_undirected()
    while len(open_list) != 0:
        cost, location, current_node = heapq.heappop(open_list)
        for neighbor in undirected_graph.neighbors(location):

            successor_cost = (
                cost + undirected_graph[location][neighbor]["weight"]
                if weighted
                else cost + 1
            )
            successor_location = neighbor
            successor = {"location": successor_location, "cost": successor_cost}

            if (
                neighbor in closed_list
                and closed_list[neighbor]["cost"] > successor_cost
            ):
                closed_list[neighbor] = successor
                heapq.heappush(
                    open_list, (successor_cost, successor_location, successor)
                )
            elif neighbor not in closed_list:
                closed_list[neighbor] = successor
                heapq.heappush(
                    open_list, (successor_cost, successor_location, successor)
                )

    for location, node in closed_list.items():
        heuristics[location] = node["cost"]
    return heuristics


def build_constraint_table(
    constraints: List[Dict], agent_id: int
) -> Dict[int, List[Dict]]:

    constraint_table = {}
    for constraint in constraints:
        if constraint["agent_id"] == agent_id:
            timestep = constraint["timestep"]
            if timestep not in constraint_table:
                constraint_table[timestep] = [constraint]
            else:
                constraint_table[timestep].append(constraint)
    return constraint_table


def is_constrained(
    current_location: int | str,
    next_location: int | str,
    timestep: int,
    constraint_table: Dict[int, List[Dict]],
    goal: bool = False,
):

    if not goal:
        if timestep in constraint_table:
            for constraint in constraint_table[timestep]:
                # if constraint["positive"]:
                if [next_location] == constraint["location"] or [
                    current_location,
                    next_location,
                ] == constraint["location"]:
                    return True
        else:
            flattened_constraints = []
            constraints = [
                constraint
                for timestep_idx, constraint in constraint_table.items()
                if timestep_idx < timestep
            ]
            for constraint in constraints:
                for c in constraint:
                    flattened_constraints.append(c)
            for constraint in flattened_constraints:
                if [next_location] == constraint["location"] and constraint["final"]:
                    return True
    else:
        flattened_constraints = []
        constraints = [
            constraint
            for timestep_idx, constraint in constraint_table.items()
            if timestep_idx > timestep
        ]
        for constraint in constraints:
            for c in constraint:
                flattened_constraints.append(c)
        for constraint in flattened_constraints:
            if [next_location] == constraint["location"]:
                return True

    return False


def extract_path(goal_node: Node) -> List[int]:
    path = []
    current_node = goal_node
    while current_node is not None:
        path.append(current_node.location)
        current_node = current_node.parent
    path.reverse()
    return path


class Node:
    def __init__(self, location, g_value, h_value, parent, timestep):
        self.parent = parent
        self.g_value = g_value
        self.h_value = h_value
        self.location = location
        self.timestep = timestep

    def __lt__(self, other):
        if self.g_value + self.h_value == other.g_value + other.h_value:
            if self.h_value == other.h_value:
                return self.timestep < other.timestep
            return self.h_value < other.h_value
        return self.g_value + self.h_value < other.g_value + other.h_value


def a_star(
    agent_id: int,
    graph: Graph,
    start: int,
    goal: int,
    heuristics: Dict[int, float],
    constraints,
    weighted: bool = False,
    max_iterations: int = 100000,
    max_time: int = 300,
) -> Union[List[int], None]:

    open_list = []
    closed_list = {}

    h_value = heuristics[start]
    constraint_table = build_constraint_table(constraints, agent_id)

    root = Node(start, 0, h_value, None, 0)
    heapq.heappush(
        open_list,
        (root.g_value + root.h_value, root.h_value, root.location, root),
    )

    closed_list[(root.location, root.timestep)] = root

    # Add self-loops
    for node in graph.nodes:
        graph.add_edge(node, node, weight=0)

    iterations = 0
    start_time = time.time()
    while len(open_list) != 0 and iterations < max_iterations:

        iterations += 1
        logging.debug(f"Size of open List: {len(open_list)}")
        current_node = heapq.heappop(open_list)[3]
        logging.debug(f"Current Node: {current_node.location}")
        logging.debug(f"Current Timestep: {current_node.timestep}")
        if current_node.location == goal and not is_constrained(
            goal, goal, current_node.timestep, constraint_table, goal=True
        ):
            return extract_path(current_node)

        for neighbor in graph.neighbors(current_node.location):
            successor_location = neighbor

            if successor_location == current_node.location:
                successor = Node(
                    successor_location,
                    current_node.g_value + 1,
                    current_node.h_value,
                    current_node,
                    current_node.timestep + 1,
                )
            else:
                successor_gadd = (
                    graph[current_node.location][neighbor]["weight"] if weighted else 1
                )
                successor = Node(
                    successor_location,
                    current_node.g_value + successor_gadd,
                    heuristics[neighbor],
                    current_node,
                    current_node.timestep + 1,
                )

            if is_constrained(
                current_node.location,
                successor.location,
                successor.timestep,
                constraint_table,
            ):
                continue

            if (successor.location, successor.timestep) in closed_list:
                existing_node = closed_list[(successor.location, successor.timestep)]
                if (
                    successor.g_value + successor.h_value
                    < existing_node.g_value + existing_node.h_value
                ):
                    logging.debug(f"Updating node {successor.location}")
                    closed_list[(successor.location, successor.timestep)] = successor
                    heapq.heappush(
                        open_list,
                        (
                            successor.g_value + successor.h_value,
                            successor.h_value,
                            successor.location,
                            successor,
                        ),
                    )
            else:
                logging.debug(f"Adding node {successor.location}")
                closed_list[(successor.location, successor.timestep)] = successor
                heapq.heappush(
                    open_list,
                    (
                        successor.g_value + successor.h_value,
                        successor.h_value,
                        successor.location,
                        successor,
                    ),
                )

        loop_time = time.time() - start_time
        if loop_time > max_time:
            logging.debug(f"Exceeded max time of {max_time}")
            return None

        # Wait Action
        # successor_location = current_node["location"]
        # successor = {
        #     "location": successor_location,
        #     "g_value": current_node["g_value"] + 1,
        #     "h_value": current_node["h_value"],
        #     "parent": current_node,
        #     "timestep": current_node["timestep"] + 1,
        # }

        # if is_constrained(
        #     current_node["location"],
        #     successor["location"],
        #     successor["timestep"],
        #     constraint_table,
        # ):
        #     continue

        # if (successor["location"], successor["timestep"]) in closed_list:
        #     existing_node = closed_list[(successor["location"], successor["timestep"])]
        #     if (
        #         successor["g_value"] + successor["h_value"]
        #         < existing_node["g_value"] + existing_node["h_value"]
        #     ):
        #         closed_list[(successor["location"], successor["timestep"])] = successor
        #         heapq.heappush(
        #             open_list,
        #             (
        #                 successor["g_value"] + successor["h_value"],
        #                 successor["h_value"],
        #                 successor["location"],
        #                 successor,
        #             ),
        #         )
        # else:
        #     closed_list[(successor["location"], successor["timestep"])] = successor
        #     heapq.heappush(
        #         open_list,
        #         (
        #             successor["g_value"] + successor["h_value"],
        #             successor["h_value"],
        #             successor["location"],
        #             successor,
        #         ),
        #     )

    return None
