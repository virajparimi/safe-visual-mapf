import heapq
import random
import logging
import numpy as np
import networkx as nx
from networkx import Graph
from numpy.typing import NDArray
from typing import List, Union, Dict

from pud.mapf.single_agent_planner import (
    a_star,
    compute_heuristics,
    compute_sum_of_costs,
)


def location_collision(path1: List[int], path2: List[int], timestep: int):

    if path1[timestep] == path2[timestep]:
        return [path1[timestep]], timestep, "vertex"
    if timestep < len(path1) - 1:
        if (
            path1[timestep] == path2[timestep + 1]
            and path1[timestep + 1] == path2[timestep]
        ):
            return ([path1[timestep], path1[timestep + 1]], timestep + 1, "edge")


def radius_collision(
    path1: List[int],
    path2: List[int],
    timestep: int,
    graph_waypoints: NDArray,
    radius: float = 0.1,
):
    if (
        np.linalg.norm(
            graph_waypoints[path1[timestep]] - graph_waypoints[path2[timestep]]
        )
        <= radius
    ):
        return [path1[timestep]], timestep, "vertex"
    if timestep < len(path1) - 1:
        if (
            np.linalg.norm(
                graph_waypoints[path1[timestep]] - graph_waypoints[path2[timestep + 1]]
            )
            <= radius
            and np.linalg.norm(
                graph_waypoints[path1[timestep + 1]] - graph_waypoints[path2[timestep]]
            )
            <= radius
        ):
            return (
                [path1[timestep], path1[timestep + 1]],
                timestep + 1,
                "edge",
            )


def detect_collision(
    pathA: List[int], pathB: List[int], graph_waypoints: NDArray, collision_radius=0.1
):

    path1 = pathA.copy()
    path2 = pathB.copy()
    if len(path1) >= len(path2):
        short_path = path2
        long_path = path1
    else:
        short_path = path1
        long_path = path2

    for _ in range(len(long_path) - len(short_path)):
        short_path.append(short_path[-1])

    for timestep in range(len(path1)):
        if collision_radius > 0:
            return radius_collision(
                path1, path2, timestep, graph_waypoints, collision_radius
            )
        else:
            return location_collision(path1, path2, timestep)

    return None


def detect_collisions(
    paths: List[List[int]], graph_waypoints: NDArray, collision_radius=0.1
) -> List[Dict]:
    agg_collisions = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            collisions = detect_collision(
                paths[i], paths[j], graph_waypoints, collision_radius
            )
            if collisions is not None:
                agg_collisions.append(
                    {
                        "agent_A": i,
                        "agent_B": j,
                        "location": collisions[0],
                        "timestep": collisions[1],
                        "type": collisions[2],
                    }
                )
    return agg_collisions


def standard_split(collision: Dict) -> List[Dict]:
    constraints = []

    if collision["type"] == "vertex":
        constraints.append(
            {
                "agent_id": collision["agent_A"],
                "location": collision["location"],
                "timestep": collision["timestep"],
                "positive": False,
                "final": False,
            }
        )
        constraints.append(
            {
                "agent_id": collision["agent_B"],
                "location": collision["location"],
                "timestep": collision["timestep"],
                "positive": False,
                "final": False,
            }
        )
    elif collision["type"] == "edge":
        constraints.append(
            {
                "agent_id": collision["agent_A"],
                "location": collision["location"],
                "timestep": collision["timestep"],
                "positive": False,
                "final": False,
            }
        )
        constraints.append(
            {
                "agent_id": collision["agent_B"],
                "location": list(reversed(collision["location"])),
                "timestep": collision["timestep"],
                "positive": False,
                "final": False,
            }
        )

    return constraints


def disjoint_split(collision: Dict) -> List[Dict]:
    agents = [collision["agent_A"], collision["agent_B"]]
    agent_choice = random.randint(0, 1)
    agent = agents[agent_choice]
    location = (
        collision["location"]
        if agent_choice == 0
        else list(reversed(collision["location"]))
    )
    return [
        {
            "agent_id": agent,
            "location": location,
            "timestep": collision["timestep"],
            "positive": True,
            "final": False,
        },
        {
            "agent_id": agent,
            "location": location,
            "timestep": collision["timestep"],
            "positive": False,
            "final": False,
        },
    ]


def to_inflate(
    constraint: Dict,
    graph: Graph,
    paths: List[List[int]],
    radius: int = 3,
):

    for idx, other_agent_path in enumerate(paths):
        if idx == constraint["agent_id"]:
            continue

        nearest_nodes = nx.ego_graph(
            graph, constraint["location"][0], radius=radius
        ).nodes
        for node in nearest_nodes:
            tpath = nx.shortest_path(
                graph, source=constraint["location"][0], target=node
            )
            for i in range(len(tpath) - 1):
                if tpath[i + 1] in other_agent_path[constraint["timestep"] :]:  # noqa
                    return True
    return False


class CBSSolver(object):

    def __init__(
        self,
        graph: Graph,
        graph_waypoints: NDArray,
        starts: List[int],
        goals: List[int],
        disjoint: bool = False,
        seed: Union[int, None] = None,
        weighted: bool = False,
        collision_radius=0.1,
        max_expanded=10000,
    ):

        if seed is not None:
            random.seed(seed)
        self.graph = graph
        self.goals = goals
        self.starts = starts
        self.weighted = weighted
        self.disjoint = disjoint
        self.num_agents = len(starts)
        self.max_expanded = max_expanded
        self.graph_waypoints = graph_waypoints
        self.collision_radius = collision_radius

        self.open_list = []
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(
                compute_heuristics(self.graph, goal, weighted=self.weighted)
            )

        self.num_expanded = 0
        self.num_generated = 0

    def find_paths(self) -> List[List[int]]:
        logging.debug("Finding paths using CBS Solver")
        root = {
            "cost": 0,
            "paths": [],
            "collisions": [],
            "constraints": [],
        }

        for i in range(self.num_agents):
            logging.debug("Computing paths for agent {}".format(i))
            agent_path = a_star(
                i,
                self.graph,
                self.starts[i],
                self.goals[i],
                self.heuristics[i],
                root["constraints"],
                weighted=self.weighted,
            )
            if agent_path is None:
                raise RuntimeError("No path found for agent {}".format(i))

            root["paths"].append(agent_path)

        root["cost"] = compute_sum_of_costs(
            root["paths"], self.graph, weighted=self.weighted
        )
        root["collisions"] = detect_collisions(
            root["paths"], self.graph_waypoints, self.collision_radius
        )

        logging.debug(root["collisions"])
        for collision in root["collisions"]:
            logging.debug(standard_split(collision))

        heapq.heappush(
            self.open_list,
            (root["cost"], len(root["collisions"]), self.num_generated, root),
        )
        logging.debug("Generated: ", self.num_generated)
        self.num_generated += 1

        while len(self.open_list) > 0 and self.num_expanded < self.max_expanded:
            id, current_node = heapq.heappop(self.open_list)[2:]
            logging.debug("Expanded: ", id)
            self.num_expanded += 1

            if len(current_node["collisions"]) == 0:
                return current_node["paths"]

            collision = random.choice(current_node["collisions"])
            constraints = (
                disjoint_split(collision)
                if self.disjoint
                else standard_split(collision)
            )

            for constraint in constraints:
                successor = {
                    "cost": 0,
                    "paths": current_node["paths"].copy(),
                    "collisions": [],
                    "constraints": [*current_node["constraints"], constraint],
                }

                # radius = 3
                # inflate = to_inflate(constraint, self.graph, successor["paths"], radius)
                # if inflate:
                #     print("Inflated!!")
                #     print("---" * 50)

                # update_h = self.heuristics[constraint["agent_id"]].copy()
                # if inflate:
                #     for key, value in update_h.items():
                #         if value < radius:
                #             update_h[key] = value + 10

                agent_path = a_star(
                    constraint["agent_id"],
                    self.graph,
                    self.starts[constraint["agent_id"]],
                    self.goals[constraint["agent_id"]],
                    (
                        self.heuristics[constraint["agent_id"]]
                        # if not inflate
                        # else update_h
                    ),
                    successor["constraints"],
                    weighted=self.weighted,
                )

                skip = False
                if agent_path is None:
                    raise RuntimeError(
                        "No path found for agent {}".format(constraint["agent_id"])
                    )
                else:
                    logging.debug(agent_path)
                    successor["paths"][constraint["agent_id"]] = agent_path
                    if constraint["positive"]:
                        violating_agents = []
                        if len(constraint["location"]) == 1:
                            for agent in range(self.num_agents):
                                if (
                                    constraint["location"][0]
                                    == successor["paths"][agent][constraint["timestep"]]
                                ):
                                    violating_agents.append(agent)
                        else:
                            for agent in range(self.num_agents):
                                if (
                                    constraint["location"]
                                    == [
                                        successor["paths"][agent][
                                            constraint["timestep"] - 1
                                        ],
                                        successor["paths"][agent][
                                            constraint["timestep"]
                                        ],
                                    ]
                                    or constraint["location"][0]
                                    == successor["paths"][agent][
                                        constraint["timestep"] - 1
                                    ]
                                    or constraint["location"][1]
                                    == successor["paths"][agent][constraint["timestep"]]
                                ):
                                    violating_agents.append(agent)

                        for agent in violating_agents:
                            constraint_copy = constraint.copy()
                            constraint_copy["agent_id"] = agent
                            constraint_copy["positive"] = False
                            successor["constraints"].append(constraint_copy)
                            agent_path = a_star(
                                agent,
                                self.graph,
                                self.starts[agent],
                                self.goals[agent],
                                self.heuristics[agent],
                                successor["constraints"],
                                weighted=self.weighted,
                            )

                            if agent_path is None:
                                skip = True
                                break
                            else:
                                logging.debug(agent_path)
                                successor["paths"][agent] = agent_path

                    if not skip:
                        successor["collisions"] = detect_collisions(
                            successor["paths"], self.graph_waypoints
                        )
                        successor["cost"] = compute_sum_of_costs(
                            successor["paths"], self.graph
                        )
                        heapq.heappush(
                            self.open_list,
                            (
                                successor["cost"],
                                len(successor["collisions"]),
                                self.num_generated,
                                successor,
                            ),
                        )
                        logging.debug("Generated: ", self.num_generated)
                        self.num_generated += 1

        raise RuntimeError("No solution found")
