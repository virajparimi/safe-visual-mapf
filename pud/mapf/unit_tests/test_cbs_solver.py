import unittest
import networkx as nx

from pud.mapf.cbs import CBSSolver

"""
python pud/algos/unit_tests/test_cbs_solver.py TestCBSSolver.test_find_paths
"""


class TestCBSSolver(unittest.TestCase):
    def setUp(self):
        self.filename = "pud/algos/unit_tests/test_cbs_input.txt"

    def test_find_paths(self):
        f = open(
            self.filename,
            "r",
        )
        line = f.readline()
        rows, columns = [int(x) for x in line.split(" ")]
        rows = int(rows)
        columns = int(columns)

        G = nx.empty_graph(0, create_using=nx.DiGraph)
        graph_waypoints = []

        boolean_map = []
        for r in range(rows):
            line = f.readline()
            boolean_map.append([])
            for cell in line:
                if cell == "@":
                    boolean_map[-1].append(True)
                elif cell == ".":
                    boolean_map[-1].append(False)

        import numpy as np

        boolean_map = np.array(boolean_map)

        line = f.readline()
        num_agents = int(line)

        starts = []
        goals = []
        for a in range(num_agents):
            line = f.readline()
            start_x, start_y, goal_x, goal_y = [int(x) for x in line.split(" ")]
            starts.append((start_x, start_y))
            goals.append((goal_x, goal_y))

            graph_waypoints.append([start_x, start_y])
            graph_waypoints.append([goal_x, goal_y])

        f.close()

        for node in range(boolean_map.shape[0] * boolean_map.shape[1]):
            node_x, node_y = node // boolean_map.shape[1], node % boolean_map.shape[1]
            graph_waypoints.append([node_x, node_y])
            potential_neighbors = [
                (node_x - 1, node_y),
                (node_x + 1, node_y),
                (node_x, node_y - 1),
                (node_x, node_y + 1),
            ]

            for neighbor in potential_neighbors:
                if (
                    neighbor[0] >= 0
                    and neighbor[0] < boolean_map.shape[0]
                    and neighbor[1] >= 0
                    and neighbor[1] < boolean_map.shape[1]
                    and not boolean_map[neighbor[0], neighbor[1]]
                ):
                    G.add_edge(
                        node, neighbor[0] * boolean_map.shape[1] + neighbor[1], weight=1
                    )

        start_ids, goal_ids = [], []
        for start_node in starts:
            start_node = start_node[0] * boolean_map.shape[1] + start_node[1]
            start_ids.append(start_node)
        for goal_node in goals:
            goal_node = goal_node[0] * boolean_map.shape[1] + goal_node[1]
            goal_ids.append(goal_node)

        graph_waypoints = np.array(graph_waypoints)
        solver = CBSSolver(G, graph_waypoints, start_ids, goal_ids, seed=0)
        paths = solver.find_paths()
        print(paths)

        self.assertTrue(len(paths) == 5)
        self.assertTrue((paths[0] == [9, 17, 17, 16, 24]))
        self.assertTrue(
            (paths[1] == [62, 61, 53, 52, 44, 43, 42, 34, 26, 25, 17, 16, 8, 0])
        )
        self.assertTrue((paths[2] == [35, 27, 26, 25, 17, 9]))
        self.assertTrue((paths[3] == [0, 8, 16, 24, 25, 26, 27, 28, 36, 44]))
        self.assertTrue((paths[4] == [8, 16, 24, 32, 33, 34, 35, 36, 37, 38, 46]))


if __name__ == "__main__":
    unittest.main()
