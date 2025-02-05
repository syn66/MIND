from planners.mind.trajectory_tree import TrajectoryTreeOptimizer
from planners.mind.configs.planning.demo_1 import TrajTreeCfg
import numpy as np
import theano.tensor as T
from common.geometry import get_point_mean_distances
from planners.basic.tree import Tree, Node
from planners.ilqr.solver import iLQR
from planners.ilqr.dynamics import AutoDiffDynamics
from planners.ilqr.cost import TreeCost
from planners.ilqr.utils import gen_dist_field
from planners.ilqr.potential import ControlPotential, StatePotential, StateConstraint, PotentialField


def get_dynamic_model(dt, wb):
    x_inputs = [
        T.dscalar("x"),
        T.dscalar("y"),
        T.dscalar("v"),
        T.dscalar("q"),
        T.dscalar("a"),
        T.dscalar("theta"),
    ]

    u_inputs = [
        T.dscalar("da"),
        T.dscalar("dtheta"),
    ]

    f = T.stack([
        x_inputs[0] + x_inputs[2] * T.cos(x_inputs[3]) * dt,
        x_inputs[1] + x_inputs[2] * T.sin(x_inputs[3]) * dt,
        x_inputs[2] + x_inputs[4] * dt,
        x_inputs[3] + x_inputs[2] / wb * T.tan(x_inputs[5]) * dt,
        x_inputs[4] + u_inputs[0] * dt,
        x_inputs[5] + u_inputs[1] * dt,
    ])

    return AutoDiffDynamics(f, x_inputs, u_inputs)

if __name__ == '__name__':
    config = TrajTreeCfg()
    ilqr = iLQR(get_dynamic_model(config.dt, 2.5))

    cost_tree = TreeCost()
    us_init = False

    if us_init is None:
        us_init = np.zeros((cost_tree.tree.size() - 1, config.action_size))


    xs, us = ilqr.fit(us_init, cost_tree)

    # return traj tree
    traj_tree = Tree()
    for node in cost_tree.tree.nodes.values():
        if node.parent_key is None:
            traj_tree.add_node(Node(node.key, None, [node.data, np.zeros(config.action_size)]))
        else:
            traj_tree.add_node(Node(node.key, node.parent_key, [xs[node.key], us[node.key]]))

