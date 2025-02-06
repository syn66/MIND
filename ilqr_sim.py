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

from matplotlib import pyplot as plt

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

if __name__ == '__main__':
    config = TrajTreeCfg()
    ilqr = iLQR(get_dynamic_model(config.dt, 2.5))

    cost_tree = Tree()
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    cost_tree.add_node(Node(-1, None, x0))

    duration = 50
    prob = 1.0
    target_vel = 10.0

    w_des_state = 0.0 * np.eye(6)
    w_des_state[0,0] = 1000.0
    w_des_state[1,1] = 1000.0
    w_des_state[2, 2] = 10.0  # weight on velocity
    w_des_state[4, 4] = 1.  # weight on acceleration
    w_des_state[5, 5] = 10.0  # weight on steering angle

    w_state_con = np.zeros((6,6))
    w_state_con[2, 2] = 50.0  # weight on velocity
    w_state_con[4, 4] = 50.0  # weight on acceleration
    w_state_con[5, 5] = 500.0  # weight on steering angle

    state_upper_bound = np.array([100000.0, 100000.0, 8.0, 10.0, 4.0, 0.2])
    state_lower_bound = np.array([-100000.0, -100000.0, 0.0, -10.0, -6.0, -0.2])
    
    w_ctrl = 5.0 * np.eye(config.action_size)

    last_index = -1

    a = 0.01
    b = 0.05
    c = 0.1
    d = 0.0

    # 生成离散点
    rx = np.linspace(0, 5, 80)  # x 值范围从 -1 到 5，共生成 100 个点
    ry = a * rx**3 + b * rx**2 + c * rx + d  # 计算对应的 y 值

    for i in range(duration):
        if i % 2 == 1:
            continue
        cur_index = len(cost_tree.nodes) - 1
        state_pot = StatePotential(w_des_state * prob,
                                    np.array([rx[i], ry[i], target_vel, 0.0, 0.0, 0.0]))
        state_con = StateConstraint(w_state_con * prob,
                                    state_lower_bound,
                                    state_upper_bound)
        ctrl_pot = ControlPotential(w_ctrl * prob)

        cost_tree.add_node(Node(cur_index, last_index, [[state_pot, state_con], [ctrl_pot]]))
        last_index = cur_index

    cost_tree.print()

    cost_tree = TreeCost(cost_tree, 6, config.action_size)

    print("1111111111111111111")

    us_init = np.zeros((cost_tree.tree.size() - 1, config.action_size))
    xs, us = ilqr.fit(us_init, cost_tree)
    print(len(xs))

    # return traj tree
    traj_tree = Tree()
    for node in cost_tree.tree.nodes.values():
        if node.parent_key is None:
            traj_tree.add_node(Node(node.key, None, [node.data, np.zeros(config.action_size)]))
        else:
            traj_tree.add_node(Node(node.key, node.parent_key, [xs[node.key], us[node.key]]))

    x, y , _,_, _, _= zip(*xs)

    plt.scatter(x,y)
    plt.plot(rx,ry)
    plt.savefig('output.png')  # 保存为图像文件
