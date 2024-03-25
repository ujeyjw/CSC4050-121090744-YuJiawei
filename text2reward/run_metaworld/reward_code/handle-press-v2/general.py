import numpy as np

def compute_dense_reward(self, action, obs):
    # Calculate distance from end effector to handle
    dist_to_handle = np.linalg.norm(self.robot.ee_position - self.obj1.position)

    # Calculate difference between handle's current and goal positions
    handle_goal_diff = np.linalg.norm(self.obj1.position - self.goal_position)

    # Regularization term (the smaller the action, the smaller the penalty)
    action_reg = np.sum(np.square(action))

    # Define weights for the components of the reward function
    w_dist = -1.0  # penalize distance to handle
    w_goal_diff = -1.0  # penalize difference from goal
    w_action_reg = -0.1  # penalize large actions

    # Calculate reward
    reward = w_dist * dist_to_handle + w_goal_diff * handle_goal_diff + w_action_reg * action_reg

    return reward