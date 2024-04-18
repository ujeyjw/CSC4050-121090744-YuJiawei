import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
def generate_vehicle_descriptions(agent_vehicle, vehicles, proximity_threshold=50):
    """ Generate descriptions only for the agent and nearby vehicles.
    
    :param agent_vehicle: The agent vehicle to focus on.
    :param vehicles: List of all vehicles.
    :param proximity_threshold: Distance threshold to consider a vehicle "nearby" (in meters).
    :return: List of descriptions for the agent and nearby vehicles.
    """
    descriptions = []
    agent_position = np.array(agent_vehicle.position)

    # Add description for the agent vehicle
    descriptions.append(
        f"Agent Vehicle -- Lane: {agent_vehicle.lane_index[2] + 1}, "
        f"Position: ({agent_vehicle.position[0]:.2f}, {agent_vehicle.position[1]:.2f}), "
        f"Speed: {agent_vehicle.speed:.2f} km/h, "
        f"Direction: {np.rad2deg(agent_vehicle.heading):.2f} degrees, "
        f"Collision Status: {'Collided' if agent_vehicle.crashed else 'No collision'}"
    )

    # Check other vehicles if they are in proximity of the agent
    for vehicle in vehicles:
        if vehicle is not agent_vehicle:
            vehicle_position = np.array(vehicle.position)
            distance = np.linalg.norm(vehicle_position - agent_position)
            if distance <= proximity_threshold:
                descriptions.append(
                    f"Nearby Vehicle -- Lane: {vehicle.lane_index[2] + 1}, "
                    f"Position: ({vehicle.position[0]:.2f}, {vehicle.position[1]:.2f}), "
                    f"Speed: {vehicle.speed:.2f} m/s, "
                    f"Direction: {np.rad2deg(vehicle.heading):.2f} degrees, "
                    f"Collision Status: {'Collided' if vehicle.crashed else 'No collision'}, "
                    f"Distance from Agent: {distance:.2f} m"
                )

    return descriptions
def control_vehicle_and_record_motion(num_episodes=5, description_filename='vehicle_descriptions_standard.txt'):
    env = gym.make("highway-v0",render_mode="human")
    env.configure({
        "manual_control": True
    })
    with open(description_filename, 'w') as desc_file:
        for episode in range(num_episodes):
            obs, info = env.reset()
            done = truncated = False
            desc_file.write(f"Episode {episode+1}:\n")
            while not (done or truncated):
                obs, reward, done, truncated, info =env.step(env.step(env.action_space.sample()))
                frame = env.render()
                
                # Generate and record vehicle descriptions
                # breakpoint()
                vehicles = env.unwrapped.road.vehicles
                agent_vehicle = env.unwrapped.road.vehicles[0]
                descriptions = generate_vehicle_descriptions(agent_vehicle, vehicles)
                # descriptions = generate_vehicle_descriptions(vehicles)
                desc_file.write(f"Time: {env.unwrapped.time:.2f}s\n")
                for desc in descriptions:
                    desc_file.write(desc + '\n')
                desc_file.write('\n')
            desc_file.write('\n')  # Add a space between episodes for clarity
if __name__ == "__main__":
    control_vehicle_and_record_motion()
