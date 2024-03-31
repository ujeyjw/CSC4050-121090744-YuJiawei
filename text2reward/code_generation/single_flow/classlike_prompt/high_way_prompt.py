from langchain.prompts import PromptTemplate

classlike_prompt = """
You are an expert in autonomous driving, reinforcement learning and code generation.
We are going to use a simulated vehicle to complete given driving tasks. 

Now I want you to help me write a reward function of reinforcement learning.
Typically, the reward function of a manipulation task is consisted of these following parts (some part is optional, so only include it if really necessary):
1. progress quickly on the road;
2. avoid collisions.
3. difference between current state of the agent and its goal state
4. the distance to the leading vehicle in the same lane to maintain a safe distance
5. lane keeping, measured by the deviation from the center of the lane
6. regularization of the vehicle's actions to encourage smooth driving

...
{environment}

{action}

You are allowed to use any existing python package if applicable. But only use these packages when it's really necessary.

I want it to fulfill the following task: {instruction}
1. Please think step by step and tell me what does this task mean;
2. Then write a function that format as `def compute_dense_reward(self, action, obs) -> float` and returns the reward. Just the function body is fine.
3. Do not invent any variable or attribute that is not given.
4. When you writing code, you can also add some comments as your thought.
""".strip()


HIGHWAY_PROMPT = PromptTemplate(
    input_variables=["environment", "action", "instruction"],
    template=classlike_prompt
)