import numpy as np

class ToyTaskEnvironment:
    def __init__(self, num_tasks=5, max_priority=10, prob_job=0.5, max_job_length=10):
        """
        Initialize the task management environment.
        
        Args:
            num_tasks      (int):   Number of tasks (nodes) to manage.
            max_priority   (int):   Maximum priority.
            prob_job       (float): Probability of a job occuring in a given time step.
            max_job_length (int):   Maximum length of job in time steps.
        """
        self.num_tasks = num_tasks
        self.max_priority = max_priority
        self.prob_job = prob_job
        self.max_job_length = max_job_length

        # Initialize available resources
        self.reset()

    def reset(self):
        """
        Reset the environment to its initial state.
        
        Returns:
            state (np.array): Node feature matrix (num_tasks x 3)
        """
        # Each task feature: [Arrival Time, Priority, Runtime]
        self.tasks = np.zeros(shape=(self.num_tasks, 3))
        self.tasks[:, 0] = np.cumsum(np.random.geometric(self.prob_job, size=self.num_tasks))   # Arrival Time 
        self.tasks[:, 1] = np.random.randint(low=1, high=self.max_priority, size=self.num_tasks, dtype="int16") # Priority
        self.tasks[:, 2] = np.random.randint(low=1, high=self.max_job_length, size=self.num_tasks)  # Runtime

        # Task status (0 = not scheduled, 1 = scheduled, 2 = finished)
        self.task_status = np.zeros(self.num_tasks, dtype=int)

        # Task completion time (used to track when the task finishes)
        self.task_completion_time = np.zeros(self.num_tasks)

        # Current timestep
        self.current_time = 0

        return self.tasks.copy()

    def step(self, actions):
        """
        Take an action to schedule a task.

        Args:
            action (int): Index of actions for each task.

        Returns:
            state (np.array): New node features.
            rewards (float): Reward for the action.
            done (bool): Whether the episode has finished.
            info (dict): Extra information (optional).
        """
        rewards = np.zeros(self.num_tasks)
        for n in range(self.num_tasks):
        
            task = self.tasks[n]
            memory_required, cpu_required, arrival_time, runtime = task
            
            action = actions[n]
            # reward = 0.0
            # Attempt to schedule task if it has arrived and it's not already scheduled
            if action == 1 and self.current_time >= arrival_time and self.task_status[n] == 0:
                    # Check if enough resources
                    if self.available_memory >= memory_required and self.available_cpu >= cpu_required:
                        # Allocate resources
                        self.available_memory -= memory_required
                        self.available_cpu -= cpu_required
                        self.task_status[n] = 1  # Mark task as scheduled
                        self.task_completion_time[n] = self.current_time + runtime  # Set completion time
        
                        rewards[n] = 1.0  # Positive reward for successful scheduling
                    else:
                        rewards[n] = -1.0  # Not enough resources (bad scheduling)
            if self.task_status[n] == 1 and self.current_time >= self.task_completion_time[n]:
                # If the task is scheduled and has completed its runtime, free the resources
                self.available_memory += memory_required
                self.available_cpu += cpu_required
                self.task_status[n] = 2  # Mark task as finished
                rewards[n] = 2.0  # Positive reward for completing a task
    
            else:
                rewards[n] = -0.5  # Invalid action: task not ready, already scheduled or already completed
                
        # Advance time (simple rule: +1 per step)
        self.current_time += 1.0
    
        # Check if all tasks are scheduled
        done = np.all(self.task_status == 2)

        return self.tasks.copy(), rewards, done

    def get_action_mask(self):
        """
        Generate an action mask indicating which tasks are available to schedule,
        prioritizing tasks with earlier arrival times and efficient resource usage.

        Returns:
            np.array: 1 if task can be scheduled now, 0 otherwise.
        """
        mask = np.zeros(self.num_tasks, dtype=int)  # Start with a mask of all zeros
        for idx, (memory_required, cpu_required, arrival_time, runtime) in enumerate(self.tasks):
            if self.task_status[idx] == 0 and self.current_time >= arrival_time:
                # Task has arrived and is not scheduled
                if self.available_memory >= memory_required and self.available_cpu >= cpu_required:
                    mask[idx] = 1  # Mark task as available to schedule
        return mask
    
    
    def get_adjacency_matrix(self, memory_threshold=0.5, cpu_threshold=0.5, arrival_time_threshold=1.0):
        """
        Create an adjacency matrix based on resource overlap and arrival time.
    
        Args:
            memory_threshold (float): Threshold for memory overlap.
            cpu_threshold (float): Threshold for CPU overlap.
            arrival_time_threshold (float): Threshold for arrival time proximity.
    
        Returns:
            adj_matrix (np.array): Adjacency matrix representing connections between tasks.
        """
        # Extract memory, CPU, and arrival time
        memory_required = self.tasks[:, 0]
        cpu_required = self.tasks[:, 1]
        arrival_time = self.tasks[:, 2]
    
        adj_matrix = np.zeros((self.num_tasks, self.num_tasks))  # Initialize the adjacency matrix with zeros
    
        for i in range(self.num_tasks):
            for j in range(i+1, self.num_tasks):  # Only check each pair once
                # Calculate normalized overlap for memory and CPU
                memory_overlap = min(memory_required[i], memory_required[j]) / max(memory_required[i], memory_required[j])
                cpu_overlap = min(cpu_required[i], cpu_required[j]) / max(cpu_required[i], cpu_required[j])
    
                # Check if memory and CPU overlap exceed the thresholds
                if memory_overlap > memory_threshold and cpu_overlap > cpu_threshold:
                    adj_matrix[i, j] = adj_matrix[j, i] = 1  # Connect tasks i and j
    
                # Check if arrival times are close enough based on the arrival_time_threshold
                if abs(arrival_time[i] - arrival_time[j]) <= arrival_time_threshold:
                    adj_matrix[i, j] = adj_matrix[j, i] = 1  # Connect tasks if arrival times are close

        return adj_matrix

    def render(self):
        """
        Print the environment status.
        """
        print(f"Current Time: {self.current_time}")
        print(f"Available Memory: {self.available_memory:.2f}, Available CPU: {self.available_cpu:.2f}")
        print(f"Task Status: {self.task_status}")

# --- Regression Script ---
if __name__ == "__main__":
    env = ToyTaskEnvironment()
    state = env.reset()
    A = env.get_adjacency_matrix()

    print("Initial state:")
    print(state)
    
    print('Adjacency matrix:')
    print(A)

    for step in range(10):
        action_mask = env.get_action_mask()
        available_actions = np.where(action_mask == 1)[0]

        if len(available_actions) > 0:
            action = np.random.choice(available_actions)
        else:
            action = np.random.randint(0, env.num_tasks)  # Random action (even if invalid)

        next_state, reward, done, _ = env.step(action)

        env.render()
        print(f"Action taken: {action}, Reward: {reward}\n")

        if done:
            print("All tasks scheduled!")
            break
