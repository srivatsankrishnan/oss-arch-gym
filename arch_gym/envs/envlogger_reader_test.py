from envlogger import reader

log_dir = "/n/holylabs/LABS/janapa_reddi_lab/Lab/skrishnan/workspace/arch-gym/sims/Sniper/envlogger"

# use astrasim_dir for AstraSim
random_walker = "/home/aadyap/Documents/oss-arch-gym/sims/customenv/random_walker_trajectories/num_steps_50_num_episodes_1"
# astrasim_dir = "../../sims/customen/random_walker_trajectories/latency/resnet18_num_steps_4_num_episodes_1"


with reader.Reader(
    data_directory = random_walker) as r:
    for episode in r.episodes:
        for step in episode:
            print(step)