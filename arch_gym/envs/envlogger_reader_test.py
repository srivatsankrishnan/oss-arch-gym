from envlogger import reader

log_dir = "/n/holylabs/LABS/janapa_reddi_lab/Lab/skrishnan/workspace/arch-gym/sims/Sniper/envlogger"

# use astrasim_dir for AstraSim
astrasim_dir = "../../sims/AstraSim/random_walker_trajectories/latency/resnet18_num_steps_4_num_episodes_1"

with reader.Reader(
    data_directory = astrasim_dir) as r:
    for episode in r.episodes:
        for step in episode:
            print(step)