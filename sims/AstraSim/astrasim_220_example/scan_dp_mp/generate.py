import os, subprocess, multiprocessing

def run_command(command, cwd=None):
    print(command)
    subprocess.run(command, shell=True, cwd=cwd)
    return True

def get_design_space():
    num_npus = 64
    dp = {1, 2, 4, 8, 16, 32, 64}
    sharded = {True, False}
    
    design_space = list()

    for ddp in dp:
        for ssharded in sharded:
            mmp = num_npus // ddp 
            if mmp <= 0:
                continue
            design_space.append((num_npus, ddp, ssharded))
    return design_space

def generate_instance(design_point):
    name = "%d_%d_%d" % design_point
    cfg_root = os.path.join(
            os.path.split(
                os.path.abspath(__file__)
            )[0], "workload", "cfg"
        )
    eg_root = os.path.join(
            os.path.split(
                os.path.abspath(__file__)
            )[0], "workload", "eg"
        )
    os.makedirs(cfg_root, exist_ok=True)
    os.makedirs(eg_root, exist_ok=True)
    num_npus, dp, sharded = design_point

    workload_cfg_template = "{ \
        \"num_npus\": %d, \
        \"dp\": %d, \
        \"weight_sharded\": %d \
    }"

    # generate cfg file
    cfg_path = os.path.join(cfg_root, name+".json")
    with open(cfg_path, "w") as cfg_file:
        cfg_file.write(workload_cfg_template%design_point)

    # generate eg
    eg_path = os.path.join(eg_root, name+".%d.eg")
    cmd = f"python3 workload_cfg_to_et.py --workload_cfg={cfg_path} --workload_et={eg_path}"
    cwd = os.path.join(os.path.split(os.path.abspath(__file__))[0], "..")
    run_command(cmd, cwd)

if __name__ == '__main__':
    design_space = get_design_space()
    with multiprocessing.Pool() as pool:
        pool.map(generate_instance, design_space)
    
