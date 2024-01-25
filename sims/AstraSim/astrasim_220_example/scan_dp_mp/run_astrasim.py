import os, subprocess, multiprocessing

def run_command(command, cwd=None):
    print(command)
    subprocess.run(command, shell=True, cwd=cwd)
    return True

def list_workloads(root):
    files = os.listdir(root)
    filtered = list()
    for file in files:
        if file.endswith(".0.eg"):
            filtered.append(os.path.join(root, file[:-5]))
    return filtered

def run_astrasim(workload_path):
    root = os.path.split(os.path.abspath(__file__))[0]
    astrasim_root = os.path.join(root, "../../astrasim_archgym_public/astra-sim")
    astrasim_bin = os.path.join(astrasim_root, "build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware")
    system = os.path.join(root, "..", "system.json")
    network = os.path.join(root, "..", "network.yml")
    memory = os.path.join(root, "..", "memory.json")
    stdout = os.path.join(root, os.path.split(workload_path)[1]+".stdout")
    cmd = f"{astrasim_bin} --system-configuration={system} --workload-configuration={workload_path} --network-configuration={network} --remote-memory-configuration={memory} > {stdout}"
    run_command(cmd)
    
if __name__ == '__main__':
    et_root = os.path.join(
        os.path.split(os.path.abspath(__file__))[0],
        "./workload/eg"
    )
    design_space = list_workloads(et_root)
    with multiprocessing.Pool() as pool:
        pool.map(run_astrasim, design_space)
