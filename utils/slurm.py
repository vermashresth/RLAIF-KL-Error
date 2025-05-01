import subprocess
from utils import CONFIGS

DEVICE_CONFIGS = CONFIGS.devices.devices


def get_idle_slurm_nodes_info():
    # Run the sinfo command to get idle nodes, number of nodes, and their GRES
    cmd = 'sinfo -t idle -h -o "%N %D %G"'
    output = subprocess.check_output(cmd, shell=True, universal_newlines=True)
    # Initialize a dictionary to store the GRES types and their corresponding node info
    gres_info = {}
    # Parse the output and populate the dictionary
    for line in output.strip().split("\n"):
        _, node_count, gres_str = line.split(maxsplit=2)
        node_count = int(node_count)
        if gres_str != "(null)":
            gres_entries = gres_str.split(",")
            for gres_entry in gres_entries:
                gres_type = gres_entry.split(":", maxsplit=1)[1]
                if gres_type not in gres_info:
                    gres_info[gres_type] = 0
                gres_info[gres_type] += node_count
    return gres_info


def find_available_nodes_with_gres(idle_nodes_info, gres):
    if gres == "(null)":
        return 0
    gpu_type, min_count = gres.split(":")
    min_count = int(min_count)
    total_available_nodes = 0
    for gres_type, node_count in idle_nodes_info.items():
        if gres_type.startswith(gpu_type):
            _, available_count = gres_type.split(":")
            available_count = int(available_count)
            if available_count >= min_count:
                total_available_nodes += node_count
    return total_available_nodes


def has_running_empty_gres():
    command = "squeue --me --state=RUNNING --format '%b'"
    output = subprocess.check_output(command, shell=True, universal_newlines=True)
    output_lines = output.strip().split("\n")
    for line in output_lines:
        if line.strip() == "N/A":
            return True
    return False


def get_pending_nodes_gres():
    command = "squeue --me --state=PENDING --format='%b'"
    output = subprocess.check_output(command, shell=True, universal_newlines=True)
    gres_list = []
    for line in output.strip().split("\n"):
        if line.startswith("gres/gpu:"):
            gres = line.replace("gres/gpu:", "").strip()
            gres_list.append(gres)
    return gres_list


def find_next_request_gres(pipeline):
    gres_list = DEVICE_CONFIGS["slurm"][pipeline]
    idle_nodes_info = get_idle_slurm_nodes_info()
    gres_with_count = [
        (gres, find_available_nodes_with_gres(idle_nodes_info, gres))
        for gres in gres_list
    ]
    sorted_gres_list = [
        gres for gres, _ in sorted(gres_with_count, key=lambda x: x[1], reverse=True)
    ]
    pending_gres_list = get_pending_nodes_gres()
    running_empty_gres_list = ["(null)"] if has_running_empty_gres() else []
    return next(
        (
            gres
            for gres in sorted_gres_list
            if gres not in (pending_gres_list + running_empty_gres_list)
        ),
        None,
    )
