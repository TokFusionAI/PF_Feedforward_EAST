# -*- coding: utf-8 -*-
import os
import numpy as np
from .compat import mdsConfig, eastAnalyzer, getData
from .compat import screen_print

def check_connections(shot: int = 100000):
    """  check mds server connections.
    """
    tree_nodes = {
        "pcs_east": "PCRL01",
        "efit_east": "WMHD",
        "energy_east": "taue_mhd",
        "east": "PLHI1",
    }
    from MDSplus import Tree
    tree_messages = []
    for tree_name, node_name in tree_nodes.items():
        environVariable = tree_name + "_path"
        dummy_server_ip = mdsConfig.get_mds_config().hostname
        os.environ[environVariable] = f"{dummy_server_ip}::"
        try:
            tree = Tree(tree=tree_name, shot=shot, mode='ReadOnly')
            node_name = '\\' + node_name
            node_data = tree.getNode(node_name).getData().data()
        except Exception as e:
            # tree_messages.append(f"{tree_name}:\n {e.message}")
            screen_print(tree_name)
            print(e.message)
    
    if len(tree_messages) == 0:
        screen_print("Pass MDS Connection Test")
        return True
    else:
        return False

def check_long_discharge():
    shot = 106915  # the discharge with more than 1000 seconds duration
    tree_name = 'east'
    node_name = 'HRS01H'
    try:
        node_data, node_time = getData.getNodeDataByTree(
            shot, 
            tree_name=tree_name, 
            node_name=node_name)
        screen_print(f'Pass long discharge test for ECE diag {node_name} in {shot}')
    except Exception as e:
        screen_print(tree_name)
        error_message = str(e) + f'\n {shot} - {tree_name} - {node_name}'
        raise type(e)(error_message) from e

def check_valid_shot(
        shot_file,
        mds_server='mds.ipp.ac.cn',
        short_time_ub: float = 2.0,
        ref_flat_top_ub: float = 0.5):
    """
        Checks if a shot has sufficient controlled duration and a flat-top phase.

        Args:
            shot (int): The shot number to check.
            mds_server (str, optional): The MDSplus server address. Defaults to 'mds.ipp.ac.cn'.
            short_time_ub (float, optional): The minimum required control time in seconds. Defaults to 5.0.
            ref_flat_top_ub (float, optional): The minimum flat-top duration in seconds. Defaults to 0.5.

        Returns:
            bool: True if both criteria are met, False otherwise.
    """
    east_analyzer = eastAnalyzer.EASTAnalyzer(
        shot_file=shot_file,
        mds_server=mds_server)
    shot_dict = east_analyzer.shot_dict

    reasons = []

    if any(v is None for v in shot_dict.values()):
        reasons.append('none_values')
        return False, reasons

    valid_control_start = shot_dict['valid_control_start']
    valid_control_end = shot_dict['valid_control_end']
    is_valid_long = (valid_control_end - valid_control_start) > short_time_ub
    if not is_valid_long:
        reasons.append('short_duration')

    segments = east_analyzer.find_continuous_flat_segments(
        flat_threshold=1e-3, min_duration=ref_flat_top_ub)
    if len(segments) == 0:
        segments = east_analyzer.find_relative_flat_segments(
            n_decimal_palaces=2, min_duration=ref_flat_top_ub, max_slow_rise_rate=0.1)
    has_flat_top = len(segments) > 0
    if not has_flat_top:
        reasons.append('no_flat_top')

    Ip = shot_dict['PCRL01']
    Ip_abs = np.abs(Ip)
    p50 = np.percentile(Ip_abs, 50)
    non_negative_shot = np.count_nonzero(Ip < -p50) < 10
    if not non_negative_shot:
        reasons.append('negative_Ip')

    is_valid = len(reasons) == 0
    return is_valid, reasons


if __name__ == "__main__":
    check_connections()