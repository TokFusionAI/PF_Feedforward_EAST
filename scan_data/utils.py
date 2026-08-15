# -*- coding: utf-8 -*-
from .compat import shape_params
from .compat import save_to_file, load_yaml_config, convert_hdf5_2dict, strpath2path
from .compat import hf_keys_fetch, calc_sample_frequency, interp_nD_time_arr
import h5py
import pathlib
import pandas as pd
import numpy as np
from scipy import interpolate
import os
from tqdm import tqdm
from typing import List
import multiprocessing as mp
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed


def _resolve_dir_nodes(dir_name, cfg=None):
    """Resolve node list for a directory from config.

    Args:
        dir_name: Key in ``dir_infos`` of ``scan_config.yml``.
        cfg: Config dict. Defaults to ``get_mds_scan_config().base_config``.

    Returns:
        Tuple of (diag_nodes, dir_type).
    """
    if cfg is None:
        from .scan_config import get_mds_scan_config
        cfg = get_mds_scan_config().base_config
    dir_info = cfg['dir_infos'][dir_name]
    nodes = []
    for group_name in dir_info['node_list']:
        if group_name.endswith('.yml'):
            from .scan_config import get_mds_scan_config
            proj_base_dir = get_mds_scan_config().proj_base_dir
            ext_cfg = load_yaml_config(proj_base_dir.joinpath('configs', group_name))
            ext_nodes = next(v for v in ext_cfg.values() if isinstance(v, list))
            nodes.extend(sorted(ext_nodes))
        else:
            nodes.extend(cfg['node_infos'][group_name])
    return nodes, dir_info.get('type', 'east')


def _load_error_shots(path: os.PathLike) -> np.ndarray:
    """Load error shot numbers from path; return empty array if file is empty."""
    if path is None:
        raise ValueError("error_shots_path must be provided.")
    if not pathlib.Path(path).exists():
        return np.array([], dtype=int)
    return np.atleast_1d(np.loadtxt(path, dtype=int))


def _cleanup_empty_error_file(path: os.PathLike | None) -> None:
    """Remove the error file if it exists but contains no shot numbers."""
    if path is None:
        return
    p = pathlib.Path(path)
    if p.exists() and p.stat().st_size == 0:
        p.unlink()

# del hf keys
def del_keys_from_file(h5_file: str, keys):
    h5_dict = {}
    with h5py.File(h5_file, 'r') as hf:
        hf_keys = list(hf.keys())
        left_keys = set(hf_keys).difference(set(keys))
        left_keys = list(left_keys)
        for key in left_keys:
            val = hf[key][()]
            if val.shape is None:
                val = None
            h5_dict[key] = val
    save_to_file(h5_file, h5_dict, is_overwrite=True)


def get_default_scan_nodes():
    nodes = []
    config_path = pathlib.Path(
        __file__).resolve().parent.joinpath('configs/scan_config.yml')
    config = load_yaml_config(config_path)
    scan_list = list(config['scan_list'])
    # scan_list.append('efit_control_params')
    scan_list.extend(config['secondary_nodes'])
    for nodes_key in scan_list:
        nodes.extend(config[nodes_key])
    nodes.remove('SYCIC2')
    return nodes


def compare_shot_dicts(
    shot_dict0,
    shot_dict1,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    path: str = "root",
    diffs: Optional[list] = None,
):
    """
    Recursively compare two shot dictionaries.

    Float arrays and float scalars are compared with tolerance so small
    differences in time arrays are ignored.
    """
    if diffs is None:
        diffs = []

    if type(shot_dict0) is not type(shot_dict1):
        diffs.append(
            f"{path}: type mismatch "
            f"{type(shot_dict0).__name__} != {type(shot_dict1).__name__}"
        )
        return diffs

    if isinstance(shot_dict0, dict):
        keys0 = set(shot_dict0.keys())
        keys1 = set(shot_dict1.keys())

        for key in sorted(keys0 - keys1):
            diffs.append(f"{path}.{key}: only in first dict")
        for key in sorted(keys1 - keys0):
            diffs.append(f"{path}.{key}: only in second dict")

        for key in sorted(keys0 & keys1):
            compare_shot_dicts(
                shot_dict0[key],
                shot_dict1[key],
                rtol=rtol,
                atol=atol,
                path=f"{path}.{key}",
                diffs=diffs,
            )
        return diffs

    if isinstance(shot_dict0, (list, tuple)):
        if len(shot_dict0) != len(shot_dict1):
            diffs.append(
                f"{path}: length mismatch {len(shot_dict0)} != {len(shot_dict1)}"
            )
            return diffs

        for idx, (item0, item1) in enumerate(zip(shot_dict0, shot_dict1)):
            compare_shot_dicts(
                item0,
                item1,
                rtol=rtol,
                atol=atol,
                path=f"{path}[{idx}]",
                diffs=diffs,
            )
        return diffs

    if isinstance(shot_dict0, np.ndarray):
        if shot_dict0.shape != shot_dict1.shape:
            diffs.append(
                f"{path}: shape mismatch {shot_dict0.shape} != {shot_dict1.shape}"
            )
            return diffs

        if np.issubdtype(shot_dict0.dtype, np.floating):
            if not np.allclose(
                shot_dict0,
                shot_dict1,
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            ):
                max_abs_diff = np.nanmax(np.abs(shot_dict0 - shot_dict1))
                diffs.append(
                    f"{path}: float array differs, max abs diff = {max_abs_diff}"
                )
        else:
            if not np.array_equal(shot_dict0, shot_dict1):
                diffs.append(f"{path}: array values differ")
        return diffs

    if isinstance(shot_dict0, (float, np.floating)):
        if not np.isclose(
            shot_dict0,
            shot_dict1,
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        ):
            diffs.append(f"{path}: {shot_dict0} != {shot_dict1}")
        return diffs

    if shot_dict0 != shot_dict1:
        diffs.append(f"{path}: {shot_dict0} != {shot_dict1}")

    return diffs

class StatH5Dir:
    def __init__(
        self,
        num_workers: int | None = None,
        eps: float = 1e-7
    ):
        if num_workers is None or num_workers == -1:
            num_workers = int(mp.cpu_count())
        self.num_workers = num_workers
        self.eps = eps

    def stat_dir(
        self,
        h5s_dir: os.PathLike,
        nodes: List[str],
        stat_h5_path: os.PathLike,
        is_skip: bool = False,
        is_freq: bool = False,
        *,
        shot_start=0,
        shot_end=np.inf,
    ):
        """
        Args:
            h5s_dir (os.PathLike): Directory containing HDF5 files to process.
            nodes (List[str]): List of node names/paths within HDF5 files to calculate statistics for.
            stat_h5_path (os.PathLike): Output path for the statistics file. If None, defaults to
                'h5_stat.csv' in the h5s_dir.
            is_skip (bool, optional): If True, skip already processed shots found in existing statistics
                file. Defaults to False.
        Returns:
            None: Results are saved to disk as CSV and Parquet files.
        Notes:
            - Uses parallel processing with ProcessPoolExecutor based on self.num_workers
            - Shot numbers are extracted from filenames (expected format: {shot_number}.h5)
            - If is_skip is True and stat_h5_path exists, only new shots are processed and merged
        """
        h5s_dir = pathlib.Path(h5s_dir)
        self.h5s_dir = h5s_dir
        if stat_h5_path is None:
            if is_freq:
                stat_h5_path = h5s_dir.joinpath('h5_stat_freqs.csv')
            else:
                stat_h5_path = h5s_dir.joinpath('h5_stat.csv')
        stat_h5_path = pathlib.Path(stat_h5_path)
        if stat_h5_path.exists() and is_skip:
            temp_stat_h5_path = h5s_dir.joinpath('temp_h5_stat.csv')
            df = pd.read_csv(stat_h5_path, index_col=0, low_memory=False)
            calced_shots = df.index.tolist()
        else:
            temp_stat_h5_path = stat_h5_path
            calced_shots = []
        shots = [int(file.parts[-1][:-3]) for file in h5s_dir.glob("*.h5")]
        shots = list(set(shots).difference(calced_shots))
        h5_files = [h5s_dir.joinpath(f"{shot}.h5") for shot in shots]
        shot_starts = [shot_start] * len(shots)
        shot_ends = [shot_end] * len(shots)
        temp_df = self._stat_files(h5_files, nodes, is_freq, shot_starts, shot_ends)
        if stat_h5_path.exists() and is_skip:
            df = pd.concat([df, temp_df])
            os.unlink(temp_stat_h5_path)
        else:
            df = temp_df
        if str(stat_h5_path).endswith('.csv'):
            df.to_csv(stat_h5_path)
        elif str(stat_h5_path).endswith('.parquet'):
            df.to_parquet(stat_h5_path)
        else:
            raise ValueError(
                f'Unsupported file format for stat_h5_path:{str(stat_h5_path)}.' 
                'Only `.csv` and `.parquet` formats are supported.')

    def _stat_files(self, h5_files, nodes, is_freq, shot_starts, shot_ends):
        dir_name = self.h5s_dir.name
        worker = self._stat_freq_file if is_freq else self._stat_file
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_file = {executor.submit(worker, h5_file, nodes, shot_start, shot_end): h5_file
                              for h5_file, shot_start, shot_end in zip(h5_files, shot_starts, shot_ends)}
            results = []
            with tqdm(as_completed(future_to_file), total=len(future_to_file), desc=f'Counting {dir_name}: ') as pbar:
                for future in pbar:
                    h5_file = future_to_file[future]
                    pbar.set_postfix_str(pathlib.Path(h5_file).name)
                    results.append(future.result())
        df = pd.DataFrame(results)
        df = df.set_index('shot')
        return df

    def _stat_file(self, h5_file, nodes, shot_start=0, shot_end=np.inf):
        """
        Calculate statistics for a single HDF5 file.

        Args:
            h5_file: Path to HDF5 file
            nodes: List of node names to process
            shot_start: Start time for filtering data (default: 0)
            shot_end: End time for filtering data (default: None, no upper limit)

        Returns:
            dict: Statistics for each node including sum, square_sum, and length
        """
        h5_stat_dict = {}
        with h5py.File(h5_file, mode='r') as hf:
            try:
                shot = hf["shot"][()]
            except KeyError:
                shot = int(h5_file.parts[-1][:-3])
            h5_stat_dict['shot'] = shot
            if shot_end is None:
                shot_end = self._get_shot_end(shot)
            keys = list(hf.keys())
            if not set(nodes).issubset(keys):
                missing_nodes = list(set(nodes).difference(keys))
                raise KeyError(f"In {str(h5_file)} missing {missing_nodes}")

            for node in nodes:
                dataset = hf[node]
                if dataset.shape is None or len(dataset.shape) == 0:
                    self._set_invalid_stats(h5_stat_dict, node)
                    continue

                data = dataset[()]
                time_keys = [f"{node}_time", "time"]
                time_key_found = None
                for time_key in time_keys:
                    if time_key in hf.keys():
                        time_key_found = time_key
                        break
                if time_key_found is None:
                    raise KeyError(
                        f"Neither `{node}_time` nor `time` found in {str(h5_file)}")
                time_key = time_key_found
                times = hf[time_key][()]
                if times.shape is None:
                    # for EFIT shot like [124360, 98791]
                    self._set_invalid_stats(h5_stat_dict, node)
                    continue
                valid_ids = (shot_start - self.eps <=
                             times) & (times <= shot_end + self.eps)

                if np.count_nonzero(valid_ids) <= 1:
                    self._set_invalid_stats(h5_stat_dict, node)
                    continue
                try:
                    valid_data = data[valid_ids]
                except Exception as e:
                    raise Exception(
                        f"Error processing node '{node}' in file '{h5_file}': {str(e)}") from e
                h5_stat_dict[node] = True
                h5_stat_dict[f"{node}_sum"] = np.sum(
                    valid_data, dtype=np.float64)
                h5_stat_dict[f"{node}_square_sum"] = np.sum(
                    valid_data**2, dtype=np.float64)
                h5_stat_dict[f"{node}_length"] = len(valid_data)
        return h5_stat_dict

    def calc_freq(
        self,
        h5s_dir: os.PathLike,
        time_nodes: List[str],
        freq_h5_path: os.PathLike = None,
        is_skip: bool = False,
        *,
        shot_start=0,
        shot_end=np.inf,
    ):
        """Calculate sample frequencies of time-axis nodes across all HDF5 files in a directory.

        Args:
            h5s_dir (os.PathLike): Directory containing HDF5 files to process.
            time_nodes (List[str]): Time-axis node names (e.g. ``['IP_time', 'NE_time']``).
            freq_h5_path (os.PathLike): Output CSV path. Defaults to ``h5_stat_freqs.csv`` in h5s_dir.
            is_skip (bool): If True, skip shots already present in an existing output file.
            shot_start: Start time boundary passed to stat_freq_file.
            shot_end: End time boundary passed to stat_freq_file.
        """
        self.stat_dir(
            h5s_dir,
            time_nodes,
            freq_h5_path,
            is_skip=is_skip,
            is_freq=True,
            shot_start=shot_start,
            shot_end=shot_end,
        )

    def _stat_freq_file(self, h5_file, nodes, shot_start=0, shot_end=np.inf):
        hf_time_d = hf_keys_fetch(h5_file, [*nodes, 'shot'])
        hf_freq_d = {}
        for key_time in nodes:
            times = hf_time_d[key_time]
            if times is None:
                freq = None
            else:
                valid_ids = (shot_start - self.eps <= times) & (times <= shot_end + self.eps)
                if np.count_nonzero(valid_ids) <= 1:
                    freq = None
                else:
                    freq = calc_sample_frequency(times[valid_ids])
            freq_key = key_time.replace('time', 'frequency')
            hf_freq_d[freq_key] = freq
        hf_freq_d['shot'] = hf_time_d['shot']
        return hf_freq_d

    def _get_shot_end(self, shot):
        EAST_dir = "$DATABASE_PATH/DataBase/EAST"
        EAST_dir = strpath2path(EAST_dir)
        east_file = EAST_dir.joinpath(f"{shot}.h5")
        with h5py.File(east_file, 'r') as hf:
            time = hf['time'][()]
            shot_end = time[-1]
        return shot_end

    def _set_invalid_stats(self, stat_dict, node):
        """Helper method to set invalid/missing statistics for a node."""
        stat_dict[node] = False
        stat_dict[f"{node}_sum"] = None
        stat_dict[f"{node}_square_sum"] = None
        stat_dict[f"{node}_length"] = 0

    def calc_MS(
        self,
        nodes: List[str],
        stat_file: pathlib.Path,
        MS_file: pathlib.Path = None,
        **kwargs,
    ):
        # Read CSV/Parquet file more efficiently
        if str(stat_file).endswith('.parquet'):
            stat_df = pd.read_parquet(stat_file)
        elif str(stat_file).endswith('.csv'):
            stat_df = pd.read_csv(stat_file, index_col=0, low_memory=False)
        else:
            raise ValueError(f'Unsupported stat_file={str(stat_file)}')

        # Pre-allocate result arrays for vectorization
        means = np.full(len(nodes), np.nan, dtype=np.float64)
        stdevs = np.full(len(nodes), np.nan, dtype=np.float64)

        # Get kwargs parameters once
        d2_signal_list = kwargs.get('d2_signal_list', [])
        num_channel_list = kwargs.get('num_channel_list', [])
        d2_signal_dict = dict(
            zip(d2_signal_list, num_channel_list)) if d2_signal_list else {}

        # Vectorized computation for all nodes
        for i, node in enumerate(nodes):
            # Get required columns once
            square_sum_col = f"{node}_square_sum"
            sum_col = f"{node}_sum"

            # Create mask for valid data (vectorized)
            square_sums = stat_df[square_sum_col].values.astype(np.float128)
            sums = stat_df[sum_col].values.astype(np.float128)
            existence = stat_df[node].values.astype(bool)
            lengths = stat_df[f"{node}_length"].values

            # Combined mask for finite, existing data
            finite_mask = np.isfinite(square_sums)
            valid_mask = finite_mask & existence

            if not np.any(valid_mask):
                continue

            # Filter data using mask
            valid_square_sums = square_sums[valid_mask]
            valid_sums = sums[valid_mask]
            valid_lengths = lengths[valid_mask]

            # Calculate total samples
            N_sample = np.sum(valid_lengths, dtype=np.int64)
            if node in d2_signal_dict:
                N_sample *= d2_signal_dict[node]

            if N_sample == 0:
                continue

            # Vectorized sum calculations
            sample_sum = np.sum(valid_sums)
            sample_square_sum = np.sum(valid_square_sums)

            # Calculate mean and standard deviation
            mean = sample_sum / N_sample
            variance = (sample_square_sum - (sample_sum**2) /
                        N_sample) / (N_sample - 1)
            stdev = np.sqrt(max(0, variance))  # Ensure non-negative variance

            means[i] = float(mean)
            stdevs[i] = float(stdev)

        # Create result DataFrame efficiently
        MS_df = pd.DataFrame({
            'mean': means,
            'stDev': stdevs
        }, index=nodes).T
        if MS_file is not None:
            MS_df.to_csv(MS_file)
        return MS_df

def add_control_time(dst_file, ctrl_file):
    """Add control time axis from ctrl_file to dst_file as a new entry.
    
    Args:
        dst_file (str or Path): Destination HDF5 file to add control time to
        ctrl_file (str or Path): Source HDF5 file containing the control time axis
    """
    with h5py.File(ctrl_file, 'r') as ctrl_hf:
        ctrl_time = ctrl_hf['time'][()]
    
    shot_dict = {'control_time': ctrl_time}
    save_to_file(dst_file, shot_dict)

def merge_nodes(dst_file, src_file, nodes):
    shot_dict = {}
    # shape_params = extract_shape_params(efit_file)
    # efit_dict.update(shape_params)
    with h5py.File(dst_file, 'r') as hf:
        dst_timeAxis = hf['time'][()]
    with h5py.File(src_file, 'r') as hf:
        src_timeAxis = hf['time'][()]
        for key_name in nodes:
            node_data = hf[key_name][()]
            if isinstance(node_data, h5py.Empty):
                new_data = None
            elif node_data.ndim == 0:
                new_data = node_data
            else:
                new_data = np.interp(dst_timeAxis, src_timeAxis, node_data)
            shot_dict[key_name] = new_data
    save_to_file(dst_file, shot_dict)

def __reinterp_LR_arr__(base_x, node_data):
    start_val = node_data[0]
    end_val = node_data[-1]
    # find left boundary
    for idx in range(len(node_data)):
        val = node_data[idx]
        left_idx = idx
        if ~np.isclose(start_val, val, atol=1e-10, rtol=0).all():
            break

    if left_idx == len(node_data) - 1:
        return node_data

    if left_idx == 0 and np.isnan(node_data).all():
        return None

    # find right boundary
    for idx in range(len(node_data)-1, -1, -1):
        val = node_data[idx]
        right_idx = idx
        if ~np.isclose(end_val, val, atol=1e-10, rtol=0).all():
            break

    val_base_x = base_x[left_idx-1:right_idx + 2]
    val_node_data = node_data[left_idx-1:right_idx + 2]
    if len(val_base_x) != len(base_x):
        interp = interpolate.interp1d(val_base_x,
                                      val_node_data,
                                      kind='slinear',
                                      fill_value='extrapolate')
        node_data = interp(base_x)
    return node_data

def reinterped_dict(h5_file):
    h5_file_dict = {}
    with h5py.File(h5_file, 'r') as hf:
        h5_file_dict['shot'] = hf['shot'][()]
        time = hf['time'][()]
        h5_file_dict['time'] = time
        efit_time = hf['efit_time'][()]
        h5_file_dict['efit_time'] = efit_time
        nodes = list(hf.keys())
        nodes.remove('time')
        nodes.remove('shot')
        nodes.remove('efit_time')
        times = hf['time'][()]
        for node in nodes:
            if hf[node].shape is None:
                node_data = None
            else:
                node_data = hf[node][()]
                # try:
                node_data = __reinterp_LR_arr__(times, node_data)
                # except ValueError:
                #     raise ValueError(f"{node} in {h5_file} ValueError")
            h5_file_dict[node] = node_data
    return h5_file_dict


def merge_MS(east_ms_file, efit_ms_file, nodes=['LCFS']):
    """  merge efit ms and east ms
    """
    east_ms_df = pd.read_csv(east_ms_file, index=0)
    efit_ms_df = pd.read_csv(efit_ms_file, index=0)
    for node in nodes:
        east_ms_df.loc[:, f'{node}'] = efit_ms_df.loc[:, f'{node}']
    return east_ms_df


def update_MS(east_ms_file):
    """ How to get east_dir h5_global_MS_add.csv
    The purpose is more numberic stable, to compress the very small data. 
    """
    df = pd.read_csv(east_ms_file, index_col=0)
    ids = df.loc['mean', :].abs() < 2e-2
    s_cols = df.columns[ids]
    df.loc['mean', s_cols] = df.loc['mean', :][ids] + 2e-2
    return df

def nodes_status_hf(nodes, h5_file):
    """
    Checks the existence and content of specified nodes in an HDF5 file.
    Args:
        nodes (list or iterable): List of node (dataset) names to check in the HDF5 file.
        h5_file (str or Path): Path to the HDF5 file to be checked.
    Returns:
        dict: A dictionary containing the file path under the key 'file', and for each node:
            - 'ok' if the node exists and is not empty,
            - 'missing' if the node does not exist in the file,
            - 'empty' if the node exists but has no data (shape is None or size is 0).
    """
    result = {}
    with h5py.File(h5_file, 'r') as hf:
        result['file'] = str(h5_file)
        for key in nodes:
            if key not in hf:
                result[key] = 'missing'
            elif hf[key].shape is None or hf[key].size == 0:
                result[key] = 'empty'
            else:
                result[key] = 'ok'
    return result

def check_diag_h5_file(h5_path, diag_nodes):
    """Check diagnostic HDF5 file for consistency between data and time axes.

    Validates that each diagnostic node has a corresponding time axis and that
    the time axis and data lengths are consistent. Detects mismatches between
    time axis and signal data shapes.

    Args:
        h5_path (PathLike): Path to the HDF5 file to validate.
        diag_nodes (List[str]): List of diagnostic node names to check.
            These are the signal/data nodes (e.g., "DAL1", "DAU1", etc.).

    Returns:
        Tuple[List[Tuple] | None, int | None]:
            On success: (warnings_list, None)
                - warnings_list: List of issue tuples ``(shot_name, node, issue_type)``
                  where shot_name is the file stem (shot number as string), and
                  issue_type is one of:

                    - ``'None'``: Time axis missing or empty
                    - ``'longer'``: Time axis has more samples than signal data
                    - ``'shorter'``: Time axis has fewer samples than signal data
                    - ``'inexist'``: Diagnostic node not found in file

                Empty list means all nodes passed validation.

            On failure: (None, shot_number)
                - shot_number: extracted from h5_path.stem (file name without ext)
                - Error typically indicates file cannot be opened or read.

    Notes:
        - Looks for time axis in this order: ``{node}_time``, then ``time``
        - re-run all warning and error shots. 
    """
    warnings = []
    try:
        with h5py.File(h5_path, 'r') as hf:
            for node in diag_nodes:
                if node in hf:
                    time_nodes = [f'{node}_time', 'time']
                    time = None
                    for time_node in time_nodes:
                        if time_node in hf:
                            time = hf[time_node][()]
                            break
                    if time is None:
                        warnings.append((h5_path.stem, node, 'None'))
                        continue
                    node_data = hf[node]
                    if node_data.shape is not None:
                        if time.shape is None:
                            warnings.append((h5_path.stem, node, 'None'))
                            continue
                        node_data = node_data[()]
                        if len(time) > len(node_data):
                            warnings.append((h5_path.stem, node, 'longer'))
                        elif len(time) < len(node_data):
                            warnings.append((h5_path.stem, node, 'shorter'))
                else:
                    warnings.append((h5_path.stem, node, 'inexist'))
    except Exception as e:
        return None, int(h5_path.stem)
    return warnings, None

def update_diag_h5_with_hard_link(h5_file, diag_nodes):
    """Update diagnostic HDF5 file by creating hard links for node time axes.
    
    *This needs to be used in `check_diag_h5_file`*. 

    Only processes files that have 'time' key but no '{node}_time' keys.
    Creates hard links from 'time' to '{node}_time' for each diagnostic node,
    converting to raw format style without duplicating data on disk.
    
    Args:
        h5_file (str or pathlib.Path): Path to the HDF5 file to update.
        diag_nodes (list): List of diagnostic node names to process.
    
    Returns:
        int: 0 if file was skipped (already has {node}_time keys), 1 if processed successfully.
    
    Raises:
        FileNotFoundError: If h5_file does not exist.
        KeyError: If 'time' key is missing in the file (only checked for non-skipped files).
    """
    h5_file = pathlib.Path(h5_file) if not isinstance(h5_file, pathlib.Path) else h5_file
    
    if not h5_file.exists():
        raise FileNotFoundError(f"HDF5 file does not exist: {h5_file}")

    
    with h5py.File(h5_file, 'a') as hf:
        # Check if any {node}_time already exists
        if any(f'{node}_time' in hf for node in diag_nodes):
            # File already has {node}_time format, skip entirely
            return 0
        
        if 'time' not in hf:
            raise KeyError(f"'time' key not found in {h5_file}")
        
        for node in diag_nodes:
            hf[f'{node}_time'] = hf['time']
        hf.flush()
    return 1

# ---------------------------------------------------------------------------
# MergeNodes
# ---------------------------------------------------------------------------

class MergeNodes:
    """Merge nodes from multiple H5 files onto a common base time axis.

    Uses grouped reads (each H5 file opened only once) and vectorised
    linear interpolation for efficiency.

    Args:
        dtype (np.dtype): Output dtype, default ``np.float32``.
        eps (float): Time-overlap tolerance for trimming the base axis.
    """

    def __init__(self, dtype=np.float32, eps=1e-7):
        self.dtype = dtype
        self.eps = eps

    def merge_with_basetime(self, base_h5,
              base_time_node='time',
              h5_data_pairs=None,
              is_interpolate=False):
        """Merge nodes from h5_data_pairs onto base_h5's time axis.

        Args:
            base_h5 (PathLike): H5 file that provides the base time axis.
            base_time_node (str): Dataset name in *base_h5* that holds the
                time axis. Defaults to ``'time'``.
            h5_data_pairs (List[Tuple]): list of
                ``(h5_path, nodes, nodes_type)`` where

                - ``nodes_type='east'``: all nodes share the file's ``'time'``
                - ``nodes_type='raw'`` or ``'diagnostic'``:
                  each node uses its own ``'{node}_time'``

            is_interpolate (bool): if True, resample each node onto
                *base_h5*'s time axis before saving.
        """
        h5_data_pairs = h5_data_pairs or []

        # Group pairs by file path so each file is opened only once
        file_groups = {}
        total_nodes = []
        for h5_path, nodes, nodes_type in h5_data_pairs:
            file_groups.setdefault(str(h5_path), []).append((nodes, nodes_type))
            total_nodes.extend(nodes)

        # Read all source files
        raw = {}
        for fp, entries in file_groups.items():
            raw.update(self._read_one(fp, entries))

        if not is_interpolate:
            save_to_file(base_h5, raw)
            return

        # Resample every node onto the base time axis
        with h5py.File(base_h5, 'r') as hf:
            base_time = hf[base_time_node][()].astype(np.float64)

        save_dict = {}
        for node in total_nodes:
            node_data = raw.get(node)
            if node_data is None:
                save_dict[node] = None
            else:
                node_time = raw[f'{node}_time']
                node_data = interp_nD_time_arr(
                    base_time, node_time, node_data, dtype=self.dtype)
                save_dict[node] = node_data
        save_to_file(base_h5, save_dict)

    def _read_one(self, h5_path, entries):
        """Open one H5 file and read all requested node groups."""
        result = {}
        with h5py.File(h5_path, 'r') as hf:
            for nodes, nodes_type in entries:
                if nodes_type == 'east':
                    time_axis = hf['time'][()]
                    for node in nodes:
                        if hf[node].shape is None:
                            result[node] = None
                            result[f'{node}_time'] = None
                        else:
                            result[node] = hf[node][()].astype(self.dtype)
                            result[f'{node}_time'] = time_axis
                elif nodes_type in ('raw', 'diagnostic'):
                    for node in nodes:
                        val = hf[node][()]
                        tval = hf[f'{node}_time'][()]
                        result[node] = None if isinstance(val, h5py.Empty) else val.astype(self.dtype)
                        result[f'{node}_time'] = None if isinstance(tval, h5py.Empty) else tval
                else:
                    raise ValueError(f"Unknown nodes_type: {nodes_type!r}")
        return result

    def direclty_merge_h5s(base_h5, merged_h5s):
        merge_dict = {}
        for h5 in merged_h5s:
            h5_dict = convert_hdf5_2dict(h5)
            merge_dict.update(h5_dict)
        save_to_file(base_h5, merge_dict, is_overwrite=False)

    @staticmethod
    def build_data_pair_lists(
        base_dir_name: str,
        other_dir_names: List[str],
        *,
        base_time_nodes: List[str],
        database_dir=None,
        config_path=None,
        valid_shots=None,
        max_shots: Optional[int] = None,
    ):
        """Build per-shot data pair lists from a base directory and other directories.

        Each shot produces a list of tuples ``(h5_path, nodes, dir_type)``.
        The first element always comes from *base_dir_name* (the time-axis
        source); subsequent elements come from *other_dir_names*.  Only shots
        whose h5 files exist in **all** specified directories are included.

        Args:
            base_dir_name: Directory name under *database_dir* that provides the
                base time axis (e.g. ``"EAST"``).  Must be a key in the
                ``dir_infos`` section of ``scan_config.yml``.
            other_dir_names: Directory names to pair with the base
                (e.g. ``["EFIT", "HRSDiag"]``).
            base_time_nodes: Node names inside the base h5 that carry the
                time axis (e.g. ``["time"]``).
            database_dir: Root path containing all directory folders.
                Defaults to ``base_dir`` in ``scan_config.yml``.
            config_path: Path to ``scan_config.yml``.  Defaults to the one
                shipped with this package.
            valid_shots: Optional array-like of shot numbers to restrict to.
                The final shot list is the intersection of this with shots
                found on disk.  Accepts any iterable of ints (list, np.ndarray,
                set, etc.).
            max_shots: Cap the number of shots returned.

        Returns:
            Tuple[List[list], List[str]]:
                - **data_pair_lists** – one entry per shot, each is
                  ``[(base_h5, base_time_nodes, base_type),
                  (other_h5, nodes, type), ...]``.
                - **all_nodes** – flat list of node names across the other
                  directories (directory order, then node order).

        Example::

            pairs, nodes = MergeNodes.build_data_pair_lists(
                "EAST", ["EFIT", "HRSDiag"],
                base_time_nodes=["time"],
            )
        """
        if config_path is None:
            script_dir = os.path.dirname(__file__)
            config_path = os.path.join(script_dir, 'configs', 'scan_config.yml')
        config_dict = load_yaml_config(config_path)

        if database_dir is None:
            database_dir = strpath2path(config_dict['base_dir'])
        database_dir = pathlib.Path(database_dir)

        node_infos = config_dict['node_infos']
        dir_infos = config_dict['dir_infos']

        def _resolve_dir(dir_name):
            if dir_name not in dir_infos:
                raise KeyError(
                    f"Directory '{dir_name}' not found in dir_infos. "
                    f"Available: {list(dir_infos.keys())}"
                )
            info = dir_infos[dir_name]
            dir_type = info.get('type', 'east')
            nodes = []
            for group_name in info['node_list']:
                nodes.extend(node_infos[group_name])
            return dir_type, nodes

        base_type, _ = _resolve_dir(base_dir_name)
        other_resolved = []
        all_nodes = []
        for d in other_dir_names:
            d_type, d_nodes = _resolve_dir(d)
            other_resolved.append((d, d_type, d_nodes))
            all_nodes.extend(d_nodes)

        # Intersect shots across all directories
        all_dir_names = [base_dir_name] + list(other_dir_names)
        common_shots = None
        for d in all_dir_names:
            d_path = database_dir / d
            shots = {int(f.stem) for f in d_path.glob('*.h5')}
            common_shots = shots if common_shots is None else common_shots & shots

        if valid_shots is not None:
            common_shots = common_shots & set(int(s) for s in valid_shots)

        common_shots = sorted(common_shots) if common_shots else []
        if max_shots is not None:
            common_shots = common_shots[:max_shots]

        data_pair_lists = []
        for shot in common_shots:
            shot_pairs = []
            base_h5 = database_dir / base_dir_name / f"{shot}.h5"
            shot_pairs.append((base_h5, list(base_time_nodes), base_type))
            for d_name, d_type, d_nodes in other_resolved:
                h5_file = database_dir / d_name / f"{shot}.h5"
                shot_pairs.append((h5_file, d_nodes, d_type))
            data_pair_lists.append(shot_pairs)

        return data_pair_lists, all_nodes
