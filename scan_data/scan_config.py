from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import pathlib
from typing import Dict
from .compat import load_yaml_config, strpath2path

class MDSScanConfig(BaseSettings):
    eps: float = 1e-7
    cache_dir: str = '.cache'
    hostname: str = 'mds.ipp.ac.cn'
    proj_base_dir: pathlib.Path = pathlib.Path(__file__).resolve().parent
    base_config: Dict = load_yaml_config(proj_base_dir.joinpath('configs/scan_config.yml'))
    DATABASE_dir: pathlib.Path = strpath2path(base_config['base_dir'])
    model_config = SettingsConfigDict(
    env_file=('.env', '.env.local'),
    env_prefix='MDSSCAN_',)

__mds_scan_config__: Optional[MDSScanConfig] = None

def get_mds_scan_config() -> MDSScanConfig:
    global __mds_scan_config__ 
    if __mds_scan_config__ is None:
        __mds_scan_config__ = MDSScanConfig()
    return __mds_scan_config__