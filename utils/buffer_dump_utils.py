"""
Buffer Dump Utilities
Provides buffer content dump and verification functionality.
"""

import pickle
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import sys
import logging

logger = logging.getLogger(__name__)


def dump_distributed_buffer_content(
    buffer_obj,
    output_file: Path
) -> Dict[str, Any]:
    """
    Dump distributed replay buffer content (used by APEX).
    
    Args:
        buffer_obj: Buffer object obtained from the replay actor
        output_file: Output pkl file path
    
    Returns:
        Statistics dict
    """
    return dump_buffer_content(buffer_obj, output_file)


def get_compressed_block_size(sample_batch) -> Dict[str, int]:
    """Get the actual size of each field in a compressed block."""
    sizes = {}
    
    for key, value in sample_batch.items():
        if isinstance(value, bytes):
            sizes[key] = len(value)
        elif isinstance(value, np.ndarray):
            sizes[key] = value.nbytes
        elif isinstance(value, (int, float, bool)):
            sizes[key] = sys.getsizeof(value)
        else:
            sizes[key] = sys.getsizeof(value)
    
    return sizes


def dump_buffer_content(
    buffer_obj,
    output_file: Path
) -> Dict[str, Any]:
    """
    Dump full buffer content (including _storage) for verification.
    
    Args:
        buffer_obj: Replay buffer object (PrioritizedBlockReplayBuffer or MultiAgentBuffer)
        output_file: Output pkl file path
    
    Returns:
        Statistics dict
    """
    
    dump_data = {
        'metadata': {},
        'raw_storage': None,  # full _storage object
        'statistics': {},
    }
    
    try:
        # Handle MultiAgent buffer
        if hasattr(buffer_obj, 'replay_buffers'):
            logger.info("Detected MultiAgent buffer")
            dump_data['metadata']['buffer_type'] = 'MultiAgent'
            dump_data['metadata']['policies'] = {}
            dump_data['raw_storage'] = {}
            
            for policy_id, policy_buffer in buffer_obj.replay_buffers.items():
                policy_info = _dump_single_buffer(policy_buffer, policy_id)
                dump_data['metadata']['policies'][policy_id] = policy_info['metadata']
                dump_data['raw_storage'][policy_id] = policy_info['raw_storage']
                dump_data['statistics'][policy_id] = policy_info['statistics']
        
        # Handle single buffer
        elif hasattr(buffer_obj, '_storage'):
            logger.info("Detected Single buffer")
            dump_data['metadata']['buffer_type'] = 'Single'
            buffer_info = _dump_single_buffer(buffer_obj, 'default')
            dump_data['metadata'] = buffer_info['metadata']
            dump_data['raw_storage'] = buffer_info['raw_storage']
            dump_data['statistics'] = buffer_info['statistics']
        
        else:
            raise ValueError("Unknown buffer type: missing _storage or replay_buffers attribute")
        
        with open(output_file, 'wb') as f:
            pickle.dump(dump_data, f)
        
        logger.info(f"Buffer content dumped to: {output_file}")
        logger.info(f"  File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
        
        return dump_data['statistics']
    
    except Exception as e:
        logger.error(f"Buffer dump failed: {e}", exc_info=True)
        raise


def _dump_single_buffer(
    buffer_obj,
    buffer_id: str
) -> Dict[str, Any]:
    """Dump full content of a single buffer (including _storage)."""
    
    result = {
        'metadata': {},
        'raw_storage': None,
        'statistics': {}
    }
    
    if not hasattr(buffer_obj, '_storage'):
        logger.warning(f"Buffer {buffer_id} has no _storage attribute")
        return result
    
    storage = buffer_obj._storage
    storage_len = len(storage)
    
    logger.info(f"  Saving full _storage object (length={storage_len})")
    
    result['raw_storage'] = {
        'storage': storage,
        'storage_len': storage_len,
        '_num_added': getattr(buffer_obj, '_num_added', None),
        '_next_idx': getattr(buffer_obj, '_next_idx', None),
        'capacity': getattr(buffer_obj, 'capacity', None),
        'sub_buffer_size': getattr(buffer_obj, 'sub_buffer_size', None),
        'compress_base': getattr(buffer_obj, 'compress_base', None),
        # Priority-related info (if available)
        '_it_sum': getattr(buffer_obj, '_it_sum', None),
        '_it_min': getattr(buffer_obj, '_it_min', None),
        '_max_priority': getattr(buffer_obj, '_max_priority', None),
    }
    
    # Basic metadata
    result['metadata'] = {
        'buffer_id': buffer_id,
        'storage_len': storage_len,
        'capacity': getattr(buffer_obj, 'capacity', None),
        'storage_type': str(type(storage)),
        'sub_buffer_size': getattr(buffer_obj, 'sub_buffer_size', None),
        'compress_base': getattr(buffer_obj, 'compress_base', None),
    }
    
    # Quick stats: only analyze first few blocks to estimate
    total_compressed_bytes = 0
    total_raw_bytes = 0
    sample_count = min(10, storage_len)
    
    for idx in range(sample_count):
        try:
            block_data = storage[idx]
            
            for key, value in block_data.items():
                if isinstance(value, bytes):
                    compressed_size = len(value)
                    total_compressed_bytes += compressed_size
                    total_raw_bytes += compressed_size * 20  # conservative estimate
                elif isinstance(value, np.ndarray):
                    total_compressed_bytes += value.nbytes
                    total_raw_bytes += value.nbytes
                else:
                    size = sys.getsizeof(value)
                    total_compressed_bytes += size
                    total_raw_bytes += size
        except Exception as e:
            logger.warning(f"  Failed to analyze block {idx}: {e}")
            continue
    
    # Compute statistics (extrapolated from samples)
    if sample_count > 0:
        avg_compressed = total_compressed_bytes / sample_count
        avg_raw = total_raw_bytes / sample_count
        est_total_compressed = avg_compressed * storage_len
        est_total_raw = avg_raw * storage_len
    else:
        est_total_compressed = 0
        est_total_raw = 0
    
    result['statistics'] = {
        'storage_len': storage_len,
        'avg_compressed_per_block': avg_compressed if sample_count > 0 else 0,
        'avg_raw_per_block': avg_raw if sample_count > 0 else 0,
        'compression_ratio': (est_total_raw / est_total_compressed) if est_total_compressed > 0 else 1.0,
        'estimated_total_memory_mb': est_total_compressed / 1024 / 1024,
    }
    
    logger.info(f"  Stats: {result['statistics']['compression_ratio']:.2f}x compression ratio, " 
                f"~{result['statistics']['estimated_total_memory_mb']:.1f} MB total memory")
    
    return result


def analyze_buffer_dump(dump_file: Path) -> None:
    """Analyze a dumped buffer file."""
    
    with open(dump_file, 'rb') as f:
        data = pickle.load(f)
    
    print("\n" + "="*80)
    print(f"Buffer Dump Analysis: {dump_file.name}")
    print("="*80)
    
    metadata = data.get('metadata', {})
    raw_storage = data.get('raw_storage', None)
    statistics = data.get('statistics', {})
    
    print(f"\nMetadata:")
    print(f"   Buffer type: {metadata.get('buffer_type', 'Unknown')}")
    
    if metadata.get('buffer_type') == 'MultiAgent':
        print(f"   Policies: {list(metadata.get('policies', {}).keys())}")
        for policy_id, policy_meta in metadata.get('policies', {}).items():
            print(f"\n   Policy: {policy_id}")
            print(f"     Storage count: {policy_meta.get('storage_len', 0):,}")
            print(f"     Capacity: {policy_meta.get('capacity', 0):,}")
            print(f"     Block size: {policy_meta.get('sub_buffer_size', 'N/A')}")
    else:
        print(f"   Storage count: {metadata.get('storage_len', 0):,}")
        print(f"   Capacity: {metadata.get('capacity', 0):,}")
        print(f"   Block size: {metadata.get('sub_buffer_size', 'N/A')}")
    
    print(f"\nStatistics:")
    if isinstance(statistics, dict):
        for key, stats in statistics.items():
            if isinstance(stats, dict):
                print(f"\n   {key}:")
                print(f"     Storage blocks: {stats.get('storage_len', 0):,}")
                print(f"     Compression ratio: {stats.get('compression_ratio', 1.0):.2f}x")
                print(f"     Avg block size (compressed): {stats.get('avg_compressed_per_block', 0) / 1024:.1f} KB")
                print(f"     Avg block size (raw): {stats.get('avg_raw_per_block', 0) / 1024:.1f} KB")
                print(f"     Estimated total memory: {stats.get('estimated_total_memory_mb', 0):.1f} MB")
    
    print(f"\nFull Storage Data:")
    if raw_storage is not None:
        if isinstance(raw_storage, dict):
            if 'storage' in raw_storage:
                storage_list = raw_storage.get('storage', [])
                print(f"   Full _storage saved (type: {type(storage_list).__name__}, length: {len(storage_list):,})")
                if len(storage_list) > 0:
                    first_block = storage_list[0]
                    print(f"   Sample block structure:")
                    print(f"     Type: {type(first_block).__name__}")
                    if hasattr(first_block, 'keys'):
                        print(f"     Keys: {list(first_block.keys())}")
            else:
                for policy_id, policy_storage in raw_storage.items():
                    storage_list = policy_storage.get('storage', [])
                    print(f"   {policy_id}: Full _storage saved (length: {len(storage_list):,})")
        else:
            print(f"   Saved (type: {type(raw_storage).__name__})")
    else:
        print(f"   Full storage not saved")
    
    print("\n" + "="*80)

