"""
Buffer Dump Utilities
提供实际buffer内容dump和验证功能
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
    Dump分布式replay buffer内容（APEX使用）
    
    Args:
        buffer_obj: 从replay actor获取的buffer对象
        output_file: 输出pkl文件路径
    
    Returns:
        统计信息字典
    """
    # 使用相同的dump逻辑
    return dump_buffer_content(buffer_obj, output_file)


def get_compressed_block_size(sample_batch) -> Dict[str, int]:
    """获取压缩块中各字段的实际大小"""
    sizes = {}
    
    for key, value in sample_batch.items():
        if isinstance(value, bytes):
            # 压缩后的bytes数据
            sizes[key] = len(value)
        elif isinstance(value, np.ndarray):
            # 未压缩的numpy数据
            sizes[key] = value.nbytes
        elif isinstance(value, (int, float, bool)):
            # 标量
            sizes[key] = sys.getsizeof(value)
        else:
            # 其他类型
            sizes[key] = sys.getsizeof(value)
    
    return sizes


def dump_buffer_content(
    buffer_obj,
    output_file: Path
) -> Dict[str, Any]:
    """
    Dump完整的buffer内容（包括_storage）用于验证
    
    Args:
        buffer_obj: replay buffer对象 (PrioritizedBlockReplayBuffer或MultiAgentBuffer)
        output_file: 输出pkl文件路径
    
    Returns:
        统计信息字典
    """
    
    dump_data = {
        'metadata': {},
        'raw_storage': None,  # 完整的_storage对象
        'statistics': {},
    }
    
    try:
        # 处理MultiAgent buffer
        if hasattr(buffer_obj, 'replay_buffers'):
            logger.info("检测到 MultiAgent buffer")
            dump_data['metadata']['buffer_type'] = 'MultiAgent'
            dump_data['metadata']['policies'] = {}
            dump_data['raw_storage'] = {}
            
            for policy_id, policy_buffer in buffer_obj.replay_buffers.items():
                policy_info = _dump_single_buffer(policy_buffer, policy_id)
                dump_data['metadata']['policies'][policy_id] = policy_info['metadata']
                dump_data['raw_storage'][policy_id] = policy_info['raw_storage']
                dump_data['statistics'][policy_id] = policy_info['statistics']
        
        # 处理single buffer
        elif hasattr(buffer_obj, '_storage'):
            logger.info("检测到 Single buffer")
            dump_data['metadata']['buffer_type'] = 'Single'
            buffer_info = _dump_single_buffer(buffer_obj, 'default')
            dump_data['metadata'] = buffer_info['metadata']
            dump_data['raw_storage'] = buffer_info['raw_storage']
            dump_data['statistics'] = buffer_info['statistics']
        
        else:
            raise ValueError("未知的buffer类型，无_storage或replay_buffers属性")
        
        # 保存到文件
        with open(output_file, 'wb') as f:
            pickle.dump(dump_data, f)
        
        logger.info(f"✓ Buffer内容已dump到: {output_file}")
        logger.info(f"  文件大小: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
        
        return dump_data['statistics']
    
    except Exception as e:
        logger.error(f"Dump buffer失败: {e}", exc_info=True)
        raise


def _dump_single_buffer(
    buffer_obj,
    buffer_id: str
) -> Dict[str, Any]:
    """Dump单个buffer的完整内容（包括_storage）"""
    
    result = {
        'metadata': {},
        'raw_storage': None,
        'statistics': {}
    }
    
    if not hasattr(buffer_obj, '_storage'):
        logger.warning(f"Buffer {buffer_id} 没有_storage属性")
        return result
    
    storage = buffer_obj._storage
    storage_len = len(storage)
    
    logger.info(f"  保存完整的_storage对象 (长度={storage_len})")
    
    # 保存完整的_storage和相关状态
    result['raw_storage'] = {
        'storage': storage,  # 完整的storage list (包含所有blocks)
        'storage_len': storage_len,
        '_num_added': getattr(buffer_obj, '_num_added', None),
        '_next_idx': getattr(buffer_obj, '_next_idx', None),
        'capacity': getattr(buffer_obj, 'capacity', None),
        'sub_buffer_size': getattr(buffer_obj, 'sub_buffer_size', None),
        'compress_base': getattr(buffer_obj, 'compress_base', None),
        # 保存priority相关信息（如果有）
        '_it_sum': getattr(buffer_obj, '_it_sum', None),
        '_it_min': getattr(buffer_obj, '_it_min', None),
        '_max_priority': getattr(buffer_obj, '_max_priority', None),
    }
    
    # 基础元信息
    result['metadata'] = {
        'buffer_id': buffer_id,
        'storage_len': storage_len,
        'capacity': getattr(buffer_obj, 'capacity', None),
        'storage_type': str(type(storage)),
        'sub_buffer_size': getattr(buffer_obj, 'sub_buffer_size', None),
        'compress_base': getattr(buffer_obj, 'compress_base', None),
    }
    
    # 快速统计：只分析前几个block来估算
    total_compressed_bytes = 0
    total_raw_bytes = 0
    sample_count = min(10, storage_len)  # 只分析前10个用于统计
    
    for idx in range(sample_count):
        try:
            block_data = storage[idx]
            
            for key, value in block_data.items():
                if isinstance(value, bytes):
                    # 压缩后的数据
                    compressed_size = len(value)
                    total_compressed_bytes += compressed_size
                    # 估算原始大小
                    total_raw_bytes += compressed_size * 20  # 保守估计
                elif isinstance(value, np.ndarray):
                    # 未压缩的numpy数组
                    total_compressed_bytes += value.nbytes
                    total_raw_bytes += value.nbytes
                else:
                    # 其他类型
                    size = sys.getsizeof(value)
                    total_compressed_bytes += size
                    total_raw_bytes += size
        except Exception as e:
            logger.warning(f"  分析block {idx} 失败: {e}")
            continue
    
    # 计算统计信息（基于采样推算）
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
    
    logger.info(f"  统计: {result['statistics']['compression_ratio']:.2f}x 压缩率, " 
                f"~{result['statistics']['estimated_total_memory_mb']:.1f} MB 总内存")
    
    return result


def analyze_buffer_dump(dump_file: Path) -> None:
    """分析已dump的buffer文件"""
    
    with open(dump_file, 'rb') as f:
        data = pickle.load(f)
    
    print("\n" + "="*80)
    print(f"📦 Buffer Dump 分析: {dump_file.name}")
    print("="*80)
    
    metadata = data.get('metadata', {})
    raw_storage = data.get('raw_storage', None)
    statistics = data.get('statistics', {})
    
    print(f"\n📋 元信息:")
    print(f"   Buffer类型: {metadata.get('buffer_type', 'Unknown')}")
    
    if metadata.get('buffer_type') == 'MultiAgent':
        print(f"   Policies: {list(metadata.get('policies', {}).keys())}")
        for policy_id, policy_meta in metadata.get('policies', {}).items():
            print(f"\n   Policy: {policy_id}")
            print(f"     存储量: {policy_meta.get('storage_len', 0):,}")
            print(f"     容量: {policy_meta.get('capacity', 0):,}")
            print(f"     Block大小: {policy_meta.get('sub_buffer_size', 'N/A')}")
    else:
        print(f"   存储量: {metadata.get('storage_len', 0):,}")
        print(f"   容量: {metadata.get('capacity', 0):,}")
        print(f"   Block大小: {metadata.get('sub_buffer_size', 'N/A')}")
    
    print(f"\n📊 统计信息:")
    if isinstance(statistics, dict):
        # Single buffer或者各policy的统计
        for key, stats in statistics.items():
            if isinstance(stats, dict):
                print(f"\n   {key}:")
                print(f"     存储块数: {stats.get('storage_len', 0):,}")
                print(f"     压缩率: {stats.get('compression_ratio', 1.0):.2f}x")
                print(f"     平均block大小 (压缩后): {stats.get('avg_compressed_per_block', 0) / 1024:.1f} KB")
                print(f"     平均block大小 (原始): {stats.get('avg_raw_per_block', 0) / 1024:.1f} KB")
                print(f"     估算总内存: {stats.get('estimated_total_memory_mb', 0):.1f} MB")
    
    print(f"\n💾 完整Storage数据:")
    if raw_storage is not None:
        if isinstance(raw_storage, dict):
            # MultiAgent情况
            if 'storage' in raw_storage:
                # Single buffer
                storage_list = raw_storage.get('storage', [])
                print(f"   已保存完整的_storage (类型: {type(storage_list).__name__}, 长度: {len(storage_list):,})")
                if len(storage_list) > 0:
                    first_block = storage_list[0]
                    print(f"   示例block结构:")
                    print(f"     类型: {type(first_block).__name__}")
                    if hasattr(first_block, 'keys'):
                        print(f"     Keys: {list(first_block.keys())}")
            else:
                # MultiAgent
                for policy_id, policy_storage in raw_storage.items():
                    storage_list = policy_storage.get('storage', [])
                    print(f"   {policy_id}: 已保存完整的_storage (长度: {len(storage_list):,})")
        else:
            print(f"   已保存 (类型: {type(raw_storage).__name__})")
    else:
        print(f"   未保存完整storage")
    
    print("\n" + "="*80)

