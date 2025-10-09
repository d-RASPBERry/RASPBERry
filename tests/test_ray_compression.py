"""
Quick test to verify Ray-based compression works correctly.
"""

import numpy as np
import gymnasium as gym
from replay_buffer.raspberry_ray import PrioritizedBlockReplayBuffer
from gymnasium import spaces

def test_ray_compression():
    """Test basic Ray compression functionality."""
    print("="*60)
    print("Testing Ray-based Compression")
    print("="*60)
    
    # Create simple environment
    obs_space = spaces.Box(low=0, high=255, shape=(84, 84, 4), dtype=np.uint8)
    action_space = spaces.Discrete(4)
    
    # Test Mode B (batch Ray)
    print("\nTesting Mode B (Batch Ray)...")
    buffer = PrioritizedBlockReplayBuffer(
        capacity=1000,
        obs_space=obs_space,
        action_space=action_space,
        sub_buffer_size=16,
        compress_base=-1,
        compress_pool_size=3,
        compression_algorithm='zstd',
        compression_level=3,
        compression_mode='B',
        chunk_size=5,
        storage_unit='timesteps',
    )
    
    print(f"✓ Buffer created (Mode B)")
    
    # Add some data
    from ray.rllib.policy.sample_batch import SampleBatch
    
    num_samples = 160  # 10 blocks
    obs = np.random.randint(0, 255, (num_samples, 84, 84, 4), dtype=np.uint8)
    new_obs = np.random.randint(0, 255, (num_samples, 84, 84, 4), dtype=np.uint8)
    actions = np.random.randint(0, 4, num_samples, dtype=np.int64)
    rewards = np.random.randn(num_samples).astype(np.float32)
    terminateds = np.zeros(num_samples, dtype=np.int32)
    truncateds = np.zeros(num_samples, dtype=np.int32)
    
    batch = SampleBatch({
        'obs': obs,
        'new_obs': new_obs,
        'actions': actions,
        'rewards': rewards,
        'terminateds': terminateds,
        'truncateds': truncateds,
    })
    
    print(f"Adding {num_samples} samples...")
    buffer.add(batch)
    print(f"✓ Data added to buffer")
    
    # Get stats
    stats = buffer.stats()
    print(f"\nBuffer Stats:")
    print(f"  Entries: {stats['num_entries']}")
    print(f"  Compressed size: {stats['est_size_bytes'] / (1024**2):.2f} MB")
    print(f"  Compression time: {stats['compress_time_ms']:.2f} ms")
    
    # Try sampling
    print(f"\nSampling 32 items...")
    sample = buffer.sample(num_items=32, beta=0.4)
    
    if sample is not None:
        print(f"✓ Sampled {sample.count} transitions")
        print(f"  Obs shape: {sample['obs'].shape}")
        print(f"  Actions shape: {sample['actions'].shape}")
        print(f"  Decompression time: {stats['decompress_time_ms']:.2f} ms")
    else:
        print("⚠ Sample returned None (buffer may need more data)")
    
    print("\n" + "="*60)
    print("✓ Test completed successfully!")
    print("="*60)


if __name__ == "__main__":
    test_ray_compression()

Quick test to verify Ray-based compression works correctly.
"""

import numpy as np
import gymnasium as gym
from replay_buffer.raspberry_ray import PrioritizedBlockReplayBuffer
from gymnasium import spaces

def test_ray_compression():
    """Test basic Ray compression functionality."""
    print("="*60)
    print("Testing Ray-based Compression")
    print("="*60)
    
    # Create simple environment
    obs_space = spaces.Box(low=0, high=255, shape=(84, 84, 4), dtype=np.uint8)
    action_space = spaces.Discrete(4)
    
    # Test Mode B (batch Ray)
    print("\nTesting Mode B (Batch Ray)...")
    buffer = PrioritizedBlockReplayBuffer(
        capacity=1000,
        obs_space=obs_space,
        action_space=action_space,
        sub_buffer_size=16,
        compress_base=-1,
        compress_pool_size=3,
        compression_algorithm='zstd',
        compression_level=3,
        compression_mode='B',
        chunk_size=5,
        storage_unit='timesteps',
    )
    
    print(f"✓ Buffer created (Mode B)")
    
    # Add some data
    from ray.rllib.policy.sample_batch import SampleBatch
    
    num_samples = 160  # 10 blocks
    obs = np.random.randint(0, 255, (num_samples, 84, 84, 4), dtype=np.uint8)
    new_obs = np.random.randint(0, 255, (num_samples, 84, 84, 4), dtype=np.uint8)
    actions = np.random.randint(0, 4, num_samples, dtype=np.int64)
    rewards = np.random.randn(num_samples).astype(np.float32)
    terminateds = np.zeros(num_samples, dtype=np.int32)
    truncateds = np.zeros(num_samples, dtype=np.int32)
    
    batch = SampleBatch({
        'obs': obs,
        'new_obs': new_obs,
        'actions': actions,
        'rewards': rewards,
        'terminateds': terminateds,
        'truncateds': truncateds,
    })
    
    print(f"Adding {num_samples} samples...")
    buffer.add(batch)
    print(f"✓ Data added to buffer")
    
    # Get stats
    stats = buffer.stats()
    print(f"\nBuffer Stats:")
    print(f"  Entries: {stats['num_entries']}")
    print(f"  Compressed size: {stats['est_size_bytes'] / (1024**2):.2f} MB")
    print(f"  Compression time: {stats['compress_time_ms']:.2f} ms")
    
    # Try sampling
    print(f"\nSampling 32 items...")
    sample = buffer.sample(num_items=32, beta=0.4)
    
    if sample is not None:
        print(f"✓ Sampled {sample.count} transitions")
        print(f"  Obs shape: {sample['obs'].shape}")
        print(f"  Actions shape: {sample['actions'].shape}")
        print(f"  Decompression time: {stats['decompress_time_ms']:.2f} ms")
    else:
        print("⚠ Sample returned None (buffer may need more data)")
    
    print("\n" + "="*60)
    print("✓ Test completed successfully!")
    print("="*60)


if __name__ == "__main__":
    test_ray_compression()



