"""
Comprehensive 18-Case Benchmark: ThreadPool vs Ray
===================================================

Test Matrix:
- Backends: ThreadPool, Ray (2)
- Modes: A, B, D (3)
- Parameter Groups: (3)
  * Group 1: chunk_size=400, workers=10
  * Group 2: chunk_size=200, workers=10
  * Group 3: chunk_size=100, workers=5

Total: 2 × 3 × 3 = 18 test cases

Configuration:
- Environment: Atari Breakout
- Data volume: 100,000 transitions
- sub_buffer_size: 16
- batch_size: 256
- Compression: zstd level 5

All tests run sequentially. If any test fails, execution terminates.
"""

import time
import numpy as np
import gymnasium as gym
import psutil
import os
import sys
import traceback
from typing import Dict, Any, List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ray.rllib.policy.sample_batch import SampleBatch


def create_atari_env():
    """Create Atari Breakout environment."""
    env = gym.make("ALE/Breakout-v5", frameskip=1)
    return env


def collect_samples(num_transitions: int = 100000) -> Tuple:
    """
    Collect samples from Atari Breakout.
    
    Returns:
        (obs, new_obs, actions, rewards, terminateds, truncateds)
    """
    print(f"\n{'=' * 70}")
    print(f"Collecting {num_transitions:,} transitions from Atari Breakout...")
    print(f"{'=' * 70}")

    env = create_atari_env()

    obs_list = []
    new_obs_list = []
    actions_list = []
    rewards_list = []
    terminateds_list = []
    truncateds_list = []

    obs, _ = env.reset()
    collected = 0

    while collected < num_transitions:
        action = env.action_space.sample()
        new_obs, reward, terminated, truncated, info = env.step(action)

        obs_list.append(obs)
        new_obs_list.append(new_obs)
        actions_list.append(action)
        rewards_list.append(reward)
        terminateds_list.append(int(terminated))
        truncateds_list.append(int(truncated))

        collected += 1
        if collected % 20000 == 0:
            print(f"  Collected {collected:,}/{num_transitions:,} transitions...")

        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = new_obs

    env.close()

    # Convert to arrays
    obs_array = np.array(obs_list, dtype=np.uint8)
    new_obs_array = np.array(new_obs_list, dtype=np.uint8)
    actions_array = np.array(actions_list, dtype=np.int64)
    rewards_array = np.array(rewards_list, dtype=np.float32)
    terminateds_array = np.array(terminateds_list, dtype=np.int32)
    truncateds_array = np.array(truncateds_list, dtype=np.int32)

    print(f"\n✓ Collection complete!")
    print(f"  Shape: {obs_array.shape}")
    print(f"  Memory: {obs_array.nbytes / (1024 ** 3):.2f} GB")

    return (obs_array, new_obs_array, actions_array,
            rewards_array, terminateds_array, truncateds_array)


def benchmark_strategy(
        test_id: int,
        strategy_name: str,
        buffer_class,
        data: Tuple,
        compression_mode: str,
        num_workers: int,
        chunk_size: int
) -> Dict[str, Any]:
    """
    Benchmark a specific strategy configuration.
    
    Args:
        test_id: Test number (1-18)
        strategy_name: e.g., "ThreadPool + Mode B + W10 + C200"
        buffer_class: Buffer class to use
        data: Tuple of (obs, new_obs, actions, rewards, terminateds, truncateds)
        compression_mode: "A", "B", or "D"
        num_workers: Number of workers/threads
        chunk_size: Chunk size for batch processing
        
    Raises:
        Exception: If the test fails, to terminate execution
    """
    print(f"\n{'=' * 70}")
    print(f"Test {test_id}/18: {strategy_name}")
    print(f"{'=' * 70}")
    print(f"  Backend: {'Ray' if 'Ray' in strategy_name else 'ThreadPool'}")
    print(f"  Mode: {compression_mode}")
    print(f"  Workers: {num_workers}")
    print(f"  Chunk size: {chunk_size}")
    print(f"{'=' * 70}")

    try:
        obs, new_obs, actions, rewards, terminateds, truncateds = data
        num_transitions = len(obs)

        # Create environment for space info
        env = create_atari_env()
        obs_space = env.observation_space
        action_space = env.action_space
        env.close()

        # Create buffer
        print(f"\nCreating buffer...")
        buffer = buffer_class(
            obs_space=obs_space,
            action_space=action_space,
            sub_buffer_size=16,
            capacity=num_transitions,
            storage_unit='timesteps',
            compress_base=-1,
            compress_pool_size=num_workers,
            compression_algorithm='zstd',
            compression_level=5,
            compression_mode=compression_mode,
            chunk_size=chunk_size,
        )
        print(f"✓ Buffer created")

        # Warm up
        print(f"\nWarming up...")
        warmup_batch = SampleBatch({
            'obs': obs[:500],
            'new_obs': new_obs[:500],
            'actions': actions[:500],
            'rewards': rewards[:500],
            'terminateds': terminateds[:500],
            'truncateds': truncateds[:500],
        })
        buffer.add(warmup_batch)
        if len(buffer._storage) > 0:
            _ = buffer.sample(num_items=256, beta=0.4)
        print(f"✓ Warmup complete")

        # Measure resources before
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 ** 3)
        cpu_times_before = process.cpu_times()

        # === Phase 1: Add (Compression) ===
        print(f"\nPhase 1: Adding {num_transitions:,} transitions...")
        t_add_start = time.time()

        # Add in chunks
        add_chunk_size = 5000
        for i in range(0, num_transitions, add_chunk_size):
            end_idx = min(i + add_chunk_size, num_transitions)
            batch = SampleBatch({
                'obs': obs[i:end_idx],
                'new_obs': new_obs[i:end_idx],
                'actions': actions[i:end_idx],
                'rewards': rewards[i:end_idx],
                'terminateds': terminateds[i:end_idx],
                'truncateds': truncateds[i:end_idx],
            })
            buffer.add(batch)

            if (end_idx) % 20000 == 0:
                print(f"  Added {end_idx:,}/{num_transitions:,}...")

        t_add_end = time.time()
        add_time = t_add_end - t_add_start
        print(f"✓ Phase 1 complete: {add_time:.2f}s")

        # === Phase 2: Sample (Decompression) ===
        print(f"\nPhase 2: Sampling 500 batches...")
        t_sample_start = time.time()

        num_samples = 500
        sample_times = []

        for i in range(num_samples):
            t0 = time.time()
            _ = buffer.sample(num_items=256, beta=0.4)
            sample_times.append(time.time() - t0)

            if (i + 1) % 100 == 0:
                print(f"  Sampled {i + 1}/{num_samples}...")

        t_sample_end = time.time()
        sample_time_total = t_sample_end - t_sample_start
        sample_time_avg = np.mean(sample_times)
        print(f"✓ Phase 2 complete: {sample_time_total:.2f}s (avg: {sample_time_avg * 1000:.3f}ms/batch)")

        # Measure resources after
        cpu_times_after = process.cpu_times()
        mem_after = process.memory_info().rss / (1024 ** 3)

        # Get buffer stats
        stats = buffer.stats()

        # Calculate metrics
        total_time = add_time + sample_time_total
        total_cpu_time = (cpu_times_after.user - cpu_times_before.user) + \
                         (cpu_times_after.system - cpu_times_before.system)

        uncompressed_size_mb = (obs.nbytes + new_obs.nbytes) / (1024 ** 2)
        compressed_size_mb = stats.get('est_size_bytes', 0) / (1024 ** 2)
        compression_ratio = uncompressed_size_mb / compressed_size_mb if compressed_size_mb > 0 else 0

        results = {
            'test_id': test_id,
            'strategy': strategy_name,
            'backend': 'Ray' if 'Ray' in strategy_name else 'ThreadPool',
            'mode': compression_mode,
            'num_workers': num_workers,
            'chunk_size': chunk_size,
            'num_transitions': num_transitions,
            'add_time_s': add_time,
            'sample_time_total_s': sample_time_total,
            'sample_time_avg_ms': sample_time_avg * 1000,
            'total_time_s': total_time,
            'cpu_time_s': total_cpu_time,
            'cpu_efficiency': total_cpu_time / total_time,
            'throughput_add_mb_s': uncompressed_size_mb / add_time,
            'uncompressed_size_mb': uncompressed_size_mb,
            'compressed_size_mb': compressed_size_mb,
            'compression_ratio': compression_ratio,
            'mem_delta_gb': mem_after - mem_before,
            'compress_time_ms': stats.get('metrics', {}).get('compress_time_ms', stats.get('compress_time_ms', 0)),
            'decompress_time_ms': stats.get('metrics', {}).get('decompress_time_ms', stats.get('decompress_time_ms', 0)),
            'success': True,
        }

        print(f"\n{'=' * 70}")
        print(f"✓ Test {test_id} PASSED")
        print(f"  Add: {add_time:.2f}s ({results['throughput_add_mb_s']:.2f} MB/s)")
        print(f"  Sample: {sample_time_total:.2f}s ({sample_time_avg * 1000:.3f}ms/batch)")
        print(f"  Total: {total_time:.2f}s")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        print(f"{'=' * 70}")

        return results

    except Exception as e:
        print(f"\n{'=' * 70}")
        print(f"❌ Test {test_id} FAILED")
        print(f"Error: {str(e)}")
        print(f"{'=' * 70}")
        traceback.print_exc()
        raise  # Re-raise to terminate execution


def print_comparison(results_list: List[Dict[str, Any]]):
    """Print comprehensive comparison table."""
    print(f"\n{'=' * 90}")
    print(f"COMPREHENSIVE BENCHMARK RESULTS ({len(results_list)} tests)")
    print(f"{'=' * 90}")

    # Overall comparison
    print(f"\n{'Test':<6} {'Strategy':<30} {'Mode':<6} {'W':<4} {'C':<5} {'Add(s)':<8} {'Sample(s)':<10} {'Total(s)':<9} {'Speedup'}")
    print(f"{'-' * 90}")

    baseline_time = results_list[0]['total_time_s'] if results_list else None

    for r in results_list:
        speedup = f"{baseline_time / r['total_time_s']:.2f}x" if baseline_time and r != results_list[0] else "baseline"

        print(f"{r['test_id']:<6} {r['strategy']:<30} {r['mode']:<6} "
              f"{r['num_workers']:<4} {r['chunk_size']:<5} "
              f"{r['add_time_s']:<8.2f} {r['sample_time_total_s']:<10.2f} "
              f"{r['total_time_s']:<9.2f} {speedup}")

    # Group by backend
    print(f"\n{'=' * 90}")
    print(f"BACKEND COMPARISON (ThreadPool vs Ray)")
    print(f"{'=' * 90}")

    threadpool_results = [r for r in results_list if r['backend'] == 'ThreadPool']
    ray_results = [r for r in results_list if r['backend'] == 'Ray']

    if threadpool_results and ray_results:
        tp_avg = np.mean([r['total_time_s'] for r in threadpool_results])
        ray_avg = np.mean([r['total_time_s'] for r in ray_results])

        print(f"\nThreadPool average: {tp_avg:.2f}s")
        print(f"Ray average: {ray_avg:.2f}s")
        print(f"Ray speedup: {tp_avg / ray_avg:.2f}x")

    # Group by mode
    print(f"\n{'=' * 90}")
    print(f"MODE COMPARISON")
    print(f"{'=' * 90}")

    for mode in ['A', 'B', 'D']:
        mode_results = [r for r in results_list if r['mode'] == mode]
        if mode_results:
            avg_time = np.mean([r['total_time_s'] for r in mode_results])
            print(f"Mode {mode} average: {avg_time:.2f}s ({len(mode_results)} tests)")

    # Group by parameter group
    print(f"\n{'=' * 90}")
    print(f"PARAMETER GROUP COMPARISON")
    print(f"{'=' * 90}")

    param_groups = {
        (400, 10): [],
        (200, 10): [],
        (100, 5): [],
    }

    for r in results_list:
        key = (r['chunk_size'], r['num_workers'])
        if key in param_groups:
            param_groups[key].append(r)

    for (chunk, workers), group_results in param_groups.items():
        if group_results:
            avg_time = np.mean([r['total_time_s'] for r in group_results])
            print(f"Chunk={chunk}, Workers={workers}: {avg_time:.2f}s ({len(group_results)} tests)")

    # Compression stats
    print(f"\n{'=' * 90}")
    print(f"COMPRESSION EFFICIENCY")
    print(f"{'=' * 90}")
    print(f"{'Test':<6} {'Strategy':<30} {'Ratio':<8} {'Comp Time(ms)':<15} {'Decomp Time(ms)'}")
    print(f"{'-' * 90}")

    for r in results_list:
        print(f"{r['test_id']:<6} {r['strategy']:<30} "
              f"{r['compression_ratio']:<8.2f} {r['compress_time_ms']:<15.2f} "
              f"{r['decompress_time_ms']:.2f}")

    print(f"\n{'=' * 90}")


def main():
    """Run comprehensive 18-case benchmark."""
    print("=" * 90)
    print("COMPREHENSIVE 18-CASE BENCHMARK: ThreadPool vs Ray")
    print("=" * 90)
    print(f"Data volume: 100,000 transitions")
    print(f"sub_buffer_size: 16")
    print(f"batch_size: 256")
    print(f"Compression: zstd level 5")
    print(f"\nTest Matrix:")
    print(f"  Backends: ThreadPool, Ray (2)")
    print(f"  Modes: A, B, D (3)")
    print(f"  Parameter Groups:")
    print(f"    Group 1: chunk_size=400, workers=10")
    print(f"    Group 2: chunk_size=200, workers=10")
    print(f"    Group 3: chunk_size=100, workers=5")
    print(f"  Total: 2 × 3 × 3 = 18 test cases")
    print("=" * 90)

    # Import buffer classes
    from replay_buffer.raspberry import PrioritizedBlockReplayBuffer as ThreadPoolBuffer
    from replay_buffer.raspberry_ray import PrioritizedBlockReplayBuffer as RayBuffer

    # Step 1: Collect samples (once for all tests)
    data = collect_samples(num_transitions=100000)

    # Step 2: Define test matrix
    test_matrix = []
    test_id = 1

    # Generate all 18 test cases
    for backend_name, buffer_class in [
        ("ThreadPool", ThreadPoolBuffer),
        ("Ray", RayBuffer)
    ]:
        for mode in ['A', 'B', 'D']:
            for chunk_size, num_workers in [
                (400, 10),  # Group 1
                (200, 10),  # Group 2
                (100, 5),  # Group 3
            ]:
                strategy_name = f"{backend_name} + Mode {mode} + W{num_workers} + C{chunk_size}"
                test_matrix.append({
                    'id': test_id,
                    'name': strategy_name,
                    'buffer_class': buffer_class,
                    'mode': mode,
                    'workers': num_workers,
                    'chunk': chunk_size,
                })
                test_id += 1

    # Step 3: Run all tests sequentially
    results = []
    failed_tests = []

    for test_config in test_matrix:
        try:
            print(f"\n\n{'#' * 90}")
            print(f"Starting Test {test_config['id']}/18")
            print(f"{'#' * 90}")

            result = benchmark_strategy(
                test_id=test_config['id'],
                strategy_name=test_config['name'],
                buffer_class=test_config['buffer_class'],
                data=data,
                compression_mode=test_config['mode'],
                num_workers=test_config['workers'],
                chunk_size=test_config['chunk']
            )
            results.append(result)

            # Small delay between tests
            if test_config['id'] < len(test_matrix):
                print(f"\nWaiting 3 seconds before next test...")
                time.sleep(3)

        except Exception as e:
            failed_tests.append({
                'id': test_config['id'],
                'name': test_config['name'],
                'error': str(e)
            })
            print(f"\n{'=' * 90}")
            print(f"EXECUTION TERMINATED DUE TO TEST FAILURE")
            print(f"Failed test: {test_config['id']} - {test_config['name']}")
            print(f"Error: {str(e)}")
            print(f"{'=' * 90}")
            break

    # Step 4: Print results
    if results:
        print_comparison(results)

    # Step 5: Summary
    print(f"\n{'=' * 90}")
    print(f"FINAL SUMMARY")
    print(f"{'=' * 90}")
    print(f"Total tests planned: {len(test_matrix)}")
    print(f"Tests completed: {len(results)}")
    print(f"Tests failed: {len(failed_tests)}")

    if failed_tests:
        print(f"\nFailed tests:")
        for ft in failed_tests:
            print(f"  Test {ft['id']}: {ft['name']}")
            print(f"    Error: {ft['error']}")

    print(f"{'=' * 90}")

    # Exit with error code if any test failed
    if failed_tests:
        sys.exit(1)
    else:
        print(f"\n✓ ALL 18 TESTS PASSED!")
        sys.exit(0)


if __name__ == "__main__":
    main()
