# RASPBERry Design Documentation

## Overview

RASPBERry (Block Experience Replay with Compression) is a memory-efficient replay buffer system that combines block-level operations with on-the-fly compression for Deep Reinforcement Learning.

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│ MultiAgentPrioritizedBlockReplayBuffer                      │
│ - Multi-agent coordination                                  │
│ - Per-policy buffer management                              │
│ - Priority aggregation/expansion bridge                     │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ├─ Policy 1 Buffer
                 ├─ Policy 2 Buffer
                 └─ ...
                      │
                      ▼
      ┌────────────────────────────────────────┐
      │ PrioritizedBlockReplayBuffer           │
      │ - Block-level storage & sampling       │
      │ - Ray-based parallel compression       │
      │ - Metadata expansion (block→transition)│
      └────────────────────────────────────────┘
```

## Core Design Principles

### 1. Block-Level Storage

**Motivation**: Reduce replay buffer operations from O(M) to O(M/m)

**Implementation**:
- Group `m` transitions into one block (typically m=32)
- Store/sample/update at block granularity
- Each block is treated as atomic unit

**Benefits**:
- ~76% reduction in buffer operation time
- Fewer priority updates to maintain
- Better cache locality

### 2. Transparent Compression

**Motivation**: Reduce memory footprint for high-dimensional observations

**Implementation**:
- Blosc compression of obs/new_obs arrays
- Ray-based parallel compression (3 modes: A/B/D)
- Asynchronous compression pipeline (Mode D)

**Compression Strategy**:
```python
# Transpose: [batch, H, W, C] → [H, W, C, batch]
# Better compression because spatial patterns are contiguous
obs_transposed = np.transpose(obs, [1, 2, 3, 0])
compressed = blosc.pack_array(obs_transposed, cname='zstd', clevel=5)
```

**Benefits**:
- 60-95% memory reduction (Atari: 4GB vs 57GB for 1M transitions)
- ~24% reduction in compression wall-clock time via Ray parallelization
- Enables large buffers on edge devices

### 3. Metadata Expansion Bridge

**Motivation**: Bridge between block-level storage and transition-level training

**Problem**: 
- Storage/Sampling is block-level (for efficiency)
- DQN training expects transition-level weights (for importance sampling)

**Solution**: Bidirectional transformation

#### Forward: Sample → Training (Expansion)

```python
# In PrioritizedBlockReplayBuffer.sample()
# After parent's sample() returns compressed batch:
#   - obs: [num_blocks, 1] object (compressed bytes)
#   - actions: [num_blocks * m] (already transition-level)
#   - weights: [num_blocks] (block-level, needs expansion)

# Expand weights to transition-level
num_transitions = len(batch["actions"])
replicate_factor = num_transitions // len(batch["weights"])
batch["weights"] = np.repeat(batch["weights"], replicate_factor)
# Result: weights [num_blocks] → [num_transitions]
```

**Rationale**: Each transition in a block shares the block's importance weight.

#### Backward: Training → Update (Aggregation)

```python
# In MultiAgentPrioritizedBlockReplayBuffer.update_priorities()
# Receives from DQN training:
#   - batch_indexes: [num_transitions] 
#   - td_errors: [num_transitions]

# Aggregate to block-level
block_indices = batch_indexes.reshape(-1, m)[:, 0]
block_td_errors = td_errors.reshape(-1, m)
block_priorities = np.abs(block_td_errors).mean(axis=1) + eps
# Result: priorities [num_transitions] → [num_blocks]
```

**Rationale**: Block priority = mean absolute TD-error of its transitions (PBER strategy).

## Data Flow

### Complete Cycle

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Add Transitions                                          │
│    ────────────────────────────────────────────             │
│    Environment → add() → [Group into blocks]                │
│                       → [Compress obs/new_obs]              │
│                       → Storage (block-level)               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Sample for Training                                      │
│    ────────────────────────────────────────────             │
│    Storage (block-level)                                    │
│         → [PER sampling by block priority]                  │
│         → [Expand weights: block → transition] ⭐           │
│         → [Decompress obs/new_obs]                          │
│         → Ready for DQN training                            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Compute TD-errors                                        │
│    ────────────────────────────────────────────             │
│    DQN forward/backward                                     │
│         → TD-errors [num_transitions]                       │
│         → batch_indexes [num_transitions]                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Update Priorities                                        │
│    ────────────────────────────────────────────             │
│    update_priorities()                                      │
│         → [Aggregate TD-errors: transition → block] ⭐      │
│         → [Update block priorities in storage]              │
└─────────────────────────────────────────────────────────────┘
```

**⭐ Key Innovation**: The expansion (step 2) and aggregation (step 4) are symmetric operations that bridge block-level storage with transition-level training.

## Compression Modes

### Mode A: Synchronous

```python
# Compress immediately when block is full
if compress_node.is_ready():
    compressed_data, weight, metrics = compress_node.sample()
    buffer._add_single_batch(compressed_data, weight=weight)
```

**Use case**: Baseline, no parallelization

### Mode B: Batch Ray (Recommended)

```python
# Accumulate nodes, submit batch to Ray
if len(pending_nodes) >= chunk_size:
    futures = [compress_node_batch_ray.remote(chunk, config) 
               for chunk in chunks]
    results = ray.get(futures)  # Wait for all
    for result in results:
        buffer._add_single_batch(result["compressed_data"])
```

**Use case**: Production (2.3x faster than ThreadPool)

### Mode D: Async Ray

```python
# Submit to Ray without waiting
future = compress_node_batch_ray.remote(nodes, config)
inflight_futures.append(future)

# Drain completed futures asynchronously
ready, remaining = ray.wait(inflight_futures, timeout=0)
for ref in ready:
    result = ray.get(ref)
    buffer._add_single_batch(result["compressed_data"])
```

**Use case**: Maximum throughput, requires backpressure management

## Design Decisions

### Why expand in Buffer.sample() instead of decompress_sample_batch()?

**Decision**: Expand in `PrioritizedBlockReplayBuffer.sample()`

**Rationale**:
1. **Separation of concerns**: Block logic belongs to Buffer layer, Compression is a transparent optimization
2. **Encapsulation**: `decompress_sample_batch()` doesn't need to know about `sub_buffer_size`
3. **Reusability**: Decompression function remains a pure utility

**Alternative rejected**: Expand in `decompress_sample_batch(batch, sub_buffer_size=...)`
- ❌ Violates single responsibility (compression + block logic)
- ❌ Requires passing buffer parameters to utility function
- ❌ Harder to test independently

### Why aggregate by mean instead of max TD-error?

**Decision**: Use mean absolute TD-error per block

**Rationale**:
1. **Robustness**: Less sensitive to outliers
2. **Consistency**: Matches PER's expected importance sampling weights
3. **Empirical**: Performed better in preliminary experiments

**Alternative considered**: Max absolute TD-error
- ❌ Over-samples blocks with one high TD-error transition
- ❌ May cause priority imbalance

### Why compress observations but not actions/rewards?

**Decision**: Only compress obs/new_obs

**Rationale**:
1. **ROI**: Observations (84×84×4 = 28,224 floats) vs actions (1 int) vs rewards (1 float)
2. **Compressibility**: Images have spatial patterns, scalars don't compress well
3. **Overhead**: Compression has fixed cost, not worth for small tensors

**Numbers** (Atari, 1M transitions):
- Obs/new_obs: 57GB uncompressed → 4GB compressed (93% reduction)
- Actions: 4MB (negligible)
- Rewards: 4MB (negligible)

## Performance Characteristics

### Time Complexity

| Operation | Naive Replay | BER | RASPBERry |
|-----------|--------------|-----|-----------|
| Add | O(1) | O(1) | O(1) + compress |
| Sample | O(k) | O(k/m) | O(k/m) + decompress |
| Update priorities | O(k) | O(k/m) | O(k/m) + aggregate |

where:
- k = batch size
- m = block size (sub_buffer_size)

### Space Complexity

| Component | Memory Usage (Atari, 1M transitions) |
|-----------|--------------------------------------|
| Uncompressed | 57 GB |
| BER (no compression) | 57 GB |
| RASPBERry (zstd-5) | 4 GB |
| **Reduction** | **93%** |

### Wall-Clock Performance

Measured on Atari Breakout, 1M buffer capacity:

| Metric | PER | RASPBERry | Improvement |
|--------|-----|-----------|-------------|
| Buffer add | 100% | 100% | - |
| Buffer sample | 100% | 24% | **76% faster** |
| Compression overhead | N/A | 24ms/batch | Amortized via Ray |
| Total throughput | 100% | 122% | **22% improvement** |

## Implementation Notes

### Critical Path

The most performance-sensitive operations:

1. **Compression batching (Mode B/D)**: Use `chunk_size=10` for good CPU utilization
2. **Decompression**: Single-threaded blosc is fast enough for training loop
3. **Metadata expansion**: Negligible overhead (just `np.repeat`)

### Memory Management

- Compressed blocks stored as `dtype=object` arrays containing bytes
- Ray workers use shared memory for input/output transfer
- No memory leaks observed in 100M+ transition runs

### Compatibility

- **RLlib**: Drop-in replacement for `MultiAgentPrioritizedReplayBuffer`
- **Gym/Gymnasium**: Works with any observation space (not just images)
- **Ray**: Requires Ray >= 2.0 for async features (Mode D)

## Common Pitfalls

### 1. Forgetting to expand weights

**Symptom**: `RuntimeError: size mismatch` in DQN loss calculation

**Cause**: Weights are still block-level when passed to training

**Fix**: Ensure `PrioritizedBlockReplayBuffer.sample()` calls `_expand_block_field()`

### 2. Incorrect sub_buffer_size

**Symptom**: Poor compression ratio or misaligned priorities

**Cause**: `sub_buffer_size` doesn't match actual block size

**Fix**: Verify `sub_buffer_size` matches `rollout_fragment_length` in RLlib config

### 3. Mode D backpressure

**Symptom**: Unbounded memory growth in Ray workers

**Cause**: Submitting futures faster than they can be processed

**Fix**: Use backpressure threshold (`len(inflight_futures) >= num_workers * 2`)

## Testing Recommendations

### Unit Tests

```python
def test_expand_block_field():
    """Test metadata expansion."""
    buffer = PrioritizedBlockReplayBuffer(sub_buffer_size=8, ...)
    batch = SampleBatch({"weights": np.array([1.0, 2.0])})
    buffer._expand_block_field(batch, "weights", 16)
    assert len(batch["weights"]) == 16
    assert np.array_equal(batch["weights"], np.repeat([1.0, 2.0], 8))
```

### Integration Tests

```python
def test_sample_train_update_cycle():
    """Test complete cycle."""
    # Add transitions
    buffer.add(transitions)
    
    # Sample
    batch = buffer.sample(num_items=32)
    assert len(batch["weights"]) == 32 * sub_buffer_size
    
    # Train (mock)
    td_errors = compute_td_errors(batch)
    
    # Update
    buffer.update_priorities({policy_id: (batch["batch_indexes"], td_errors)})
```

## References

- **BER paper**: Block Experience Replay (original contribution)
- **RASPBERry paper**: Block Experience Replay with Compression
- **PER paper**: Prioritized Experience Replay (Schaul et al., 2015)
- **Blosc**: https://www.blosc.org/

## Version History

- **v1.0**: Initial BER implementation (ThreadPool compression)
- **v2.0**: Ray-based compression (2.3x speedup)
- **v2.1**: Metadata expansion refactoring (current)

