# TLA+ Specifications for Silicon Memory

This directory contains TLA+ specifications for formal verification of the Silicon Memory system.

## Specifications

- `MemorySystem.tla` - Main memory system with all layers
- `BeliefSystem.tla` - Belief confidence and contradiction handling
- `TemporalDecay.tla` - Temporal decay of confidence over time
- `WorkingMemory.tla` - Working memory with TTL eviction

## Running the Specifications

### Using TLC Model Checker

```bash
# Install TLA+ Toolbox or use command line tools
tlc MemorySystem.tla
tlc BeliefSystem.tla
tlc TemporalDecay.tla
tlc WorkingMemory.tla
```

### Properties Verified

1. **Safety Properties**
   - Beliefs never have negative confidence
   - Confidence never exceeds 1.0
   - Contradictions are detected
   - TTL entries expire correctly

2. **Liveness Properties**
   - Eventually all experiences are processed
   - Decay eventually reduces old beliefs
   - Expired entries are eventually removed

## Model Checking Configuration

Each `.cfg` file contains:
- Initial state constraints
- Invariants to check
- Temporal properties
