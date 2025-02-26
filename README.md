# Mojo HyperLogLog

A Mojo implementation of HyperLogLog - a probabilistic data structure for counting unique elements with minimal memory usage.

## Features

- Configurable precision (4-16)
- Perfect accuracy for small sets (up to 2^(p-3) elements)
- Automatic optimization for small and large sets
- Merge support for combining counters
- Serialization for saving/loading
- High accuracy (1-2% error rate for large sets)

## Usage

```mojo
from hyperloglog import HyperLogLog
from murmur import murmur3_64

# Create counter
var hll = HyperLogLog(14)  # precision = 14
                          # exact counting up to 2048 elements

# Add values
for i in range(1000):
    hll.add_hash(Int(murmur3_64(i)))

# Get count
var count = hll.cardinality()

# Merge counters
var hll2 = HyperLogLog(14)
hll.merge(hll2)
```

## Memory Usage

Memory depends on precision (p):
- p = 14: ~16KB
- p = 15: ~32KB  
- p = 16: ~64KB

## License

MIT License