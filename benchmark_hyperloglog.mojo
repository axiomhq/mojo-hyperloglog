from benchmark import benchmark, Unit
from hyperloglog import HyperLogLog


fn hash_int(x: Int) -> Int:
    """Simple hash function for integers."""
    var h = x
    h = ((h >> 16) ^ h) * 0x45D9F3B
    h = ((h >> 16) ^ h) * 0x45D9F3B
    h = (h >> 16) ^ h
    return h


fn benchmark_add_sparse() raises:
    """Benchmark adding elements while HLL remains in sparse mode."""
    var hll = HyperLogLog(14)
    for i in range(1000):  # Small enough to stay sparse
        hll.add_hash(hash_int(i))


fn benchmark_add_dense() raises:
    """Benchmark adding elements in dense mode."""
    var hll = HyperLogLog(14)
    for i in range(100_000):  # Large enough to trigger dense mode
        hll.add_hash(hash_int(i))


fn benchmark_cardinality_sparse() raises:
    """Benchmark cardinality estimation in sparse mode."""
    var hll = HyperLogLog(14)
    for i in range(1000):
        hll.add_hash(hash_int(i))
    _ = hll.cardinality()


fn benchmark_cardinality_dense() raises:
    """Benchmark cardinality estimation in dense mode."""
    var hll = HyperLogLog(14)
    for i in range(100_000):
        hll.add_hash(hash_int(i))
    _ = hll.cardinality()


fn benchmark_merge_sparse() raises:
    """Benchmark merging two sparse HLLs."""
    var hll1 = HyperLogLog(14)
    var hll2 = HyperLogLog(14)
    for i in range(1000):
        hll1.add_hash(hash_int(i))
        hll2.add_hash(hash_int(i + 1000))
    hll1.merge(hll2)


fn benchmark_merge_dense() raises:
    """Benchmark merging two dense HLLs."""
    var hll1 = HyperLogLog(14)
    var hll2 = HyperLogLog(14)
    for i in range(100_000):
        hll1.add_hash(hash_int(i))
        hll2.add_hash(hash_int(i + 100_000))
    hll1.merge(hll2)


fn main() raises:
    print("Running HyperLogLog Benchmarks...")
    print("\nSparse Mode Operations:")
    print("-----------------------")

    print("\nAdding elements (sparse):")
    var report = benchmark.run[benchmark_add_sparse]()
    report.print(Unit.ms)

    print("\nCardinality estimation (sparse):")
    report = benchmark.run[benchmark_cardinality_sparse]()
    report.print(Unit.ms)

    print("\nMerging HLLs (sparse):")
    report = benchmark.run[benchmark_merge_sparse]()
    report.print(Unit.ms)

    print("\nDense Mode Operations:")
    print("---------------------")

    print("\nAdding elements (dense):")
    report = benchmark.run[benchmark_add_dense]()
    report.print(Unit.ms)

    print("\nCardinality estimation (dense):")
    report = benchmark.run[benchmark_cardinality_dense]()
    report.print(Unit.ms)

    print("\nMerging HLLs (dense):")
    report = benchmark.run[benchmark_merge_dense]()
    report.print(Unit.ms)
