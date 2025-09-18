from benchmark import Bench, BenchConfig, BenchId, Bencher, keep
from hyperloglog import HyperLogLog


fn hash_int(x: Int) -> Int:
    """Simple hash function for integers."""
    var h = x
    h = ((h >> 16) ^ h) * 0x45D9F3B
    h = ((h >> 16) ^ h) * 0x45D9F3B
    h = (h >> 16) ^ h
    return h

@parameter
fn benchmark_add_sparse(mut b: Bencher) raises:
    """Benchmark adding elements while HLL remains in sparse mode."""
    var hll = HyperLogLog[14]()
    @always_inline
    @parameter
    fn call_fn():
        for i in range(1000):  # Small enough to stay sparse
            hll.add_hash(hash_int(i))
        keep(hll.registers._data)  # Prevent optimization away

    b.iter[call_fn]()


@parameter
fn benchmark_add_dense(mut b: Bencher) raises:
    """Benchmark adding elements in dense mode."""
    var hll = HyperLogLog[14]()
    @always_inline
    @parameter
    fn call_fn():
        for i in range(100_000):  # Large enough to trigger dense mode
            hll.add_hash(hash_int(i))
        keep(hll.registers._data)  # Prevent optimization away

    b.iter[call_fn]()
    


@parameter
fn benchmark_cardinality_sparse(mut b: Bencher) raises:
    var hll = HyperLogLog[14]()
    for i in range(1000):
        hll.add_hash(hash_int(i))

    @always_inline
    @parameter
    fn call_fn():
        var c = hll.cardinality()
        keep(c)  # Prevent optimization away

    b.iter[call_fn]()


@parameter
fn benchmark_cardinality_dense(mut b: Bencher) raises:
    var hll = HyperLogLog[14]()
    for i in range(100_000):
        hll.add_hash(hash_int(i))

    @always_inline
    @parameter
    fn call_fn():
        var c = hll.cardinality()
        keep(c)  # Prevent optimization away

    b.iter[call_fn]()

@parameter
fn benchmark_merge_sparse(mut b: Bencher) raises:
    """Benchmark merging two sparse HLLs."""
    var hll1 = HyperLogLog[14]()
    var hll2 = HyperLogLog[14]()
    for i in range(1000):
        hll1.add_hash(hash_int(i))
        hll2.add_hash(hash_int(i + 1000))

    @always_inline
    @parameter
    fn call_fn() raises:
        hll1.merge(hll2)
    
    b.iter[call_fn]()
    _ = hll1
    _ = hll2

@parameter
fn benchmark_merge_dense(mut b: Bencher) raises:
    """Benchmark merging two dense HLLs."""
    var hll1 = HyperLogLog[14]()
    var hll2 = HyperLogLog[14]()
    for i in range(100_000):
        hll1.add_hash(hash_int(i))
        hll2.add_hash(hash_int(i + 100_000))
    @always_inline
    @parameter
    fn call_fn() raises:
        hll1.merge(hll2)
    
    b.iter[call_fn]()
    _ = hll1
    _ = hll2


fn main() raises:
    var m = Bench(
        BenchConfig(
            num_repetitions=5,
        )
    )
    m.bench_function[benchmark_add_sparse](
        BenchId("benchmark_add_sparse 1000 elements"),
    )
    m.bench_function[benchmark_add_dense](
        BenchId("benchmark_add_dense 100_000 elements"),
    )

    m.bench_function[benchmark_cardinality_sparse](
        BenchId("benchmark_cardinality_sparse 1000 elements"),
    )
    m.bench_function[benchmark_cardinality_dense](
        BenchId("benchmark_cardinality_dense 100_000 elements"),
    )

    m.bench_function[benchmark_merge_sparse](
        BenchId("benchmark_merge_sparse 1000 elements"),
    )
    m.bench_function[benchmark_merge_dense](
        BenchId("benchmark_merge_dense 100_000 elements"),
    )

    m.dump_report()
