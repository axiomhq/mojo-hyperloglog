from testing import assert_equal, assert_true
from hyperloglog import HyperLogLog
from murmur import murmur3_64


fn test_initialization() raises:
    """Test HyperLogLog initialization with different precision values."""
    # Test valid precision
    var hll = HyperLogLog(14)
    assert_equal(hll.precision, 14)
    assert_equal(hll.max_zeros, 50)  # 64 - 14
    assert_true(hll.is_sparse)

    # Test minimum precision
    var hll_min = HyperLogLog(4)
    assert_equal(hll_min.precision, 4)
    assert_equal(hll_min.max_zeros, 60)  # 64 - 4

    # Test maximum precision
    var hll_max = HyperLogLog(16)
    assert_equal(hll_max.precision, 16)
    assert_equal(hll_max.max_zeros, 48)  # 64 - 16


fn test_sparse_to_dense_conversion() raises:
    """Test conversion from sparse to dense representation."""
    var hll = HyperLogLog(4)  # Small precision for easier testing
    var num_registers = 1 << 4

    # Add elements until conversion threshold
    for i in range(num_registers + 1):
        hll.add_hash(Int(murmur3_64(i)))

    # Should have converted to dense
    assert_true(not hll.is_sparse)
    assert_equal(len(hll.sparse_set), 0)  # Sparse set should be cleared


fn test_sparse_register_state() raises:
    """Test that registers vector is not allocated until conversion from sparse mode.
    """
    var hll = HyperLogLog(14)

    # Add some values but stay in sparse mode
    for i in range(100):
        hll.add_hash(Int(murmur3_64(i)))

    # Verify we're still in sparse mode
    assert_true(hll.is_sparse)

    # Registers vector should have size 0 in sparse mode
    assert_equal(
        len(hll.registers), 0, "Registers vector should be empty in sparse mode"
    )
    assert_equal(
        len(hll.sparse_set), 100, "Sparse set should contain all values"
    )

    # Force conversion to dense by adding many values
    for i in range(1000000):
        hll.add_hash(Int(murmur3_64(i + 1000)))

    # Verify conversion happened
    assert_true(not hll.is_sparse)

    # After conversion, registers should be allocated with proper size
    assert_equal(
        len(hll.registers),
        1 << hll.precision,
        "Registers should be allocated after conversion",
    )
    assert_equal(
        len(hll.sparse_set), 0, "Sparse set should be empty after conversion"
    )

    # Some registers should be non-zero
    var has_nonzero = False
    for i in range(len(hll.registers)):
        if hll.registers[i] != 0:
            has_nonzero = True
            break
    assert_true(
        has_nonzero, "Some registers should be non-zero after conversion"
    )


fn test_cardinality_low_cardinality() raises:
    """Test cardinality estimation for small range of values."""
    var hll = HyperLogLog(14)
    var test_size = 100

    # Add values from 1 to 100
    for i in range(1, test_size + 1):
        hll.add_hash(Int(murmur3_64(i)))

    var estimated = hll.cardinality()

    # For small sets in sparse mode, should be exact
    assert_equal(estimated, test_size)


fn test_cardinality_large_cardinality() raises:
    """Test cardinality estimation for large range of values."""
    var hll = HyperLogLog(14)
    var test_size = 100_000

    # Add values from 1 to 100K
    for i in range(1, test_size + 1):
        hll.add_hash(Int(murmur3_64(i)))

    var estimated = hll.cardinality()
    var error_margin = Float64(abs(estimated - test_size)) / Float64(test_size)

    # For large sets, HyperLogLog should be within 2% error
    assert_true(error_margin <= 0.01)


fn test_merge_sparse_sparse() raises:
    """Test merging two HyperLogLog sketches."""
    var hll1 = HyperLogLog(14)
    var hll2 = HyperLogLog(14)
    var test_size = 1000

    # Add different values to each sketch
    for i in range(1, test_size + 1):
        hll1.add_hash(Int(murmur3_64(i)))
    for i in range(test_size + 1, test_size * 2 + 1):
        hll2.add_hash(Int(murmur3_64(i)))

    # Merge sketches
    hll1.merge(hll2)
    var merged_card = hll1.cardinality()

    assert_true(hll1.is_sparse)

    # Merged cardinality should be approximately sum of individual cardinalities
    var error_margin = Float64(abs(merged_card - (test_size * 2))) / Float64(
        test_size * 2
    )
    assert_true(error_margin <= 0.02)


fn test_merge_dense_dense() raises:
    """Test merging two HyperLogLog sketches."""
    var hll1 = HyperLogLog(14)
    var hll2 = HyperLogLog(14)
    var test_size = 100_000

    # Add different values to each sketch
    for i in range(1, test_size + 1):
        hll1.add_hash(Int(murmur3_64(i)))
    for i in range(test_size + 1, test_size * 2 + 1):
        hll2.add_hash(Int(murmur3_64(i)))

    # Merge sketches
    hll1.merge(hll2)
    var merged_card = hll1.cardinality()

    assert_true(not hll1.is_sparse)

    # Merged cardinality should be approximately sum of individual cardinalities
    var error_margin = Float64(abs(merged_card - (test_size * 2))) / Float64(
        test_size * 2
    )
    assert_true(error_margin <= 0.02)


fn test_merge_sparse_dense() raises:
    """Test merging sparse and dense HyperLogLog sketches."""
    var hll_sparse = HyperLogLog(14)
    var hll_dense = HyperLogLog(14)

    # Make first sketch sparse (small cardinality < 1K)
    for i in range(500):
        hll_sparse.add_hash(Int(murmur3_64(i)))
    assert_true(hll_sparse.is_sparse)

    # Make second sketch dense (large cardinality > 1M)
    for i in range(2_000_000):
        hll_dense.add_hash(
            Int(murmur3_64(i + 100))
        )  # Small overlap with sparse set
    assert_true(not hll_dense.is_sparse)

    # Test merging sparse into dense
    var hll_merge1 = hll_sparse
    hll_merge1.merge(hll_dense)
    var merged_card1 = hll_merge1.cardinality()

    # Test merging dense into sparse
    var hll_merge2 = hll_dense
    hll_merge2.merge(hll_sparse)
    var merged_card2 = hll_merge2.cardinality()

    # Both merges should give similar results
    var diff = Float64(abs(merged_card1 - merged_card2)) / Float64(merged_card1)
    assert_true(diff <= 0.01)

    # Verify cardinality is close to actual (2,000,400 unique values)
    var error_margin = Float64(abs(merged_card1 - 2_000_400)) / 2_000_400
    assert_true(error_margin <= 0.02)


fn test_merge_different_sizes() raises:
    """Test merging HyperLogLog sketches with different sizes."""
    var hll_small = HyperLogLog(14)
    var hll_large = HyperLogLog(14)

    # Add small number to first sketch (< 1K)
    for i in range(800):
        hll_small.add_hash(Int(murmur3_64(i)))

    # Add large number to second sketch (> 1M)
    for i in range(1_500_000):
        hll_large.add_hash(Int(murmur3_64(i + 10000)))  # No overlap

    # Merge small into large
    var hll_merge = hll_small
    hll_merge.merge(hll_large)
    var merged_card = hll_merge.cardinality()

    # Verify the merged cardinality is close to sum (1,500,800 unique values)
    var error_margin = Float64(abs(merged_card - 1_500_800)) / 1_500_800
    assert_true(error_margin <= 0.02)


fn test_serialization() raises:
    """Test serialization and deserialization."""
    var hll = HyperLogLog(14)
    var test_size = 1000000

    # Add some values
    for i in range(test_size):
        hll.add_hash(Int(murmur3_64(i)))

    # Serialize and deserialize
    var serialized = hll.serialize()
    var deserialized = HyperLogLog.deserialize(serialized)

    # Verify properties
    assert_equal(deserialized.precision, hll.precision)
    assert_equal(deserialized.is_sparse, hll.is_sparse)

    # Verify cardinality estimate
    var orig_card = hll.cardinality()
    var deser_card = deserialized.cardinality()
    assert_equal(deser_card, orig_card)


fn test_serialization_sparse() raises:
    """Test serialization of sparse HyperLogLog."""
    var hll = HyperLogLog(14)
    var test_size = 100  # Small enough to stay sparse

    # Add some values
    for i in range(test_size):
        hll.add_hash(Int(murmur3_64(i)))

    assert_true(hll.is_sparse)

    # Serialize and deserialize
    var serialized = hll.serialize()
    var deserialized = HyperLogLog.deserialize(serialized)

    # Verify properties
    assert_equal(deserialized.precision, hll.precision)
    assert_equal(deserialized.is_sparse, True)
    assert_equal(deserialized.cardinality(), test_size)


fn test_serialization_dense() raises:
    """Test serialization of dense HyperLogLog."""
    var hll = HyperLogLog(14)
    var test_size = 1000000  # Large enough to convert to dense

    # Add values to force dense conversion
    for i in range(test_size):
        hll.add_hash(Int(murmur3_64(i)))

    assert_true(not hll.is_sparse)

    # Serialize and deserialize
    var serialized = hll.serialize()
    var deserialized = HyperLogLog.deserialize(serialized)

    # Verify properties
    assert_equal(deserialized.precision, hll.precision)
    assert_equal(deserialized.is_sparse, False)

    # Verify cardinality is preserved
    var orig_card = hll.cardinality()
    var deser_card = deserialized.cardinality()
    assert_equal(deser_card, orig_card)


fn test_copy_low_cardinality() raises:
    """Test copying HyperLogLog sketches with low cardinality."""
    var original = HyperLogLog(14)
    var test_size = 1000

    # Add values to original
    for i in range(test_size):
        original.add_hash(Int(murmur3_64(i)))

    # Create copy
    var copied = original

    # Verify copied sketch has same properties
    assert_equal(copied.precision, original.precision)
    assert_equal(copied.is_sparse, original.is_sparse)
    assert_equal(copied.cardinality(), original.cardinality())

    # Modify original
    for i in range(test_size, test_size * 2):
        original.add_hash(Int(murmur3_64(i)))

    # Verify copy remains unchanged
    assert_true(copied.cardinality() != original.cardinality())


fn test_copy_high_cardinality() raises:
    """Test copying HyperLogLog sketches with high cardinality."""
    var original = HyperLogLog(14)
    var test_size = 1000000  # Large enough to convert to dense

    # Add values to original
    for i in range(test_size):
        original.add_hash(Int(murmur3_64(i)))

    # Create copy
    var copied = original

    # Verify copied sketch has same properties
    assert_equal(copied.precision, original.precision)
    assert_equal(copied.is_sparse, original.is_sparse)
    assert_equal(copied.cardinality(), original.cardinality())
