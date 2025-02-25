from testing import assert_equal, assert_true
from hyperloglog import HyperLogLog


fn test_initialization() raises:
    """Test HyperLogLog initialization with different precision values."""
    print("Running initialization tests...")

    # Test valid precision
    var hll = HyperLogLog(14)
    assert_equal(hll.precision, 14)
    assert_equal(hll.num_registers, 1 << 14)
    assert_equal(hll.max_zeros, 50)  # 64 - 14

    # Test minimum precision
    var hll_min = HyperLogLog(4)
    assert_equal(hll_min.precision, 4)
    assert_equal(hll_min.num_registers, 1 << 4)
    assert_equal(hll_min.max_zeros, 60)  # 64 - 4

    # Test maximum precision
    var hll_max = HyperLogLog(16)
    assert_equal(hll_max.precision, 16)
    assert_equal(hll_max.num_registers, 1 << 16)
    assert_equal(hll_max.max_zeros, 48)  # 64 - 16

    print("✓ Initialization tests passed")


fn test_cardinality_large_range() raises:
    """Test cardinality estimation for large range of values."""
    print("Running large range cardinality tests...")

    var hll = HyperLogLog(14)

    # Add values from 1 to 1M
    print("Adding 1M elements...")
    for i in range(1, 1_000_001):
        hll.add(i)

    var estimated = hll.cardinality()
    var actual = 1_000_000
    var error_margin = Float64(abs(estimated - actual)) / Float64(actual)

    # Print debug information
    print("Actual count:", actual)
    print("Estimated count:", estimated)
    print("Error margin:", error_margin * 100, "%")

    # For large sets, HyperLogLog should be within 2% error
    assert_true(
        error_margin <= 0.02,
        "Error margin " + String(error_margin) + " exceeds threshold 0.02",
    )

    print("✓ Large range cardinality tests passed")


fn test_cardinality_duplicates() raises:
    """Test cardinality estimation with duplicate values."""
    print("Running duplicate values tests...")

    var hll = HyperLogLog(14)

    # Add same value multiple times
    print("Adding same value 1M times...")
    for _ in range(1_000_000):
        hll.add(42)

    var estimated = hll.cardinality()
    print("Estimated cardinality for duplicates:", estimated)
    # Should estimate close to 1 since we only added one unique value
    assert_true(
        estimated <= 3,
        "Estimated cardinality "
        + String(estimated)
        + " is too high for single unique value",
    )

    print("✓ Duplicate values tests passed")


fn test_merge() raises:
    """Test merging two HyperLogLog sketches."""
    print("Running merge tests...")

    var hll1 = HyperLogLog(14)
    var hll2 = HyperLogLog(14)

    # Add different values to each sketch
    print("Adding 500K elements to first sketch...")
    for i in range(1, 500_001):
        hll1.add(i)

    print("Adding 500K elements to second sketch...")
    for i in range(500_001, 1_000_001):
        hll2.add(i)

    # Get individual cardinalities
    var card1 = hll1.cardinality()
    var card2 = hll2.cardinality()
    print("First sketch cardinality:", card1)
    print("Second sketch cardinality:", card2)

    # Merge sketches
    print("Merging sketches...")
    hll1.merge(hll2)
    var merged_card = hll1.cardinality()
    print("Merged cardinality:", merged_card)

    # Merged cardinality should be approximately sum of individual cardinalities
    var error_margin = Float64(abs(merged_card - 1_000_000)) / 1_000_000.0
    print("Merge error margin:", error_margin * 100, "%")
    assert_true(
        error_margin <= 0.02,
        "Merge error margin "
        + String(error_margin)
        + " exceeds threshold 0.02",
    )

    print("✓ Merge tests passed")


fn test_copy() raises:
    """Test copying HyperLogLog sketches."""
    print("Running copy tests...")

    var original = HyperLogLog(14)
    print("Adding 1M elements to original sketch...")
    for i in range(1, 1_000_001):
        original.add(i)

    print("Creating copy...")
    var copied = original  # This will use __copyinit__ automatically

    # Verify copied sketch has same properties
    assert_equal(copied.precision, original.precision)
    assert_equal(copied.num_registers, original.num_registers)
    assert_equal(copied.max_zeros, original.max_zeros)

    var orig_card = original.cardinality()
    var copy_card = copied.cardinality()
    print("Original cardinality:", orig_card)
    print("Copied cardinality:", copy_card)
    assert_equal(copy_card, orig_card)

    # Modify original with many new values to ensure detectable change
    print("Modifying original with 100K new elements...")
    for i in range(1_000_001, 1_100_001):
        original.add(i)

    var new_orig_card = original.cardinality()
    var new_copy_card = copied.cardinality()
    print("Original cardinality after modification:", new_orig_card)
    print("Copy cardinality after modification:", new_copy_card)

    assert_true(
        new_copy_card != new_orig_card,
        "Copy was not independent of original (orig: "
        + String(new_orig_card)
        + ", copy: "
        + String(new_copy_card)
        + ")",
    )

    print("✓ Copy tests passed")


fn test_serialize_deserialize() raises:
    """Test serialization and deserialization of HyperLogLog sketches."""
    print("Running serialization tests...")

    # Create and populate a HLL
    var original = HyperLogLog(14)
    for i in range(1000):
        original.add(i)

    # Serialize and deserialize
    var serialized = original.serialize()
    var deserialized = HyperLogLog.deserialize(serialized)

    # Verify precision and derived values
    assert_equal(deserialized.precision, original.precision)
    assert_equal(deserialized.num_registers, original.num_registers)
    assert_equal(deserialized.max_zeros, original.max_zeros)
    assert_equal(deserialized.alpha, original.alpha)

    # Verify registers
    for i in range(original.num_registers):
        assert_equal(
            deserialized.registers[i],
            original.registers[i],
            "Register mismatch at index " + String(i),
        )

    # Verify cardinality estimate
    var orig_card = original.cardinality()
    var deser_card = deserialized.cardinality()
    assert_equal(
        deser_card, orig_card, "Cardinality mismatch after deserialization"
    )

    print("✓ Serialization tests passed")


fn main() raises:
    print("Starting HyperLogLog tests...")
    test_initialization()
    test_cardinality_large_range()
    test_cardinality_duplicates()
    test_merge()
    test_copy()
    test_serialize_deserialize()
    print("All tests passed! ✨")
