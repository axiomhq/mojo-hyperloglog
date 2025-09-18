from math import log2, exp2
from collections import List, Set
from beta import get_beta
from bit import count_leading_zeros

struct HyperLogLog[P: Int](ImplicitlyCopyable):
    """
    HyperLogLog using the LogLog-Beta algorithm for cardinality estimation.
    Provides bias correction for improved accuracy.
    """
    alias precision = P                    # Number of bits used for register indexing (4-16)
    alias max_zeros = 64 - P              # Maximum possible leading zeros
    alias m = 1 << P                      # Number of registers
    var registers: List[UInt8]            # Dense representation
    var sparse_set: Set[Int]              # Sparse representation for low cardinality
    var is_sparse: Bool                   # Tracks current representation mode

    @staticmethod
    @parameter
    fn _get_alpha() -> Float64:
        """Get alpha constant at compile time based on precision."""
        @parameter
        if P == 4:
            return 0.673
        elif P == 5:
            return 0.697
        elif P == 6:
            return 0.709
        else:
            return 0.7213 / (1.0 + 1.079 / Float64(1 << P))

    fn __init__(out self) raises:
        """Initialize HyperLogLog with compile-time precision P."""
        # Compile-time validation
        constrained[P >= 4 and P <= 16, "Precision must be between 4 and 16"]()

        # Initialize empty data structures
        self.registers = List[UInt8]()
        self.sparse_set = Set[Int]()
        self.is_sparse = True

    fn __copyinit__(out self, existing: Self):
        """Copy-initialize from an existing HyperLogLog."""
        self.is_sparse = existing.is_sparse
        self.sparse_set = Set[Int]()

        if self.is_sparse:
            self.registers = List[UInt8]()
            for item in existing.sparse_set:
                self.sparse_set.add(item)
        else:
            self.registers = List[UInt8]()
            for i in range(len(existing.registers)):
                self.registers.append(existing.registers[i])

    fn add_hash(mut self, hash: Int):
        """Incorporate a new hash value into the sketch."""
        var hash_int = Int(hash)
        if self.is_sparse:
            # Convert to dense representation when sparse set grows too large
            alias threshold = 1 << (Self.precision - 3)
            if len(self.sparse_set) >= threshold:
                self._convert_to_dense()
                self._add_to_dense(hash_int)
            else:
                self.sparse_set.add(hash_int)
        else:
            self._add_to_dense(hash_int)

    fn _get_bucket_and_zeros(mut self, hash_int: Int) -> Tuple[Int, UInt8]:
        """Extract the bucket index and count the leading zeros."""
        alias mask: Int = (1 << Self.precision) - 1
        var bucket: Int = (hash_int >> (64 - Self.precision)) & mask
        var pattern: Int = (hash_int << Self.precision) | (1 << (Self.precision - 1))
        var zeros: UInt8 = UInt8(count_leading_zeros(pattern) + 1)
        return bucket, zeros

    fn _add_to_dense(mut self, hash_int: Int):
        """Update the dense registers using the given hash."""
        if len(self.registers) == 0:
            self._convert_to_dense()
        var bucket: Int
        var zeros: UInt8
        bucket, zeros = self._get_bucket_and_zeros(hash_int)
        if self.registers[bucket] < zeros:
            self.registers[bucket] = zeros

    fn _convert_to_dense(mut self):
        """Switch from sparse to dense representation."""
        self.registers = List[UInt8]()

        # Initialize all registers to 0
        for _ in range(Self.m):
            self.registers.append(0)

        # Process all hashes from sparse set
        for h in self.sparse_set:
            var value = h
            var bucket: Int
            var zeros: UInt8
            bucket, zeros = self._get_bucket_and_zeros(value)
            if self.registers[bucket] < zeros:
                self.registers[bucket] = zeros

        self.is_sparse = False
        self.sparse_set.clear()

    fn cardinality(self) -> Int:
        """Estimate number of unique elements."""
        if self.is_sparse:
            return len(self.sparse_set)

        var sum: Float64 = 0.0
        var ez: Float64 = 0.0  # Count of empty registers

        # Calculate harmonic mean of register values
        alias vector_width = min(Self.m, 64)
        for i in range(0, Self.m, vector_width):
            var reg = self.registers.unsafe_ptr().load[width=vector_width](i)
            var reg_flag = reg.eq(0)
            ez += Float64(reg_flag.cast[DType.uint8]().reduce_add())
            var exp_reg = reg_flag.select(SIMD[DType.float16, vector_width](0), 1 / exp2(reg.cast[DType.float16]()))
            sum += exp_reg.reduce_add().cast[DType.float64]()

        # Apply LogLog-Beta bias correction
        return Int(Self._get_alpha() * Self.m * (Self.m - ez) / (get_beta[Self.precision](ez) + sum))

    fn merge(mut self, other: Self) raises:
        """Merge another sketch into this one."""
        # Precision is now compile-time guaranteed to match

        # Handle dense mode merging
        if not self.is_sparse or not other.is_sparse:
            if self.is_sparse:
                self._convert_to_dense()

            if other.is_sparse:
                # Merge sparse into dense
                for h in other.sparse_set:
                    var value = h
                    var bucket: Int
                    var zeros: UInt8
                    bucket, zeros = self._get_bucket_and_zeros(value)
                    if self.registers[bucket] < zeros:
                        self.registers[bucket] = zeros
            else:
                # Merge dense into dense
                for i in range(Self.m):
                    if self.registers[i] < other.registers[i]:
                        self.registers[i] = other.registers[i]
        else:
            # Both are sparse, simply merge sets
            for h in other.sparse_set:
                self.sparse_set.add(h)

    fn serialize(mut self) -> List[UInt8]:
        """Serialize the sketch into a byte list."""
        var buffer = List[UInt8]()

        # Write header
        buffer.append(UInt8(Self.precision))
        buffer.append(UInt8(1 if self.is_sparse else 0))

        if self.is_sparse:
            # Write sparse set size
            var count = len(self.sparse_set)
            buffer.append(UInt8((count >> 24) & 0xFF))
            buffer.append(UInt8((count >> 16) & 0xFF))
            buffer.append(UInt8((count >> 8) & 0xFF))
            buffer.append(UInt8(count & 0xFF))

            # Write sparse set values
            for h in self.sparse_set:
                var value = h
                for shift in range(56, -8, -8):
                    buffer.append(UInt8((value >> shift) & 0xFF))
        else:
            # Ensure we're in dense mode
            if len(self.registers) == 0:
                self._convert_to_dense()

            # Write register values
            for i in range(Self.m):
                buffer.append(self.registers[i])

        return buffer^

    @staticmethod
    fn deserialize[TargetP: Int](buffer: List[UInt8]) raises -> HyperLogLog[TargetP]:
        """Deserialize a sketch from the given byte list."""
        if len(buffer) < 2:
            raise Error("Invalid serialized data: buffer too short")

        # Read and validate precision matches type parameter
        var stored_precision = Int(buffer[0])
        if stored_precision != TargetP:
            raise Error("Stored precision does not match expected precision")

        var is_sparse = buffer[1] == 1
        var hll = HyperLogLog[TargetP]()
        hll.is_sparse = is_sparse

        if is_sparse:
            if len(buffer) < 6:
                raise Error("Invalid serialized data: sparse buffer too short")

            # Read sparse set size
            var count = (Int(buffer[2]) << 24) | (Int(buffer[3]) << 16) |
                       (Int(buffer[4]) << 8) | Int(buffer[5])

            # Read sparse set values
            var pos = 6
            for _ in range(count):
                if pos + 8 > len(buffer):
                    raise Error("Invalid serialized data: incomplete sparse item")
                var value: Int = 0
                for j in range(8):
                    value = (value << 8) | Int(buffer[pos + j])
                hll.sparse_set.add(value)
                pos += 8
        else:
            # Verify buffer size
            alias expected_size = (1 << TargetP) + 2
            if len(buffer) != expected_size:
                raise Error("Invalid serialized data: wrong buffer length")

            # Read register values
            alias num_registers = 1 << TargetP
            hll.registers = List[UInt8]()
            for i in range(num_registers):
                hll.registers.append(buffer[i + 2])

        return hll^
