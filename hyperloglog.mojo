from math import log2, exp2
from collections import List, Set
from beta import get_beta
from bit import count_leading_zeros

struct HyperLogLog:
    """
    HyperLogLog using the LogLog-Beta algorithm for cardinality estimation.
    Provides bias correction for improved accuracy.
    """
    var precision: Int           # Number of bits used for register indexing (4-16)
    var max_zeros: Int           # Maximum possible leading zeros
    var alpha: Float64           # Bias-correction constant 
    var registers: List[UInt8]  # Dense representation
    var sparse_set: Set[Int]     # Sparse representation for low cardinality
    var is_sparse: Bool          # Tracks current representation mode

    fn __init__(out self, p: Int = 14) raises:
        """Initialize with precision between 4 and 16."""
        if p < 4 or p > 16:
            raise Error("Precision must be between 4 and 16")

        self.precision = p
        self.max_zeros = 64 - p
        var m = 1 << p

        # Set alpha based on the number of registers (m)
        if m == 16:
            self.alpha = 0.673
        elif m == 32:
            self.alpha = 0.697
        elif m == 64:
            self.alpha = 0.709
        else:
            self.alpha = 0.7213 / (1.0 + 1.079 / Float64(m))

        # Initialize empty data structures
        self.registers = List[UInt8]()
        self.sparse_set = Set[Int]()
        self.is_sparse = True

    fn __copyinit__(out self, existing: Self):
        """Copy-initialize from an existing HyperLogLog."""
        self.precision = existing.precision
        self.max_zeros = existing.max_zeros
        self.alpha = existing.alpha
        self.is_sparse = existing.is_sparse
        self.sparse_set = Set[Int]()

        if self.is_sparse:
            self.registers = List[UInt8]()
            for item in existing.sparse_set:
                self.sparse_set.add(item)
        else:
            var num_registers = 1 << self.precision
            self.registers = List[UInt8](num_registers)
            for i in range(num_registers):
                self.registers[i] = existing.registers[i]

    fn add_hash(mut self, hash: Int):
        """Incorporate a new hash value into the sketch."""
        var hash_int = Int(hash)
        if self.is_sparse:
            # Convert to dense representation when sparse set grows too large
            if len(self.sparse_set) >= (1 << (self.precision - 3)):
                self._convert_to_dense()
                self._add_to_dense(hash_int)
            else:
                self.sparse_set.add(hash_int)
        else:
            self._add_to_dense(hash_int)

    fn _get_bucket_and_zeros(mut self, hash_int: Int) -> Tuple[Int, UInt8]:
        """Extract the bucket index and count the leading zeros."""
        var mask: Int = (1 << self.precision) - 1
        var bucket: Int = (hash_int >> (64 - self.precision)) & mask
        var pattern: Int = (hash_int << self.precision) | (1 << (self.precision - 1))
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
        var num_registers = 1 << self.precision
        self.registers = List[UInt8](num_registers)

        # Initialize all registers to 0
        for _ in range(num_registers):
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

        var m = 1 << self.precision
        var sum: Float64 = 0.0
        var ez: Float64 = 0.0  # Count of empty registers

        # Calculate harmonic mean of register values
        for i in range(m):
            var reg = self.registers[i]
            if reg == 0:
                ez += 1.0
            else:
                sum += 1.0 / exp2(Float64(reg))

        # Apply LogLog-Beta bias correction
        return Int(self.alpha * m * (m - ez) / (get_beta(self.precision, ez) + sum))

    fn merge(mut self, other: Self) raises:
        """Merge another sketch into this one."""
        if self.precision != other.precision:
            raise Error("Cannot merge sketches with different precisions")

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
                var m = 1 << self.precision
                for i in range(m):
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
        buffer.append(UInt8(self.precision))
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
            var num_registers = 1 << self.precision
            for i in range(num_registers):
                buffer.append(self.registers[i])

        return buffer

    @staticmethod
    fn deserialize(buffer: List[UInt8]) raises -> Self:
        """Deserialize a sketch from the given byte list."""
        if len(buffer) < 2:
            raise Error("Invalid serialized data: buffer too short")

        # Read header
        var precision = Int(buffer[0])
        var is_sparse = buffer[1] == 1
        var hll = HyperLogLog(precision)
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
            var expected_size = (1 << precision) + 2
            if len(buffer) != expected_size:
                raise Error("Invalid serialized data: wrong buffer length")

            # Read register values
            var num_registers = 1 << precision
            hll.registers = List[UInt8](num_registers)
            for i in range(num_registers):
                hll.registers[i] = buffer[i + 2]

        return hll