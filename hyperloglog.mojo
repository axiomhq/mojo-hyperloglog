from math import log2, exp2
from memory import memset_zero
from collections import InlinedFixedVector, List, Set, InlineArray
from beta import get_beta
from bit import count_leading_zeros


struct HyperLogLog:
    """
    HyperLogLog uses the LogLog-Beta implementation for cardinality estimation.
    This is a variant of LogLog that provides better accuracy through bias correction.
    """

    var precision: Int
    var max_zeros: Int  # Maximum value for leading zeros
    var alpha: Float64  # Alpha constant for bias correction
    var registers: InlinedFixedVector[UInt8]
    var sparse_set: Set[Int]  # For low cardinality
    var is_sparse: Bool  # Track if we're using sparse representation

    fn __init__(mut self, p: Int = 14) raises:
        """Initialize HyperLogLog with given precision (4-16)."""
        if p < 4 or p > 16:
            raise Error("Precision must be between 4 and 16")

        self.precision = p
        self.max_zeros = 64 - p  # Maximum possible leading zeros
        var m = 1 << p
        
        # Alpha values from the original paper
        if m == 16:
            self.alpha = 0.673
        elif m == 32:
            self.alpha = 0.697
        elif m == 64:
            self.alpha = 0.709
        else:
            self.alpha = 0.7213 / (1.0 + 1.079 / Float64(m))
    
        self.registers = InlinedFixedVector[UInt8](0)  # Empty vector until conversion
        self.sparse_set = Set[Int]()
        self.is_sparse = True

    fn __copyinit__(mut self, existing: Self):
        """Initialize this HyperLogLog as a copy of another."""
        self.precision = existing.precision
        self.max_zeros = existing.max_zeros
        self.alpha = existing.alpha
        self.sparse_set = Set[Int]()
        self.is_sparse = existing.is_sparse

        if existing.is_sparse:
            self.registers = InlinedFixedVector[UInt8](0)  # Start with empty vector
            # Copy sparse set
            for item in existing.sparse_set.__iter__():
                self.sparse_set.add(item[])
        else:
            # Copy registers
            self.registers = InlinedFixedVector[UInt8](1 << self.precision)
            for i in range(1 << self.precision):
                self.registers.__setitem__(i, existing.registers[i])

    fn add_hash(mut self, hash: Int):
        """Add a value to the sketch."""
        var hash_int = Int(hash)  # Convert UInt64 to Int

        if self.is_sparse:
            # Check if we need to convert to dense representation
            # Convert when sparse set size reaches 1/8 of register size
            if len(self.sparse_set) >= (1 << (self.precision-3)):
                self._convert_to_dense()  # First convert to dense
                self._add_to_dense(hash_int)  # Then add the new value
            else:
                self.sparse_set.add(hash_int)
            return

        self._add_to_dense(hash_int)

    fn _get_bucket_and_zeros(mut self, hash_int: Int, precision: Int) -> Tuple[Int, UInt8]:
        """Extract bucket and calculate leading zeros from hash value."""
        # Extract bucket from top p bits
        var bucket = (hash_int >> (64 - precision)) & ((1 << precision) - 1)
        
        # Prepare pattern for leading zeros calculation
        # Shift left by precision and set a marker bit
        var pattern = (hash_int << precision) | (1 << (precision - 1))
        var zeros = UInt8(count_leading_zeros(pattern) + 1)
        
        return bucket, zeros

    fn _add_to_dense(mut self, hash_int: Int):
        """Add a hash value to dense registers."""
        # Ensure registers are allocated
        if len(self.registers) == 0:
            self._convert_to_dense()
            
        bucket, zeros = self._get_bucket_and_zeros(hash_int, self.precision)
        if self.registers[bucket] < zeros:
            self.registers.__setitem__(bucket, zeros)

    fn _convert_to_dense(mut self):
        """Convert from sparse to dense representation."""
        # First allocate the registers with proper size
        var num_registers = 1 << self.precision
        self.registers = InlinedFixedVector[UInt8](num_registers)
        
        # Initialize all registers to 0
        for _ in range(num_registers):
            self.registers.append(0)
        
        # Process all hashes in sparse set
        for h in self.sparse_set.__iter__():
            var value = h[]
            # Extract bucket from top p bits
            bucket, zeros = self._get_bucket_and_zeros(value, self.precision)
            if self.registers[bucket] < zeros:
                self.registers.__setitem__(bucket, zeros)

        # Mark as dense and clear sparse set
        self.is_sparse = False
        self.sparse_set.clear()

    fn cardinality(self) -> Int:
        """Estimate number of unique elements."""
        if self.is_sparse:
            return len(self.sparse_set)

        var sum: Float64 = 0.0
        var zeros: Float64 = 0.0
        var m = Float64(1 << self.precision)

        for i in range(1 << self.precision):
            if self.registers[i] == 0:
                zeros += 1.0
            sum += 1.0 / Float64(1 << self.registers[i])

        var estimate = self.alpha * m * m / sum
        
        # Apply correction for empty registers
        if zeros > 0:
            var ez = zeros / m  # Empirical zeros ratio
            var beta = get_beta(self.precision, ez)
            if beta > 0:  # Only apply correction if beta is positive
                estimate = m * log2(m/zeros)
            
        # Apply small range correction
        if estimate <= 2.5 * m:
            if zeros > 0:
                estimate = m * log2(m/zeros)
            else:
                estimate = estimate  # Keep current estimate
                
        # Apply large range correction
        if estimate > Float64(1 << 32):
            estimate = -Float64(1 << 32) * log2(1.0 - estimate/Float64(1 << 32))
            
        return Int(estimate)

    fn merge(mut self, other: Self) raises:
        """Merge another sketch into this one."""
        if self.precision != other.precision:
            raise Error("Cannot merge HyperLogLog sketches with different precisions")

        # If either is dense, we need to work in dense mode
        if not self.is_sparse or not other.is_sparse:
            # Convert self to dense if needed
            if self.is_sparse:
                self._convert_to_dense()
            
            # If other is sparse, convert its values
            if other.is_sparse:
                for item in other.sparse_set.__iter__():
                    var value = item[]
                    bucket, zeros = self._get_bucket_and_zeros(value, self.precision)
                    if self.registers[bucket] < zeros:
                        self.registers.__setitem__(bucket, zeros)
            else:
                # Both are dense, take max of registers
                for i in range(1 << self.precision):
                    if self.registers[i] < other.registers[i]:
                        self.registers.__setitem__(i, other.registers[i])
        else:
            # Both are sparse, just merge the sets
            # This is safe because we're below the sparse threshold
            for item in other.sparse_set.__iter__():
                self.sparse_set.add(item[])

    fn serialize(mut self) -> List[UInt8]:
        """Serialize to bytes."""
        var buffer = List[UInt8]()
        buffer.append(UInt8(self.precision))
        buffer.append(UInt8(1 if self.is_sparse else 0))

        if self.is_sparse:
            # First add count of items
            var count = len(self.sparse_set)
            buffer.append(UInt8((count >> 24) & 0xFF))
            buffer.append(UInt8((count >> 16) & 0xFF))
            buffer.append(UInt8((count >> 8) & 0xFF))
            buffer.append(UInt8(count & 0xFF))
            
            # Then add each item as 8 bytes
            for item in self.sparse_set.__iter__():
                var value = item[]
                for shift in range(56, -8, -8):
                    buffer.append(UInt8((value >> shift) & 0xFF))
        else:
            # Convert to dense if needed
            if len(self.registers) == 0:
                self._convert_to_dense()
            # Serialize registers
            for i in range(1 << self.precision):
                buffer.append(self.registers[i])
        return buffer

    @staticmethod
    fn deserialize(buffer: List[UInt8]) raises -> Self:
        """Create from serialized bytes."""
        if len(buffer) < 2:
            raise Error("Invalid serialized data: buffer too short")

        var precision = Int(buffer[0])
        var is_sparse = buffer[1] == 1
        var hll = HyperLogLog(precision)
        hll.is_sparse = is_sparse

        if is_sparse:
            if len(buffer) < 6:
                raise Error("Invalid serialized data: sparse buffer too short")
            
            # Read count
            var count = (Int(buffer[2]) << 24) | (Int(buffer[3]) << 16) | 
                       (Int(buffer[4]) << 8) | Int(buffer[5])
            
            # Read items
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
            var expected_size = (1 << precision) + 2
            if len(buffer) != expected_size:
                raise Error("Invalid serialized data: wrong buffer length")
            # Allocate registers
            hll.registers = InlinedFixedVector[UInt8](1 << precision)
            for i in range(1 << precision):
                hll.registers.__setitem__(i, buffer[i + 2])

        return hll
