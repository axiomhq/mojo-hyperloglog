from math import log2, exp2
from memory import memset_zero
from collections import InlinedFixedVector, List
from murmur import murmur3_64
from beta import get_beta
from bit import count_leading_zeros


fn get_alpha(precision: Int) -> Float64:
    """Get the alpha constant for a given precision.

    Args:
        precision: The precision value (4-16).

    Returns:
        The alpha constant for bias correction.
    """
    var m = 1 << precision  # Number of registers

    if m == 16:
        return 0.673
    elif m == 32:
        return 0.697
    elif m == 64:
        return 0.709
    else:
        return 0.7213 / (1.0 + 1.079 / Float64(m))


struct HyperLogLog:
    """
    HyperLogLog uses the LogLog-Beta implementation for cardinality estimation.
    This is a variant of LogLog that provides better accuracy through bias correction.
    """

    var precision: Int
    var num_registers: Int
    var max_zeros: Int  # Maximum value for leading zeros
    var alpha: Float64  # Alpha constant for bias correction
    var registers: InlinedFixedVector[UInt8]

    fn __init__(mut self, precision: Int = 14) raises:
        """Initialize HyperLogLog with given precision (4-16)."""
        if precision < 4 or precision > 16:
            raise Error("Precision must be between 4 and 16")

        self.precision = precision
        self.num_registers = 1 << precision
        self.max_zeros = 64 - precision
        self.alpha = get_alpha(precision)
        self.registers = InlinedFixedVector[UInt8](self.num_registers)

        # Initialize registers to 0
        for i in range(self.num_registers):
            self.registers.__setitem__(i, 0)

    fn __copyinit__(mut self, existing: Self):
        """Initialize this HyperLogLog as a copy of another."""
        self.precision = existing.precision
        self.num_registers = existing.num_registers
        self.max_zeros = existing.max_zeros
        self.alpha = existing.alpha
        self.registers = InlinedFixedVector[UInt8](self.num_registers)
        for i in range(self.num_registers):
            self.registers.__setitem__(i, existing.registers[i])

    fn add(mut self, value: Int):
        """Add a value to the sketch."""
        var hash = murmur3_64(value)

        # Get bucket (lower precision bits)
        var bucket = Int(hash & ((1 << self.precision) - 1))

        # Count leading zeros in remaining bits
        var pattern = hash >> self.precision
        var zeros = count_leading_zeros(pattern)
        # Adjust for the bits we shifted out
        zeros = zeros - self.precision
        if zeros > self.max_zeros:
            zeros = self.max_zeros

        # Update register if new value is larger
        var new_value = UInt8(zeros + 1)
        if self.registers[bucket] < new_value:
            self.registers.__setitem__(bucket, new_value)

    fn cardinality(self) -> Int:
        """Estimate number of unique elements."""
        var sum: Float64 = 0.0
        var zeros: Float64 = 0.0

        for i in range(self.num_registers):
            if self.registers[i] == 0:
                zeros += 1.0
            sum += 1.0 / exp2(Float64(self.registers[i]))

        var m = Float64(self.num_registers)
        return Int(
            self.alpha
            * m
            * (m - zeros)
            / (get_beta(self.precision, zeros) + sum)
        )

    fn merge(mut self, other: Self):
        """Merge another sketch into this one."""
        for i in range(self.num_registers):
            if self.registers[i] < other.registers[i]:
                self.registers.__setitem__(i, other.registers[i])

    fn serialize(self) -> List[UInt8]:
        """Serialize to bytes."""
        var buffer = List[UInt8]()
        buffer.append(UInt8(self.precision))
        for i in range(self.num_registers):
            buffer.append(self.registers[i])
        return buffer

    @staticmethod
    fn deserialize(buffer: List[UInt8]) raises -> Self:
        """Create from serialized bytes."""
        var hll = HyperLogLog(Int(buffer[0]))
        for i in range(hll.num_registers):
            hll.registers.__setitem__(i, buffer[i + 1])
        return hll
