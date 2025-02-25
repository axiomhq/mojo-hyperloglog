fn rotl(x: UInt64, r: Int) -> UInt64:
    """Rotate left operation."""
    return (x << r) | (x >> (64 - r))


fn fmix64(k: UInt64) -> UInt64:
    """64-bit finalizer function."""
    var h = k
    h ^= h >> 33
    h *= 0xFF51AFD7ED558CCD
    h ^= h >> 33
    h *= 0xC4CEB9FE1A85EC53
    h ^= h >> 33
    return h


fn murmur3_64(key: Int, seed: UInt64 = 0) -> UInt64:
    """MurmurHash3 64-bit implementation."""
    var h1: UInt64 = seed
    var k1: UInt64 = UInt64(key)

    # Body
    k1 *= 0x87C37B91114253D5
    k1 = rotl(k1, 31)
    k1 *= 0x4CF5AD432745937F

    h1 ^= k1
    h1 = rotl(h1, 27)
    h1 = h1 * 5 + 0x52DCE729

    # Finalization
    h1 ^= 8  # Length in bytes
    return fmix64(h1)
