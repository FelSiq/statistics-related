"""Linear congruental RNG suggested by POSIX.1-2001."""

_seed = 0


def set_seed(seed: int) -> None:
    global _seed
    _seed = seed


def lcrng():
    global _seed
    _seed = _seed * 1103515245 + 12345
    return (_seed // 65536) % 32768


if __name__ == "__main__":
    set_seed(256)
    vals = set()

    tried = 0
    random_n = lcrng()

    while random_n not in vals:
        vals.add(random_n)
        random_n = lcrng()
        tried += 1

    print("Tried {} times.".format(tried))
