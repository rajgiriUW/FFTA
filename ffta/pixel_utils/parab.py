"""parab.py: Parabola fit around three points to find a true vertex."""

def fit(f, x):
    """
    f = array
    x = index of peak, typically just argmax

    Uses parabola equation to fit to the peak and two surrounding points
    """

    x1 = x - 1
    x2 = x
    x3 = x + 1

    y1 = f[x - 1]
    y2 = f[x]
    y3 = f[x + 1]

    d = (x1 - x3) * (x1 - x2) * (x2 - x3)

    A = (x1 * (y3 - y2) + x2 * (y1 - y3) + x3 * (y2 - y1)) / d

    B = (x1 ** 2.0 * (y2 - y3) +
         x2 ** 2.0 * (y3 - y1) +
         x3 ** 2.0 * (y1 - y2)) / d

    C = (x2 * x3 * (x2 - x3) * y1 +
         x3 * x1 * (x3 - x1) * y2 +
         x1 * x2 * (x1 - x2) * y3) / d

    xindex = -B / (2.0 * A)
    yindex = C - B ** 2.0 / (4.0 * A)

    return xindex, yindex
