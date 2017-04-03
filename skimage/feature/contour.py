class Point:
    def __init__(self, i, j):
        self.i = i
        self.j = j

    def __eq__(self, other):
        if other is None:
            return False
        return self.i == other.i and self.j == other.j

    def __repr__(self):
        return "(%s, %s)" % (self.i, self.j)


class Direction:
    """Helper class for processing directions.

    We condiser all 8 directions in a 2D image array here.
    """

    di = [-1, -1, 0, 1, 1, 1, 0, -1]
    dj = [0, 1, 1, 1, 0, -1, -1, -1]

    def __init__(self, index=None):
        self.index = index

    def __eq__(self, other):
        return self.index == other.index

    def __int__(self):
        return self.index

    def clockwise(self):
        return Direction((self.index + 1) % 8)

    def anticlockwise(self):
        return Direction((self.index + 7) % 8)

    def get_active_pixel(self, image, pt):
        height = len(image)
        width = len(image[0])
        i = pt.i + Direction.di[self.index]
        j = pt.j + Direction.dj[self.index]

        if i < 0 or i >= height or j < 0 or j >= width:
            return None

        return Point(i, j) if image[i][j] != 0 else None

    def get_direction(pt1, pt2):
        for i in range(8):
            if (pt1.i + Direction.di[i] == pt2.i and
                    pt1.j + Direction.dj[i] == pt2.j):
                return Direction(i)
        return None


class Contour:
    """An image contour.

    Represents a contour found in the processed image, with pointers
    to its parent and children.

    """

    HOLE = 1
    OUTER = 2
    identifier = 1

    def __init__(self, contour_type=None, pt=None):
        self.parent = None
        self.children = []
        self.points = []
        self.identifier = Contour.identifier
        Contour.identifier += 1
        if pt is not None:
            self.points.append(pt)
        if contour_type is not None:
            self.contour_type = contour_type

    # Adding (i, j) coordinates
    def add_point(self, i, j):
        self.points.append(Point(i, j))

    def add_child(self, child):
        self.children.append(child)

    def set_parent(self, parent):
        self.parent = parent
        parent.add_child(self)


def find_contours(image):
    """Finds contours in a binary image.

    Performs a raster scan of a 2D image array. Modifies the image in
    the process.

    Suzuki, S. and Abe, K., Topological Structural Analysis of
    Digitized Binary Images by Border Following. CVGIP 30 1, pp 32-46 (1985)
    """

    nbd = [1]
    lnbd = [1]
    height, width = len(image), len(image[0])

    # Add outer frame as hole border
    root = Contour(Contour.HOLE)
    for i in range(height):
        root.add_point(i, 0)
        root.add_point(i, width - 1)
    for j in range(1, width - 1):
        root.add_point(0, j)
        root.add_point(height - 1, j)

    # Initialize set of borders as outer frame
    border_map = {lnbd[0]: root}

    for i in range(height):
        lnbd[0] = 1
        for j in range(width):
            fij = image[i][j]
            if fij == 0:
                continue

            is_outer = is_outer_border_start(image, i, j)
            is_hole = is_hole_border_start(image, i, j)

            if is_outer or is_hole:
                border = Contour(Point(i, j))
                border_prime = None
                from_pt = Point(i, j)
                if is_outer:
                    nbd[0] += 1
                    from_pt.j -= 1
                    border.contour_type = Contour.OUTER
                    border_prime = border_map.get(lnbd[0])

                    if border_prime.contour_type == Contour.OUTER:
                        border.set_parent(border_prime.parent)
                    elif border_prime.contour_type == Contour.HOLE:
                        border.set_parent(border_prime)
                else:
                    nbd[0] += 1
                    if (fij > 1):
                        lnbd[0] = fij
                    border_prime = border_map.get(lnbd[0])
                    from_pt.j += 1
                    border.contour_type = Contour.HOLE

                    if border_prime.contour_type == Contour.OUTER:
                        border.set_parent(border_prime)
                    elif border_prime.contour_type == Contour.HOLE:
                        border.set_parent(border_prime.parent)

                ij = Point(i, j)
                directed_contour(image, ij, from_pt, border, nbd)

                if len(border.points) == 0:
                    border.add_point(ij.i, ij.j)
                    image[i][j] = -nbd[0]
                border_map[nbd[0]] = border

            if fij != 1:
                lnbd[0] = abs(fij)
    return image, border_map


def directed_contour(image, ij, i2j2, border, nbd):
    direction = Direction.get_direction(ij, i2j2)
    trace = direction.clockwise()

    i1j1 = None
    while trace != direction:
        active_pixel = trace.get_active_pixel(image, ij)
        if active_pixel is not None:
            i1j1 = active_pixel
            break
        trace = trace.clockwise()

    if i1j1 is None:
        return

    i2j2 = i1j1
    i3j3 = ij
    checked = [False for _ in range(8)]

    while True:
        i4j4 = None
        direction = Direction.get_direction(i3j3, i2j2)
        trace = direction.anticlockwise()
        for i in range(8):
            checked[i] = False
        while True:
            i4j4 = trace.get_active_pixel(image, i3j3)
            if i4j4 is not None:
                break
            checked[int(trace)] = True
            trace = trace.anticlockwise()

        # add points
        border.add_point(i3j3.i, i3j3.j)
        if crosses_east_border(image, checked, i3j3):
            image[i3j3.i][i3j3.j] = -nbd[0]
        elif image[ij.i][ij.j] == 1:
            image[i3j3.i][i3j3.j] = nbd[0]

        if i4j4 == ij and i3j3 == i1j1:
            break
        i2j2 = i3j3
        i3j3 = i4j4


def crosses_east_border(image, checked, pt):
    direction = int(Direction.get_direction(pt, Point(pt.i, pt.j + 1)))
    return (image[pt.i][pt.j] != 0 and
            (pt.j == len(image[0]) - 1 or checked[direction]))


def is_outer_border_start(image, i, j):
    return (image[i][j] == 1 and (j == 0 or image[i][j - 1] == 0))


def is_hole_border_start(image, i, j):
    return (image[i][j] >= 1 and
            (j == len(image[0]) - 1 or image[i][j + 1] == 0))
