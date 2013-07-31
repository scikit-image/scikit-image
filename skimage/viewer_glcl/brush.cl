__kernel void draw(
    __global uchar* canvas,
    __global uint* points,
    int n_points,
    char label
)
{
    int gx = get_global_id(0);

    if (gx > n_points-1)
        return;

    canvas[points[gx]] = label;
}
