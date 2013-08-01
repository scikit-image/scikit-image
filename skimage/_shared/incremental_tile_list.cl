#define OPERATOR_EQUAL 0
#define OPERATOR_GT 1
#define OPERATOR_LT 2
#define OPERATOR_GTE 3
#define OPERATOR_LTE 4

#define LOGICAL_AND 0
#define LOGICAL_OR 1

#define TRUE 1
#define FALSE 0

__kernel void init(
	__global int* tiles,
	int2 tilesDim
) {
	int gx = get_global_id(0);
	int gy = get_global_id(1);

	if (gx > tilesDim.x-1 || gy > tilesDim.y-1)
		return;

	tiles[gy*tilesDim.x + gx] = FALSE;
}

__kernel void init_incremental(
	__global int* tiles,
	int2 tilesDim,
	int iteration
) {
	int gx = get_global_id(0);
	int gy = get_global_id(1);

	if (gx > tilesDim.x-1 || gy > tilesDim.y-1)
		return;

	tiles[gy*tilesDim.x + gx] = iteration;
}

__kernel void flag(
	__global int* in,
	__global int* out,
	int length,
    int operator,
    int operand
)
{
	int gx = get_global_id(0);

	if (gx > length-1)
		return;

	int flag = FALSE;

	switch(operator) {
		case OPERATOR_EQUAL: if (in[gx] == operand) flag = TRUE; break;
		case OPERATOR_GT:    if (in[gx] >  operand) flag = TRUE; break;
		case OPERATOR_LT:    if (in[gx] <  operand) flag = TRUE; break;
		case OPERATOR_GTE:   if (in[gx] >= operand) flag = TRUE; break;
		case OPERATOR_LTE:   if (in[gx] <= operand) flag = TRUE; break;
	}

	out[gx] = flag;
}

__kernel void flag_logical(
	__global int* in1,
	__global int* in2,
	__global int* out,
	int length,
    int operator1,
    int operator2,
    int operand1,
    int operand2,
	int logical
)
{
	int gx = get_global_id(0);

	if (gx > length-1)
		return;

	int flag1 = FALSE;
	int flag2 = FALSE;

	switch(operator1) {
		case OPERATOR_EQUAL: if (in1[gx] == operand1) flag1 = TRUE; break;
		case OPERATOR_GT:    if (in1[gx] >  operand1) flag1 = TRUE; break;
		case OPERATOR_LT:    if (in1[gx] <  operand1) flag1 = TRUE; break;
		case OPERATOR_GTE:   if (in1[gx] >= operand1) flag1 = TRUE; break;
		case OPERATOR_LTE:   if (in1[gx] <= operand1) flag1 = TRUE; break;
	}

	switch(operator2) {
		case OPERATOR_EQUAL: if (in2[gx] == operand2) flag2 = TRUE; break;
		case OPERATOR_GT:    if (in2[gx] >  operand2) flag2 = TRUE; break;
		case OPERATOR_LT:    if (in2[gx] <  operand2) flag2 = TRUE; break;
		case OPERATOR_GTE:   if (in2[gx] >= operand2) flag2 = TRUE; break;
		case OPERATOR_LTE:   if (in2[gx] <= operand2) flag2 = TRUE; break;
	}

	switch(logical) {
		case LOGICAL_AND: flag1 = flag1 && flag2; break;
		case LOGICAL_OR : flag1 = flag1 || flag2; break;
	}

	out[gx] = flag1;
}

__kernel void increment_logical(
	__global int* in1,
	__global int* in2,
	__global int* out,
	int length,
    int operator1,
    int operator2,
    int operand1,
    int operand2,
	int logical,
	int iteration
)
{
	int gx = get_global_id(0);

	if (gx > length-1)
		return;

	int flag1 = FALSE;
	int flag2 = FALSE;

	switch(operator1) {
		case OPERATOR_EQUAL: if (in1[gx] == operand1) flag1 = TRUE; break;
		case OPERATOR_GT:    if (in1[gx] >  operand1) flag1 = TRUE; break;
		case OPERATOR_LT:    if (in1[gx] <  operand1) flag1 = TRUE; break;
		case OPERATOR_GTE:   if (in1[gx] >= operand1) flag1 = TRUE; break;
		case OPERATOR_LTE:   if (in1[gx] <= operand1) flag1 = TRUE; break;
	}

	switch(operator2) {
		case OPERATOR_EQUAL: if (in2[gx] == operand2) flag2 = TRUE; break;
		case OPERATOR_GT:    if (in2[gx] >  operand2) flag2 = TRUE; break;
		case OPERATOR_LT:    if (in2[gx] <  operand2) flag2 = TRUE; break;
		case OPERATOR_GTE:   if (in2[gx] >= operand2) flag2 = TRUE; break;
		case OPERATOR_LTE:   if (in2[gx] <= operand2) flag2 = TRUE; break;
	}

	switch(logical) {
		case LOGICAL_AND: flag1 = flag1 && flag2; break;
		case LOGICAL_OR : flag1 = flag1 || flag2; break;
	}

	if (flag1)
		out[gx] = iteration;
}

__kernel void filter(
	__global int* input,
	int2 dim,
	int iteration,
	__global int* output
) {
	int gx = get_global_id(0);
	int gy = get_global_id(1);
	int2 gxy = (int2) (gx, gy);

	int bxy = (gy/TILEH)*TILESW + gx/TILEW;

    int out;
    if (input[bxy] == iteration)
        out = 0xFF0000FF;
    else
        out = 0xFF000000;

    output[gy*dim[0] + gx] = out;
}