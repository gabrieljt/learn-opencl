#pragma OPENCL EXTENSION cl_khr_byte_addessable_store : enable

__kernel void printString(__global char* out)
{
	// TODO: out of bounds error check
	size_t tid = get_global_id(0);
	out[tid] = out[tid];
}