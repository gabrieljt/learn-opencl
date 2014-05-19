#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.hpp>

#include <utility>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <iterator>
#include <memory>


const std::string message("String processed by the GPU!\n");

inline void checkErr(cl_int err, const char* name)
{
	if (err != CL_SUCCESS)
	{
		std::cerr << "ERROR: " << name << " (" << err << ")" << std::endl;
		std::cout << std::endl << "Press any key to exit..." << std::endl;
		std::cin.get();
		exit(EXIT_FAILURE);
	}		
}

int main(int argc, char** argv)
{
	// Error catcher
	cl_int err;

	// Get Platforms
	std::vector<cl::Platform> platforms;	
	cl::Platform::get(&platforms);
	checkErr(platforms.size() > 0 ? CL_SUCCESS : -1, "cl::Platform::get()");
	std::cout << platforms.size() << " OpenCL platform(s) available." << std::endl;

	std::string platformVendor;
	for (auto platform : platforms)
	{		
		platform.getInfo((cl_platform_info) CL_PLATFORM_VENDOR, &platformVendor);
		std::cout << "Platform Vendor: " << platformVendor << std::endl;
		
	}

	// Create Context
	cl_context_properties contextProperties[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};
	cl::Context context(CL_DEVICE_TYPE_GPU, contextProperties, nullptr, nullptr, &err);
	checkErr(err, "Context::Context()");
	std::cout << "OpenCL Context Created." << std::endl;

	// Create Device Buffer
	std::unique_ptr<char[]> outHost(new char[message.length() + 1]);
	char* outHostContent = outHost.get();
	for (unsigned int i = 0; i < message.length(); ++i)
		outHostContent[i] = message[i];
	outHostContent[message.length()] = '\0';
				
	cl::Buffer outCL(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, message.length() + 1, outHost.get(), &err);
	checkErr(err, "Buffer::Buffer()");
	std::cout << "OpenCL Buffer Created." << std::endl;

	// Get Devices
	std::vector<cl::Device> devices;
	devices = context.getInfo<CL_CONTEXT_DEVICES>();
	checkErr(devices.size() > 0 ? CL_SUCCESS : -1, "devices.size() > 0");

	std::string deviceInfo;
	for (auto device : devices)
	{
		device.getInfo((cl_device_info) CL_DEVICE_NAME, &deviceInfo);
		std::cout << "Device info: " << deviceInfo << std::endl;
	}

	// Kernel Program
	std::ifstream kernelFile("kernels.cl");
	checkErr(kernelFile.is_open() ? CL_SUCCESS : -1, "kernels.cl");
	std::string programString(std::istreambuf_iterator<char>(kernelFile), (std::istreambuf_iterator<char>()));
	cl::Program::Sources source(1, std::make_pair(programString.c_str(), programString.length() + 1));
	cl::Program program(context, source);
	err = program.build(devices, "");
	checkErr(err, "Program::build()");

	// Kernel Object
	cl::Kernel kernel(program, "printString", &err);
	checkErr(err, "Kernel::Kernel()");
	err = kernel.setArg(0, outCL);
	checkErr(err, "Kernel::setArg()");
	std::cout << "Kernel Object created." << std::endl;

	// CommandQueue: Host -> Kernel -> CommandQueue -> Device <-> DeviceBuffer -> Host
	cl::CommandQueue queue(context, devices[0], 0, &err);
	checkErr(err, "CommandQueue::CommandQueue()");
	cl::Event event;
	err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(message.length() + 1), cl::NDRange(1, 1), nullptr, &event);
	checkErr(err, "CommandQueue::enqueueNDRangeKernel()");
	std::cout << "Kernel Object enqueued, waiting for the GPU..." << std::endl;
	event.wait();
	std::cout << "GPU finished processing, reading from Buffer..." << std::endl;
	err = queue.enqueueReadBuffer(outCL, CL_TRUE, 0, message.length() + 1, outHost.get());
	checkErr(err, "CommandQueue::enqueueReadBuffer()");
		
	std::cout << "Message in Buffer is: " << outHost.get();
        std::cout << std::endl;
	return EXIT_SUCCESS;
}
