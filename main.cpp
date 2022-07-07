#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <CImg.h>

using namespace std;
using namespace cimg_library;

struct rgba_pixel {
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
};

constexpr unsigned int r_channel_idx = 0;
constexpr unsigned int g_channel_idx = 1;
constexpr unsigned int b_channel_idx = 2;
constexpr unsigned int a_channel_idx = 3;

std::vector<rgba_pixel> convert_cimg_to_rgba_buffer(const cimg_library::CImg<unsigned char>& img) {
    const unsigned int img_height = static_cast<unsigned int>(img.height());
    const unsigned int img_width = static_cast<unsigned int>(img.width());
    const unsigned int number_of_channels = static_cast<unsigned int>(img.spectrum());

    const bool has_r_channel = number_of_channels > r_channel_idx;
    const bool has_g_channel = number_of_channels > g_channel_idx;
    const bool has_b_channel = number_of_channels > b_channel_idx;
    const bool has_a_channel = number_of_channels > a_channel_idx;

    std::vector<rgba_pixel> rgba_buf(static_cast<std::size_t>(img_width) * img_height);
    for (unsigned int y = 0; y < img_height; ++y) {
        for (unsigned int x = 0; x < img_width; ++x) {
            const std::size_t pixel_idx = static_cast<std::size_t>(img_width) * y + x;
            rgba_buf[pixel_idx].r = has_r_channel ? *img.data(x, y, 0, r_channel_idx) : 0;
            rgba_buf[pixel_idx].g = has_g_channel ? *img.data(x, y, 0, g_channel_idx) : 0;
            rgba_buf[pixel_idx].b = has_b_channel ? *img.data(x, y, 0, b_channel_idx) : 0;
            rgba_buf[pixel_idx].a = has_a_channel ? *img.data(x, y, 0, a_channel_idx) : UCHAR_MAX;
        }
    }
    return rgba_buf;
}

void copy_rgba_buffer_to_cimg(const std::vector<rgba_pixel>& rgba_buf, cimg_library::CImg<unsigned char>& img) {
    const unsigned int img_height = static_cast<unsigned int>(img.height());
    const unsigned int img_width = static_cast<unsigned int>(img.width());
    const unsigned int number_of_channels = static_cast<unsigned int>(img.spectrum());

    const bool has_r_channel = number_of_channels > r_channel_idx;
    const bool has_g_channel = number_of_channels > g_channel_idx;
    const bool has_b_channel = number_of_channels > b_channel_idx;
    const bool has_a_channel = number_of_channels > a_channel_idx;

    for (unsigned int y = 0; y < img_height; ++y) {
        for (unsigned int x = 0; x < img_width; ++x) {
            const std::size_t pixel_idx = static_cast<std::size_t>(img_width) * y + x;
            if (has_r_channel) *img.data(x, y, 0, r_channel_idx) = rgba_buf[pixel_idx].r;
            if (has_g_channel) *img.data(x, y, 0, g_channel_idx) = rgba_buf[pixel_idx].g;
            if (has_b_channel) *img.data(x, y, 0, b_channel_idx) = rgba_buf[pixel_idx].b;
            if (has_a_channel) *img.data(x, y, 0, a_channel_idx) = rgba_buf[pixel_idx].a;
        }
    }
}

int main(int argc, char* argv[]){
    const char *kernel_source = "\n" \
    "__constant sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;\n" \
        "__kernel void img_turn(\n" \
            "__read_only image2d_t I,\n" \
            "__write_only image2d_t O\n" \
        ")\n" \
        "{\n" \
            "int gid_x = get_global_id(0);\n" \
            "int gid_y = get_global_id(1);\n" \
            "int w = get_image_width(I);\n" \
            "int h = get_image_height(I);\n" \
            "if (gid_x >= w || gid_y >= h)\n" \
                "return;\n" \
            "uint4 p = read_imageui(I, sampler, (int2)(gid_x, gid_y));\n" \
            "write_imageui(O, (int2)(gid_x, h - gid_y - 1), p);\n" \
        "}\n" \
        "\n";

    cl_context context;
    cl_platform_id platform;
    cl_uint num_platforms;
    cl_device_id device;
    cl_command_queue command_queue;
    cl_program program;
    cl_kernel kernel;
    cl_int errNum;

    const char* img_file_name = argc > 1 ? argv[1]: "lena.png";

    errNum = clGetPlatformIDs(1, &platform, &num_platforms);
    if(errNum != CL_SUCCESS){
       cerr << "Errore nel prendere la piattaforma OpenCL";
    }

    errNum = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if(errNum != CL_SUCCESS){
       cerr << "Errore nel prendere il device OpenCL";
    }

    char* value;
    size_t valueSize;

    clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &valueSize);
    value = (char*) malloc(valueSize);
    clGetDeviceInfo(device, CL_DEVICE_NAME, valueSize, value, NULL);
    cout << "Device: " << value << endl;
    free(value);

    cl_context_properties properties [] = {
       CL_CONTEXT_PLATFORM, (cl_context_properties) platform, 0
    };

    context = clCreateContext(properties, 1, &device, NULL, NULL, &errNum);
    if(errNum != CL_SUCCESS){
        cerr << "Errore nel context OpenCL";
    }

    const cl_queue_properties props(CL_QUEUE_PROFILING_ENABLE);

    command_queue = clCreateCommandQueueWithProperties(context, device, &props, &errNum);
    if(errNum != CL_SUCCESS){
        cout << "Errore nella creazione command queue OpenCL!" << endl;
    }

    program = clCreateProgramWithSource(context, 1, (const char **) &kernel_source, NULL, &errNum);
    if(errNum != CL_SUCCESS){
        cout << "Errore nella creazione del programma OpenCL!" << endl;
    }

    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(errNum != CL_SUCCESS){
        cout << "Errore nella build dell'eseguibile OpenCL!" << endl;
    }

    kernel = clCreateKernel(program, "img_turn", &errNum);
    if(errNum != CL_SUCCESS){
        cout << "Errore nel kernel OpenCL!" << endl;
    }

    // INPUT IMAGE //

    CImg<unsigned char> img_in(img_file_name);

    cl_image_format format = {
        CL_RGBA,
        CL_UNSIGNED_INT8,
    };

    cl_image_desc desc = {
        .image_type = CL_MEM_OBJECT_IMAGE2D,
        .image_width = (size_t) img_in.width(),
        .image_height = (size_t) img_in.height(),
        .image_row_pitch = 0,
        .image_slice_pitch = 0,
        .num_mip_levels = 0,
        .num_samples = 0,
        .buffer = NULL,
    };
    
    cl_mem input_img = clCreateImage(
        context,
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        (const cl_image_format *) &format,
        (const cl_image_desc *) &desc,
        img_in.data(),
        &errNum
    );

    if(errNum != CL_SUCCESS){
        cerr << "Errore nella creazione dell'immagine di input!: " << errNum << endl;
    }

    // OUTPUT IMAGE //

    CImg<unsigned char> img_out(img_in.width(), img_in.height(), 1, 4);
    //cout << img_out.data() << endl;

    format = {
        CL_RGBA,
        CL_UNSIGNED_INT8,
    };

    desc = {
        .image_type = CL_MEM_OBJECT_IMAGE2D,
        .image_width = (size_t) img_out.width(),
        .image_height = (size_t) img_out.height(),
        .image_row_pitch = 0,
        .image_slice_pitch = 0,
        .num_mip_levels = 0,
        .num_samples = 0,
        .buffer = NULL,
    };

    cl_mem output_img = clCreateImage(
        context,
        CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
        (const cl_image_format *) &format,
        (const cl_image_desc *) &desc,
        img_out.data(),
        &errNum
    );

    if(errNum != CL_SUCCESS){
        cerr << "Errore nella creazione dell'immagine di output!: " << errNum << endl;
    }

    size_t origins[3] = {0, 0, 0};
    size_t region[3] = {(size_t) img_in.width(), (size_t) img_in.height(), (size_t) 1};

    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), input_img);
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), output_img);

    size_t global[2] = {(size_t) img_in.width(), (size_t) img_in.height()};
    clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global, NULL, 0, NULL, NULL);

    auto rgba_buf = convert_cimg_to_rgba_buffer(img_in);
    
    errNum = clEnqueueWriteImage(command_queue, input_img, CL_TRUE, origins, region, 0, 0, rgba_buf.data(), 0, NULL, NULL);
    errNum = clEnqueueReadImage(command_queue, output_img, CL_TRUE, origins, region, 0, 0, rgba_buf.data(), 0, NULL, NULL);

    copy_rgba_buffer_to_cimg(rgba_buf, img_out);
    img_out.save("./output_img.png");

    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
    clReleaseDevice(device);
    clReleaseKernel(kernel);
    clReleaseProgram(program);

    clReleaseMemObject(input_img);
    clReleaseMemObject(output_img);

    return 0;

}