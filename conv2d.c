#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

/* The string length of every float in the feature map. Example line: "0.594 0.934 0.212\n". 
So, 3 floats, each looks like "X.XXX" which is 5 chars, but then all have a space or new-line 
character. */
#define FLOAT_STRING_LENGTH 6

#define max(a,b) (((a) > (b)) ? (a) : (b))
#define min(a,b) (((a) < (b)) ? (a) : (b))
#define IDX(row, col, stride) ((row) * (stride) + (col))

/*
~~~~~~~~~ IMPORTANT NOTES (for report + meetings) ~~~~~~~~~

- Normally, the kernel is flipped before calculation. In all provided examples, the expected 
    output occurs when the kernel is NOT flipped. This indicates that it is reasonable to assume
    that all input kernel data has already been flipped.

- To handle evenly sized kernels (e.g., 2x2, 4x6, etc.) we use asymmetric centering.

- Need to make sure we optimise for these things: 
    1. High cache-locality,
        - Don't use 2d arrays, use 1d arrays with calculated offsets instead.
    2. Avoiding false sharing,
    3. Avoiding race conditions,
    4. Avoiding redundant calculations,
    5. Avoiding unnecessary memory allocations,
*/

typedef struct {
    float* arr;
    char* padding;
} float_array;

/*
* Extracts the dimensions from a file.
* @param filepath     The filepath where the data is stored.
* @param height       Pointer to the location where the height will be stored.
* @param width        Pointer to the location where the width will be stored.
*/
int extract_dimensions(char* filepath, int* height, int* width) {

    if (filepath == NULL) { return 1; }
    char firstline[16];

    FILE* file_ptr = fopen(filepath, "r");
    if (file_ptr == NULL){
        return 1;
    }

    // Reads the first line
    fgets(firstline, sizeof(firstline), file_ptr);
    if (strcmp(firstline, "") == 0){
        return 2;
    }

    char* token = strtok(firstline, " ");
    *height = atoi(token);
    token = strtok(NULL, " ");
    *width = atoi(token);

    fclose(file_ptr);

    return 0;
}

/* 
* Reads an input file and extracts data into an output. 
* @param filepath         The filepath where the data is stored.
* @param width            The number of elements in each line. Width.
* @param height           The number of rows. Height.
* @param padding_width    The number of zeroes to pad the width with.
* @param padding_height   The number of zeroes to pad the height with.
* @param output           The stream into which the inputs will be stored.
*/
int extract_data(char* filepath, int width, int height, int padding_width, int padding_height, float* *output) {
    
    if (filepath == NULL){ return 1; }
    FILE* file_ptr = fopen(filepath, "r");
    if (file_ptr == NULL){ return 1; }

    // Create a buffer to place extracted strings into
    const size_t buffer_size = (FLOAT_STRING_LENGTH * width) + 2; // +2 for new-line and null-byte
    
    char* buffer = (char*)malloc(buffer_size);

    // get the header line here, so we can safely ignore it later
    fgets(buffer, buffer_size, file_ptr);

    // Now loop over each line in the file
    int row_index = 0;
    while (row_index < (height + padding_height)){
        
        if(row_index < padding_height){
            row_index++;
            continue;
        }

        fgets(buffer, buffer_size, file_ptr);

        if (buffer == NULL) {
            continue;
        }

        char* token = strtok(buffer, " ");

        // Now loop over each number in the line
        int column_index = padding_width;
        while (token != NULL){
            float element = (float)atof(token);
            (*output)[IDX(row_index, column_index, width + 2 * padding_width)] = element; // Add to output.
            token = strtok(NULL, " ");
            column_index++;
        }
        row_index++;
    }
    free(buffer);
    fclose(file_ptr);
    return 0;
}

/* 
* Performs serial 2D discrete convolutions. 
* @param f             Pointer to the Feature Map.
* @param H            Height of the Feature Map.
* @param W            Width of the Feature Map.
* @param g            Pointer to the Kernel.
* @param kH           Height of the Kernel.
* @param w_padding    Width of the padding in the Feature Map.
* @param h_padding    Height of the padding in the Feature Map.
* @param output       Pointer to the location where outputs are stored.
*/
int conv2d(float* f, int H, int W, float* g, int kH, int kW, int w_padding, int h_padding, float* output){

    const int total_height = H + h_padding*2;
    const int total_width = W + w_padding*2;

    // dimensions for convolution window
    const int M = (kH / 2);
    const int N = (kW / 2);

    // Offsets allow for asymmetric centering to account for evenly sized kernels
    const int M_offset = kH % 2 == 0 ? 1 : 0;
    const int N_offset = kW % 2 == 0 ? 1 : 0;

    // Iterate over every value in the feature map
    for (int n = h_padding; n < total_height - h_padding; n++){     // feature : Iterate over Rows
        for (int k = w_padding; k < total_width - w_padding; k++){  // feature : Iterate over columns
            float result = 0.0f;
            for (int i = -M; i <= M - M_offset; i++){               // kernel : Iterate over Rows
                for (int j = -N; j <= N - N_offset; j++){           // kernel : Iterate over columns
                    //result += f[col][row] * g[i + M][j + N];
                    result += f[IDX(n + i, k + j, total_width)] * g[IDX(i + M, j + N, kW)];
                }
            }
            //output[n - h_padding][k - w_padding] = result;
            output[IDX(n - h_padding, k - w_padding, W)] = result;
        }
    }
    return 0;
}

/* 
* Performs Parallel 2D discrete convolutions. 
* @param f             Pointer to the Feature Map.
* @param H            Height of the Feature Map.
* @param W            Width of the Feature Map.
* @param g            Pointer to the Kernel.
* @param kH           Height of the Kernel.
* @param w_padding    Width of the padding in the Feature Map.
* @param h_padding    Height of the padding in the Feature Map.
* @param output       Pointer to the location where outputs are stored.
*/
int parallel_conv2d(float* f, int H, int W, float* g, int kH, int kW, int w_padding, int h_padding, float_array padded_output){

    // const int total_height = H + h_padding*2;
    // const int total_width = W + w_padding*2;

    // dimensions for convolution window
    const int M = (kH / 2);
    const int N = (kW / 2);

    // Offsets allow for asymmetric centering to account for evenly sized kernels
    const int M_offset = kH % 2 == 0 ? 1 : 0;
    const int N_offset = kW % 2 == 0 ? 1 : 0;

    #pragma omp parallel for collapse(2) schedule(dynamic, W) firstprivate(f, g, H, W, kH, kW, w_padding, h_padding, M, N, M_offset, N_offset)
    for (int n = 0; n < H; n++){
        for (int k = 0; k < W; k++){
            float result = 0.0f;
            
            #pragma omp simd collapse(2) reduction(+:result)
            for (int j = -N; j <= N - N_offset; j++){
                for (int i = -M; i <= M - M_offset; i++){

                    #define COL (n+i)
                    #define ROW (k+j)

                    // Logical zero padding
                    float f_val = 0.0f;
                    if (COL >= 0 && COL < H && ROW >= 0 && ROW < W) {
                        f_val = f[IDX(COL, ROW, W)];
                    }

                    result += f_val * g[IDX(i + M, j + N, kW)];
                    //result += f[IDX(n + i, k + j, total_width)] * g[IDX(i + M, j + N, kW)];
                }
            }
            //padded_output[n - h_padding].arr[k - w_padding] = result;
            padded_output.arr[IDX(n, k, W)] = result;
        }
    }
    return 0;
}

/*
Writes outputs to a file.
@param filepath         The filepath of where to find/put the output file.
@param outputs          A 2d array of float32s. This is what is written to the file.
@param padded_outputs   A padded 2d array of float32s. This is written to the file, instead of `outputs`, when outputting data from a parallel convolution.
@param h_dimension      The height of the outputs. Should be the same as the feature map.
@param w_dimension      The width of the outputs. Should be the same as the feature map.
*/
int write_data_to_file(char* filepath, float* outputs, float_array padded_outputs, int h_dimension, int w_dimension, int h_padding, int w_padding){
    if (filepath == NULL){ return 1; }
    FILE* file_ptr = fopen(filepath, "w");
    if (file_ptr == NULL){ return 1; }

    // Empty the file, then close it.
    fclose(file_ptr);

    // Reopen file in append mode
    file_ptr = fopen(filepath, "a");

    // Append the dimensions to the file
    fprintf(file_ptr, "%d %d\n", h_dimension, w_dimension);
    
    for (int i = h_padding; i < h_dimension + h_padding; i++){
        for (int j = w_padding; j < w_dimension + w_padding; j++){

            // Depending if paralleism is enabled or not, print the outputs
            if (outputs != NULL){
                fprintf(file_ptr, "%.3f ", outputs[IDX(i-h_padding, j-w_padding, w_dimension)]);
            } else if (padded_outputs.arr != NULL){
                fprintf(file_ptr, "%.3f ", padded_outputs.arr[IDX(i-h_padding, j-w_padding, w_dimension)]);
            } else { return 1; }
            
        }
        fprintf(file_ptr, "\n");
    }

    fclose(file_ptr);

    return 0;

}

// TODO: Delete. This is for debugging only.
void print2df(char* msg, float** arr, int size_x, int size_y){
    printf("\n%s\n", msg);
    for (int i=0; i<size_y; i++){
        for (int j=0; j<size_x; j++){
            printf("%f ", arr[i][j]);
        }
        printf("\n");
    }
}

/*
Generates a 2d array of random floats.
@param height   The height of the array.
@param width    The width of the array.
@param output   The location where the generated data will be stored.
*/
int generate_data(int height, int width, float* *output){
    
    // Reallocate required memory
    if(posix_memalign((void**)output, 64, height * width * sizeof(float)) != 0){
        return 1;
    }

    // Make a new random seed. This stops f from being the same as g when the code runs too fast.
    srand(rand());
    
    for (int i=0; i<height; i++){
        for (int j=0; j<width; j++){
            (*output)[IDX(i,j,width)] = (float)rand() / (float)RAND_MAX;
        }
    }
    return 0;
}


// Just a simple test to see if parallelisation is working
void parallel_testing(float** numbers, int height, int width, int threads){
    
    omp_set_num_threads(threads);
    printf("Threads:        %d\n", threads);
    float result = 0.0f;

    printf("\nNow doing single loop!\n");
    int chunk = 4;
    #pragma omp parallel for reduction(+:result) schedule(dynamic, chunk)
    for (int i = 0; i < height; i++){
        result += numbers[i][0];
        if (omp_get_thread_num() == 0 || 1){
            printf("Thread: %d,   Iteration: %d\n", omp_get_thread_num(), i);
        }
    }
    return;
    printf("\nNow doing Nested!\n");

    result = 0;
    chunk = 100;
    #pragma omp parallel for reduction(+:result) collapse(2) schedule(dynamic, chunk)
    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            result += numbers[i][j];
            int kH = 3;

            for (int k = 0; k < kH; k++){
                for (int l = 0; l < kH; l++){
                    if (omp_get_thread_num() == 0){
                        printf("[%d][%d],   [%d][%d],   Iteration: %d\n", i, j, k, l, l + k*3 + j*9 + i*90);
                    }
                }
            }
            
        }
    }
}



int main(int argc, char** argv) {
    
    omp_set_num_threads(4); // TODO: Maybe make this a flag?
    omp_set_nested(1); // Allow nested parallelism

    // Seed for random generation later
    srand(time(0));

    double file_start_time = omp_get_wtime();

    // Initialising variables for future use
    // TODO: we should align all of these, to avoid False Sharing
    int H = 0;
    int W = 0;
    int kH = 0;
    int kW = 0;
    char* feature_file = NULL;
    char* kernel_file = NULL;
    char* output_file = NULL;

    // DEBUG FLAGS
    int benchmark_mode = 1;    // -b ... 1=false, 0=true
    int verbose_mode = 1;      // -v
    int parallel_mode = 1;          // -p

    // Extract arguments into their variables
    for (int i = 1; i < argc; i++) {

        if (i + 1 > argc) { break; }

        // Check all flags
        if (strcmp(argv[i], "-H") == 0) {
            H = atoi(argv[++i]);
            continue;
        }
        if (strcmp(argv[i], "-W") == 0) {
            W = atoi(argv[++i]);
            continue;
        }
        if (strcmp(argv[i], "-kH") == 0) {
            kH = atoi(argv[++i]);
            continue;
        }
        if (strcmp(argv[i], "-kW") == 0) {
            kW = atoi(argv[++i]);
            continue;
        }
        if (strcmp(argv[i], "-f") == 0) {
            feature_file = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "-g") == 0) {
            kernel_file = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "-o") == 0) {
            output_file = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "-b") == 0) {
            benchmark_mode = 0;
            continue;
        }
        if (strcmp(argv[i], "-v") == 0) {
            verbose_mode = 0;
            continue;
        }
        if (strcmp(argv[i], "-p") == 0) {
            parallel_mode = 0;
            continue;
        }
    }

    
    
    /* 
    TODO: Error catching for incorrect flag usage
        
        Examples: 
        1. No flags provided, 
        2. Generating array but provided only height not width, 
        3. Not generating and provided feature but no kernel
        4. Provided output, but generated/provided no input
        5. Incompatible datatype passed through for that flag.

    
    TODO:
        - (optionally) generate inputs. Both kernel and feature map.
        - Test to see if weirdly shaped kernels also work, e.g., 5x3, 2x1, 1x1, 1x9, 50x1, 25x10, etc
        - Compile with `-Wall -Werror` flags, to catch all potential issues. Fix them all, no exceptions.
    */

    // TODO: remove this before submission, this is just for testing
    //for (int iteration = 0; iteration < 15; iteration++){

    // ~~~~~~~~~~~~~~ KERNEL ~~~~~~~~~~~~~~ // 

    // Check if we need to generate our own kernel
    
    float* kernel = NULL;

    // Generate Kernel 
    if (kH > 0 || kW > 0){

        // Allows users to specify only 1 dimension, and prevents them from inputting negative numbers
        kH = max(kH, 1);
        kW = max(kW, 1);

        // Allocate and align memory to cache lines
        if (posix_memalign((void**)&kernel, 64, kW * kH * sizeof(float)) != 0){
            // TODO: Handle error
        }

        // Randomly generate floats
        generate_data(kH, kW, &kernel);

        // If wanting to save inputs, write to kernel file
        if (kernel_file != NULL){
            if(write_data_to_file(kernel_file, kernel, (float_array){0}, kH, kW, 0, 0) != 0){
                // TODO: Handle when it can't write to kernel file
            }
        }

    // Extract Kernel
    } else if (kernel_file != NULL){

        // Allocating memory
        if (posix_memalign((void**)&kernel, 64, kW * kH * sizeof(float)) != 0){
            // TODO: Handle error
        }

        if(extract_dimensions(kernel_file, &kH, &kW) != 0) { 
            // TODO: Handle when it can't extract file dimensions
        }
        
        if(extract_data(kernel_file, kW, kH, 0, 0, &kernel) != 0){
            // TODO: Handle when can't extract kernel
        }
    }

    // This is the "same padding" that'll be added to the feature map.
    const int padding_width = kW / 2;
    const int padding_height = kH / 2;

    
    
    // ~~~~~~~~~~~~~~ FEATURE MAP ~~~~~~~~~~~~~~ // 

    float* feature_map = NULL;

    // Generate Feature Map
    if (H > 0 || W > 0){

        // Allows users to specify only 1 dimension, and prevents them from inputting negative numbers
        H = max(H, 1);
        W = max(W, 1);

        const int total_width = W + padding_width*2;
        const int total_height = H + padding_height*2;

        // Allocating memory
        if (posix_memalign((void**)&feature_map, 64, total_width * total_height * sizeof(float)) != 0){
            // TODO: Handle error
        }
        for (int i = 0; i < total_width * total_height; i++){
            feature_map[i] = 0.0f;
        }
        

        generate_data(total_height, total_width, &feature_map);

        // If wanting to save inputs, write to feature file
        if (feature_file != NULL){
            int status = write_data_to_file(feature_file, feature_map, (float_array){0}, H, W, padding_height, padding_width);
            if (status != 0){
                // TODO: Handle when it can't write to feature file
            }
        }


    // Extract Feature Map
    } else {

        // Extract dimensions of the feature map
        if (feature_file != NULL){
            int status = extract_dimensions(feature_file, &H, &W);
            if (status != 0){ 
                // TODO: Handle when it can't extract file dimensions
            }
        }

        const int total_width = W + padding_width*2;
        const int total_height = H + padding_height*2;

        // Allocate memory for the feature map of the feature map.
        if (posix_memalign((void**)&feature_map, 64, W * H * sizeof(float)) != 0){
            // TODO: Handle error
        }

        // Extract Feature Map
        if (feature_file != NULL){
            int status = extract_data(feature_file, W, H, 0, 0, &feature_map);
            if (status != 0){
                // TODO: Handle when it can't extract data
            }
        }

        
    }
        


    

    
    // ~~~~~~~~~~~~~~ conv2d() ~~~~~~~~~~~~~~ //
    

    

    if (kernel == NULL || feature_map == NULL){
        printf("To generate an output, please provide all inputs.\n");
        return 1;
    }

    // Defining output pointers
    float* outputs = NULL;      // Used for serial convolution
    float_array padded_outputs = {0}; // Used for parallel convolution

    // if (posix_memalign((void**)&padded_outputs, 64, sizeof(float_array))){
    //     // TODO: Handle error
    // }    
    

    // Parallel Convolutions
    if (parallel_mode == 0){
        
        // The size of the array padding. Used to prevent false sharing.
        // Equal to the number of bytes left over in the cache line containing the final element in float array.
        const int cache_padding_size = 64 - ((W * sizeof(float)) % 64);

        if (posix_memalign((void**)&padded_outputs.arr, 64, W * H * sizeof(float)) != 0){
            // TODO: Handle error
            printf("Error allocating memory for padded output.\n");
        }
        padded_outputs.padding = cache_padding_size == 64 ? NULL : (char*)malloc(cache_padding_size);


        // Timing begins here, because implementation only starts here.
        double start_time = omp_get_wtime();

        int status = parallel_conv2d(feature_map, H, W, kernel, kH, kW, padding_width, padding_height, padded_outputs);
        if (status != 0) {
            // TODO: Handle when can't perform convolutions
        }

        if (benchmark_mode == 0) printf("Parallel Time: %f\n", (omp_get_wtime() - start_time));
        
    // Serial Convolutions
    } else {

        if (posix_memalign((void**)&outputs, 64, W * H * sizeof(float)) != 0){
            // TODO: Handle error
            printf("Error allocating memory for outputs.\n");
        }

        double start_time = omp_get_wtime();

        int status = conv2d(feature_map, H, W, kernel, kH, kW, padding_width, padding_height, outputs);
        if (status != 0){
            // TODO: Handle when can't perform convolutions
        }

        if (benchmark_mode == 0) printf("Serial Time: %f\n", (omp_get_wtime() - start_time));
    }
        
        


    // ~~~~~~~~~~~~~~ Write to Output ~~~~~~~~~~~~~~ //

    if (output_file != NULL){
        int status = write_data_to_file(output_file, outputs, padded_outputs, H, W, 0, 0);
        if (status != 0){
            // TODO: Handle when can't write to output.
        }

        free(outputs);
        if (padded_outputs.arr != NULL) { free(padded_outputs.arr); }
        if (padded_outputs.padding != NULL) { free(padded_outputs.padding); }

    }

    free(feature_map);
    free(kernel);


    double file_end_time = omp_get_wtime();
    double file_time_taken = (file_end_time - file_start_time);
    //if (benchmark_mode == 0) printf("Total Time:    %f\n", file_time_taken);

    //} // TODO: remove this before submission, this is just for testing many iterations.
    return 0;
}