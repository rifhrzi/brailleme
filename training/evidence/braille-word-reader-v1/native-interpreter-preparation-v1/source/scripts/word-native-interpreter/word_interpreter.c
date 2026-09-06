/* Standalone Android diagnostic; mirrors fast-tflite 3.0.1's default C API. */
#define _POSIX_C_SOURCE 200809L
#include <tflite/c/c_api.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

static int current_case = -1;

static void fail(const char *stage) {
  fprintf(stderr, "{\"status\":\"error\",\"stage\":\"%s\",\"case_index\":%d,\"errno\":%d}\n", stage, current_case, errno);
  exit(1);
}

static unsigned char *read_bytes(const char *path, size_t *size) {
  FILE *file = fopen(path, "rb");
  if (!file || fseek(file, 0, SEEK_END)) fail("open_input");
  long length = ftell(file);
  if (length <= 0 || fseek(file, 0, SEEK_SET)) fail("input_length");
  *size = (size_t)length;
  unsigned char *bytes = malloc(*size);
  if (!bytes || fread(bytes, 1, *size, file) != *size || fclose(file)) fail("read_input");
  return bytes;
}

static void shape(const TfLiteTensor *tensor, const int *expected, size_t bytes) {
  if (!tensor || TfLiteTensorType(tensor) != kTfLiteFloat32 ||
      TfLiteTensorNumDims(tensor) != 4 || TfLiteTensorByteSize(tensor) != bytes) fail("tensor_contract");
  for (int i = 0; i < 4; i++) if (TfLiteTensorDim(tensor, i) != expected[i]) fail("tensor_shape");
}

static double seconds(void) {
  struct timespec value;
  if (clock_gettime(CLOCK_MONOTONIC, &value)) fail("clock");
  return value.tv_sec + value.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
  if (argc != 4) { fprintf(stderr, "usage: word-interpreter MODEL INPUT_LIST OUTPUT_DIRECTORY\n"); return 2; }
  const uint32_t endian = 1;
  if (*(const unsigned char *)&endian != 1 || sizeof(float) != 4) fail("float32_little_endian_required");
  size_t model_size;
  unsigned char *model_bytes = read_bytes(argv[1], &model_size);
  TfLiteModel *model = TfLiteModelCreate(model_bytes, model_size);
  if (!model) fail("model_create");
  // Exactly the options used by loadTensorflowModel(..., []): no thread,
  // delegate, XNNPACK, NNAPI, or error-reporter setters are introduced here.
  TfLiteInterpreterOptions *options = TfLiteInterpreterOptionsCreate();
  if (!options) fail("options_create");
  TfLiteInterpreter *interpreter = TfLiteInterpreterCreate(model, options);
  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);
  if (!interpreter) fail("interpreter_create");
  if (TfLiteInterpreterAllocateTensors(interpreter) != kTfLiteOk) fail("allocate_tensors");
  if (TfLiteInterpreterGetInputTensorCount(interpreter) != 1 ||
      TfLiteInterpreterGetOutputTensorCount(interpreter) != 1) fail("tensor_count");
  const int input_shape[4] = {1, 192, 480, 3}, output_shape[4] = {1, 48, 120, 11};
  const size_t input_size = 192 * 480 * 3 * sizeof(float), output_size = 48 * 120 * 11 * sizeof(float);
  TfLiteTensor *input = TfLiteInterpreterGetInputTensor(interpreter, 0);
  const TfLiteTensor *output = TfLiteInterpreterGetOutputTensor(interpreter, 0);
  shape(input, input_shape, input_size);
  shape(output, output_shape, output_size);
  FILE *list = fopen(argv[2], "r");
  if (!list) fail("input_list");
  printf("{\"kind\":\"runtime\",\"tflite_version\":\"%s\",\"model_bytes\":%zu,\"input_shape\":[1,192,480,3],\"output_shape\":[1,48,120,11],\"dtype\":\"float32_little_endian\",\"options\":\"c_api_defaults_no_setters\"}\n", TfLiteVersion(), model_size);
  fflush(stdout);
  char path[4096], destination[4096];
  unsigned int count = 0;
  while (fgets(path, sizeof(path), list)) {
    current_case = (int)count;
    size_t length = strlen(path);
    if (length == sizeof(path) - 1) fail("input_path_too_long");
    while (length && (path[length-1] == '\n' || path[length-1] == '\r')) path[--length] = '\0';
    if (!length) fail("empty_input_path");
    size_t actual_size;
    unsigned char *bytes = read_bytes(path, &actual_size);
    if (actual_size != input_size) fail("input_byte_count");
    const float *values = (const float *)bytes;
    for (size_t i = 0; i < input_size / sizeof(float); i++)
      if (!isfinite(values[i]) || values[i] < 0 || values[i] > 1) fail("input_not_rgb_float32_0_to_1");
    if (TfLiteTensorCopyFromBuffer(input, bytes, input_size) != kTfLiteOk) fail("copy_input");
    free(bytes);
    double start = seconds();
    TfLiteStatus status = TfLiteInterpreterInvoke(interpreter);
    double elapsed_ms = (seconds() - start) * 1000;
    if (status != kTfLiteOk) {
      printf("{\"kind\":\"case\",\"index\":%u,\"invoke_status\":%d,\"invoke_ms\":%.9f,\"status\":\"error\"}\n", count, status, elapsed_ms);
      fflush(stdout); fail("invoke");
    }
    output = TfLiteInterpreterGetOutputTensor(interpreter, 0);
    shape(output, output_shape, output_size);
    const float *result = (const float *)TfLiteTensorData(output);
    if (!result) fail("null_output");
    for (size_t i = 0; i < output_size / sizeof(float); i++)
      if (!isfinite(result[i]) || result[i] < 0 || result[i] > 1) fail("output_not_finite_probability");
    int written = snprintf(destination, sizeof(destination), "%s/%04u.f32", argv[3], count);
    if (written < 0 || (size_t)written >= sizeof(destination)) fail("output_path_too_long");
    FILE *file = fopen(destination, "wb");
    if (!file || fwrite(result, 1, output_size, file) != output_size || fclose(file)) fail("write_output");
    printf("{\"kind\":\"case\",\"index\":%u,\"input_bytes\":%zu,\"output_bytes\":%zu,\"invoke_status\":0,\"invoke_ms\":%.9f,\"all_finite\":true,\"probabilities_in_range\":true,\"file\":\"%04u.f32\"}\n", count, input_size, output_size, elapsed_ms, count);
    fflush(stdout);
    count++;
  }
  if (ferror(list) || fclose(list)) fail("read_input_list");
  if (!count) fail("no_inputs");
  TfLiteInterpreterDelete(interpreter);
  free(model_bytes);
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage)) fail("getrusage");
  printf("{\"kind\":\"complete\",\"cases\":%u,\"peak_rss_kib\":%ld,\"status\":\"passed_inference_only\"}\n", count, usage.ru_maxrss);
  return 0;
}
