package com.margelo.nitro.braillewordgeometry;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** Stateless Pillow-compatible 22-bit, two-pass uint8 bilinear antialias resizer. */
public final class WordPixelResize {
  public static final int WIDTH = 480;
  public static final int HEIGHT = 192;
  private static final int PRECISION = 4194304;
  private static final int ROUNDING = 2097152;

  private WordPixelResize() {}

  public static final class Result {
    public final ByteBuffer tensor;
    public final double meanLuminance;
    public final double contrast;
    Result(ByteBuffer tensor, double meanLuminance, double contrast) {
      this.tensor = tensor;
      this.meanLuminance = meanLuminance;
      this.contrast = contrast;
    }
  }

  private static final class Weights {
    final int[] starts;
    final int[] counts;
    final int[] values;
    final int kernelSize;
    Weights(int inputSize, int outputSize) {
      double scale = (double) inputSize / outputSize;
      double support = Math.max(1, scale), inverseSupport = 1 / support;
      kernelSize = (int) Math.ceil(support) * 2 + 1;
      starts = new int[outputSize];
      counts = new int[outputSize];
      values = new int[Math.multiplyExact(outputSize, kernelSize)];
      for (int target = 0; target < outputSize; target++) {
        double center = (target + 0.5) * scale;
        // Cast truncates toward zero, matching Math.trunc in the TypeScript oracle.
        int start = Math.max(0, (int) (center - support + 0.5));
        int end = Math.min(inputSize, (int) (center + support + 0.5));
        double total = 0;
        for (int index = start; index < end; index++) total += Math.max(0, 1 - Math.abs((index - center + 0.5) * inverseSupport));
        starts[target] = start;
        counts[target] = end - start;
        for (int index = start; index < end; index++) {
          double weight = Math.max(0, 1 - Math.abs((index - center + 0.5) * inverseSupport)) / total;
          values[target * kernelSize + index - start] = (int) Math.floor(weight * PRECISION + 0.5);
        }
      }
    }
  }

  private static int dimension(double value) {
    if (!Double.isFinite(value) || value <= 0 || value != Math.rint(value) || value > Integer.MAX_VALUE) {
      throw new IllegalArgumentException("Dimensi pixel foto tidak valid.");
    }
    return (int) value;
  }

  public static Result resize(ByteBuffer buffer, double widthValue, double heightValue, String pixelFormat, boolean androidBitmap) {
    int width = dimension(widthValue), height = dimension(heightValue);
    String format = androidBitmap && "BGRA".equals(pixelFormat) ? "RGBA" : pixelFormat;
    int stride, red, green, blue;
    if (format == null) throw new IllegalArgumentException("Format pixel foto tidak didukung.");
    switch (format) {
      case "ARGB": case "XRGB": stride = 4; red = 1; green = 2; blue = 3; break;
      case "BGRA": case "BGRX": stride = 4; red = 2; green = 1; blue = 0; break;
      case "ABGR": case "XBGR": stride = 4; red = 3; green = 2; blue = 1; break;
      case "RGBA": case "RGBX": stride = 4; red = 0; green = 1; blue = 2; break;
      case "RGB": stride = 3; red = 0; green = 1; blue = 2; break;
      case "BGR": stride = 3; red = 2; green = 1; blue = 0; break;
      default: throw new IllegalArgumentException("Format pixel foto tidak didukung.");
    }
    long expectedBytes = (long) width * height * stride;
    if (buffer == null || expectedBytes > Integer.MAX_VALUE || buffer.limit() != expectedBytes || (long) WIDTH * height * 3 > Integer.MAX_VALUE) {
      throw new IllegalArgumentException("Ukuran buffer pixel foto tidak sesuai.");
    }
    // Own the short-lived byte copy. Never retain the caller's frame/bitmap buffer.
    ByteBuffer input = buffer.duplicate();
    input.position(0);
    byte[] source = new byte[(int) expectedBytes];
    input.get(source);
    Weights horizontal = new Weights(width, WIDTH), vertical = new Weights(height, HEIGHT);
    byte[] intermediate = new byte[WIDTH * height * 3];
    for (int row = 0; row < height; row++) {
      for (int column = 0; column < WIDTH; column++) {
        long r = ROUNDING, g = ROUNDING, b = ROUNDING;
        int start = (row * width + horizontal.starts[column]) * stride;
        int count = horizontal.counts[column], kernel = column * horizontal.kernelSize;
        for (int tap = 0; tap < count; tap++) {
          int index = start + tap * stride, weight = horizontal.values[kernel + tap];
          r += (long) (source[index + red] & 255) * weight;
          g += (long) (source[index + green] & 255) * weight;
          b += (long) (source[index + blue] & 255) * weight;
        }
        int target = (row * WIDTH + column) * 3;
        intermediate[target] = (byte) Math.min(255, r / PRECISION);
        intermediate[target + 1] = (byte) Math.min(255, g / PRECISION);
        intermediate[target + 2] = (byte) Math.min(255, b / PRECISION);
      }
    }
    byte[] resized = new byte[WIDTH * HEIGHT * 3];
    int rowStride = WIDTH * 3;
    for (int row = 0; row < HEIGHT; row++) {
      int start = vertical.starts[row] * rowStride, count = vertical.counts[row], kernel = row * vertical.kernelSize;
      for (int column = 0; column < rowStride; column++) {
        long value = ROUNDING;
        for (int tap = 0; tap < count; tap++) value += (long) (intermediate[start + tap * rowStride + column] & 255) * vertical.values[kernel + tap];
        resized[row * rowStride + column] = (byte) Math.min(255, value / PRECISION);
      }
    }
    ByteBuffer tensor = ByteBuffer.allocateDirect(resized.length * Float.BYTES).order(ByteOrder.nativeOrder());
    double luminanceSum = 0, luminanceSquaredSum = 0;
    for (int index = 0; index < resized.length; index += 3) {
      double r = (resized[index] & 255) / 255.0, g = (resized[index + 1] & 255) / 255.0, b = (resized[index + 2] & 255) / 255.0;
      tensor.putFloat((float) r); tensor.putFloat((float) g); tensor.putFloat((float) b);
      double luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      luminanceSum += luminance;
      luminanceSquaredSum += luminance * luminance;
    }
    tensor.flip();
    double pixels = WIDTH * HEIGHT, mean = luminanceSum / pixels;
    return new Result(tensor, mean, Math.sqrt(Math.max(0, luminanceSquaredSum / pixels - mean * mean)));
  }
}
