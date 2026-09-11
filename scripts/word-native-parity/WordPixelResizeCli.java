import com.margelo.nitro.braillewordgeometry.WordPixelResize;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

/** Host-JVM oracle driver; does not load Android, JNI, Nitro or a camera. */
public final class WordPixelResizeCli {
  private static void rejects(Runnable action) {
    try { action.run(); } catch (IllegalArgumentException expected) { return; }
    throw new AssertionError("Expected invalid input rejection");
  }
  private static void selfTest() {
    ByteBuffer pixel = ByteBuffer.wrap(new byte[] { 12, 34, 56 });
    for (double dimension : new double[] { 0, -1, 1.5, Double.NaN, Double.POSITIVE_INFINITY, 2147483648.0 }) {
      rejects(() -> WordPixelResize.resize(pixel, dimension, 1, "RGB", false));
      rejects(() -> WordPixelResize.resize(pixel, 1, dimension, "RGB", false));
    }
    rejects(() -> WordPixelResize.resize(null, 1, 1, "RGB", false));
    rejects(() -> WordPixelResize.resize(pixel, 1, 1, "invalid", false));
    rejects(() -> WordPixelResize.resize(pixel, 2, 1, "RGB", false));
    rejects(() -> WordPixelResize.resize(pixel, 1, 1, "RGBA", false));
    pixel.position(2);
    WordPixelResize.Result first = WordPixelResize.resize(pixel, 1, 1, "RGB", false);
    if (pixel.position() != 2) throw new AssertionError("Input cursor mutated");
    float saved = first.tensor.getFloat(0);
    WordPixelResize.Result second = WordPixelResize.resize(ByteBuffer.wrap(new byte[] { 90, 80, 70 }), 1, 1, "RGB", false);
    if (first.tensor == second.tensor || first.tensor.getFloat(0) != saved) throw new AssertionError("Output shared across calls");
    if (!first.tensor.isDirect() || first.tensor.capacity() != 480 * 192 * 3 * 4 || first.tensor.order() != ByteOrder.nativeOrder()) throw new AssertionError("Wrong output contract");
    System.out.println("{\"invalid_input_and_ownership_passed\":true,\"invalid_cases\":16}");
  }
  public static void main(String[] args) throws Exception {
    if (args.length == 1 && args[0].equals("--self-test")) { selfTest(); return; }
    if (args.length != 6) throw new IllegalArgumentException("width height format androidBitmap input.bin output-prefix");
    double width = Double.parseDouble(args[0]), height = Double.parseDouble(args[1]);
    String format = args[2]; boolean correction = Boolean.parseBoolean(args[3]);
    ByteBuffer input = ByteBuffer.wrap(Files.readAllBytes(Path.of(args[4])));
    for (int i = 0; i < 3; i++) WordPixelResize.resize(input, width, height, format, correction);
    double[] timings = new double[5];
    WordPixelResize.Result result = null;
    for (int i = 0; i < timings.length; i++) {
      long start = System.nanoTime();
      result = WordPixelResize.resize(input, width, height, format, correction);
      timings[i] = (System.nanoTime() - start) / 1e6;
    }
    byte[] tensor = new byte[result.tensor.remaining()];
    result.tensor.duplicate().get(tensor);
    Files.write(Path.of(args[5] + ".java.f32"), tensor);
    byte[] rgb = new byte[tensor.length / 4];
    for (int i = 0; i < rgb.length; i++) rgb[i] = (byte) Math.round(result.tensor.getFloat(i * 4) * 255.0);
    Files.write(Path.of(args[5] + ".java.rgb"), rgb);
    Arrays.sort(timings);
    System.out.println("{\"meanLuminance\":" + result.meanLuminance + ",\"contrast\":" + result.contrast
      + ",\"host_java_median_ms\":" + timings[2] + ",\"host_java_max_ms\":" + timings[4] + "}");
  }
}
