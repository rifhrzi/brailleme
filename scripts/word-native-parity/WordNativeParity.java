import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Matrix;
import android.os.Build;
import java.io.File;
import java.io.FileOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import org.json.JSONArray;
import org.json.JSONObject;

/** Standalone Android ART harness. It does not install or modify the application. */
public final class WordNativeParity {
  private static JSONArray numbers(double... values) throws Exception {
    JSONArray result = new JSONArray();
    for (double value : values) result.put(value);
    return result;
  }

  private static void write(File root, String name, byte[] bytes) throws Exception {
    try (FileOutputStream stream = new FileOutputStream(new File(root, name))) { stream.write(bytes); }
  }

  private static void png(File root, String name, Bitmap bitmap) throws Exception {
    try (FileOutputStream stream = new FileOutputStream(new File(root, name))) {
      if (!bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)) throw new Exception("PNG encoding failed");
    }
  }

  private static byte[] raw(Bitmap bitmap) {
    ByteBuffer buffer = ByteBuffer.allocateDirect(bitmap.getByteCount());
    bitmap.copyPixelsToBuffer(buffer);
    buffer.rewind();
    byte[] bytes = new byte[buffer.remaining()];
    buffer.get(bytes);
    return bytes;
  }

  /** RGB channel reference from Android getPixels, independent of raw byte ordering. */
  private static byte[] rgb(Bitmap bitmap) {
    int[] colors = new int[bitmap.getWidth() * bitmap.getHeight()];
    bitmap.getPixels(colors, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
    byte[] bytes = new byte[colors.length * 3];
    for (int index = 0; index < colors.length; index++) {
      bytes[index * 3] = (byte) Color.red(colors[index]);
      bytes[index * 3 + 1] = (byte) Color.green(colors[index]);
      bytes[index * 3 + 2] = (byte) Color.blue(colors[index]);
    }
    return bytes;
  }

  private static JSONObject runCase(File root, Bitmap base, int rotation, boolean mirror, double aspect, JSONObject reference) throws Exception {
    String id = "rotation-" + rotation + "-mirror-" + mirror + "-aspect-" + (int) aspect;
    // These are the same Android Bitmap operations used by HybridPhoto.toImage().
    Matrix matrix = new Matrix();
    if (mirror) matrix.preScale(-1f, 1f);
    if (rotation != 0) matrix.postRotate((float) rotation);
    Bitmap upright = matrix.isIdentity() ? base : Bitmap.createBitmap(base, 0, 0, base.getWidth(), base.getHeight(), matrix, false);
    double iw = upright.getWidth(), ih = upright.getHeight(), pw = 320, ph = 320;
    double gh = Math.min(pw * .9 / 8, ph * .3), gw = gh * aspect;
    double scale = Math.max(pw / iw, ph / ih);
    double sx = ((pw - gw) / 2 + (iw * scale - pw) / 2) / scale;
    double sy = ((ph - gh) / 2 + (ih * scale - ph) / 2) / scale;
    double sw = gw / scale, sh = gh / scale;
    double cw = Math.max(sw, sh * 2.5), ch = cw / 2.5;
    double cx = Math.max(0, Math.min(iw - cw, sx + sw / 2 - cw / 2));
    double cy = Math.max(0, Math.min(ih - ch, sy + sh / 2 - ch / 2));
    int left = (int) Math.floor(cx), top = (int) Math.floor(cy);
    int right = Math.min((int) iw, (int) Math.ceil(cx + cw));
    int bottom = Math.min((int) ih, (int) Math.ceil(cy + ch));
    JSONArray independentCrop = numbers(left, top, right, bottom);
    // Native production code receives these integer crop coordinates from TypeScript.
    // Use the actual runtime values, including its normalized floating-point round trip.
    JSONArray coordinates = reference.getJSONArray("context_crop_pixels");
    left = coordinates.getInt(0); top = coordinates.getInt(1);
    right = coordinates.getInt(2); bottom = coordinates.getInt(3);
    Bitmap crop = Bitmap.createBitmap(upright, left, top, right - left, bottom - top);
    // androidx.core.graphics.Bitmap.scale(w,h,true) delegates to this operation.
    Bitmap resized = Bitmap.createScaledBitmap(crop, 480, 192, true);
    png(root, id + "-upright.png", upright);
    png(root, id + "-crop.png", crop);
    write(root, id + "-crop.raw", raw(crop));
    png(root, id + "-resized.png", resized);
    write(root, id + ".rgb", rgb(resized));
    write(root, id + ".raw", raw(resized));
    JSONObject row = new JSONObject();
    row.put("id", id).put("rotation_clockwise", rotation).put("mirror_before_rotation", mirror);
    row.put("guide_aspect", aspect).put("upright_size", numbers(iw, ih));
    row.put("selection_photo", numbers(sx / iw, sy / ih, sw / iw, sh / ih));
    row.put("context_crop_pixels", numbers(left, top, right, bottom));
    row.put("independent_pixel_formula_crop", independentCrop);
    row.put("selection_context", numbers((sx - left) / (right - left), (sy - top) / (bottom - top), sw / (right - left), sh / (bottom - top)));
    row.put("row_bytes", resized.getRowBytes()).put("byte_count", resized.getByteCount());
    if (resized != crop) resized.recycle();
    if (crop != upright) crop.recycle();
    if (upright != base) upright.recycle();
    return row;
  }

  public static void main(String[] args) throws Exception {
    if (args.length != 2) throw new IllegalArgumentException("Pass output directory and TypeScript geometry JSON paths.");
    File root = new File(args[0]);
    if (!root.isDirectory() && !root.mkdirs()) throw new Exception("Cannot create output directory");
    JSONArray references = new JSONObject(new String(Files.readAllBytes(Paths.get(args[1])), StandardCharsets.UTF_8)).getJSONArray("cases");
    Bitmap color = Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888);
    color.setPixel(0, 0, Color.rgb(11, 22, 33));
    JSONArray colorBytes = new JSONArray();
    for (byte value : raw(color)) colorBytes.put(value & 255);
    color.recycle();

    int width = 1440, height = 1920;
    int[] pixels = new int[width * height];
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        pixels[y * width + x] = Color.rgb((3 * x + 5 * y) & 255, (11 * x + 2 * y) & 255, ((x / 3 + y / 5) & 1) * 255);
      }
    }
    Bitmap base = Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888);
    png(root, "source.png", base);
    JSONArray cases = new JSONArray();
    int referenceIndex = 0;
    for (double aspect : new double[] {1, 4, 8}) cases.put(runCase(root, base, 0, false, aspect, references.getJSONObject(referenceIndex++)));
    for (int rotation : new int[] {90, 180, 270}) cases.put(runCase(root, base, rotation, false, 4, references.getJSONObject(referenceIndex++)));
    cases.put(runCase(root, base, 90, true, 4, references.getJSONObject(referenceIndex)));
    base.recycle();

    JSONObject report = new JSONObject();
    report.put("scope", "Standalone Android Bitmap primitives with synthetic inputs; not a camera, Nitro bridge, or model accuracy test.");
    report.put("sdk", Build.VERSION.SDK_INT).put("device", Build.MODEL).put("fingerprint", Build.FINGERPRINT);
    report.put("native_byte_order", ByteOrder.nativeOrder().toString());
    report.put("argb8888_raw_bytes_for_rgb_11_22_33", colorBytes);
    report.put("cases", cases);
    write(root, "native.json", report.toString(2).getBytes(StandardCharsets.UTF_8));
    System.out.println("WORD_NATIVE_PARITY_OK " + cases.length() + " cases; bytes=" + colorBytes.toString());
  }
}
