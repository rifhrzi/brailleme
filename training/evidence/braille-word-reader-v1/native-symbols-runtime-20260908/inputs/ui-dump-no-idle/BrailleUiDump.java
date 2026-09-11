import android.accessibilityservice.AccessibilityServiceInfo;
import android.app.UiAutomation;
import android.graphics.Rect;
import android.os.HandlerThread;
import android.os.Looper;
import android.os.SystemClock;
import android.util.Xml;
import android.view.accessibility.AccessibilityNodeInfo;
import java.io.ByteArrayOutputStream;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import org.xmlpull.v1.XmlSerializer;

/** One current accessibility snapshot; no waitForIdle, input injection or app changes. */
public final class BrailleUiDump {
  private static int nodes;
  private static long deadline;

  public static void main(String[] args) {
    int exit = 1;
    HandlerThread thread = null;
    UiAutomation automation = null;
    AccessibilityNodeInfo root = null;
    try {
      if (args.length != 1 || !validOutput(args[0]))
        throw new IllegalArgumentException("Expected new /data/local/tmp/brailleme-ui-NAME.xml");
      if (Files.exists(Paths.get(args[0]))) throw new IllegalArgumentException("Output already exists");
      // app_process starts a plain main thread. Android accessibility creates a
      // Handler on the main Looper even when UiAutomation uses its own thread.
      if (Looper.getMainLooper() == null) Looper.prepareMainLooper();
      thread = new HandlerThread("brailleme-ui-snapshot");
      thread.start();
      Class<?> connectionInterface = Class.forName("android.app.IUiAutomationConnection");
      Constructor<?> connectionConstructor = Class.forName("android.app.UiAutomationConnection").getDeclaredConstructor();
      connectionConstructor.setAccessible(true);
      Object connection = connectionConstructor.newInstance();
      Constructor<UiAutomation> constructor = UiAutomation.class.getDeclaredConstructor(Looper.class, connectionInterface);
      constructor.setAccessible(true);
      automation = constructor.newInstance(thread.getLooper(), connection);
      UiAutomation.class.getMethod("connect", int.class).invoke(automation, UiAutomation.FLAG_DONT_SUPPRESS_ACCESSIBILITY_SERVICES);
      AccessibilityServiceInfo service = automation.getServiceInfo();
      if (service == null) throw new IllegalStateException("No accessibility service info");
      service.flags &= ~AccessibilityServiceInfo.FLAG_INCLUDE_NOT_IMPORTANT_VIEWS;
      service.flags |= AccessibilityServiceInfo.FLAG_REPORT_VIEW_IDS;
      automation.setServiceInfo(service);
      long started = SystemClock.uptimeMillis();
      // Registration and its first window event can lag connect(). Poll only
      // missing roots, never require the continuously updating app to be idle.
      long rootDeadline = started + 5000;
      do {
        automation.clearCache();
        root = automation.getRootInActiveWindow();
        if (root != null || SystemClock.uptimeMillis() >= rootDeadline) break;
        SystemClock.sleep(Math.min(100, Math.max(1, rootDeadline - SystemClock.uptimeMillis())));
      } while (SystemClock.uptimeMillis() < rootDeadline);
      if (root == null) throw new IllegalStateException("No active accessibility root");
      nodes = 0;
      long snapshotStarted = SystemClock.uptimeMillis();
      deadline = snapshotStarted + 5000;
      ByteArrayOutputStream bytes = new ByteArrayOutputStream();
      XmlSerializer xml = Xml.newSerializer();
      xml.setOutput(bytes, "UTF-8");
      xml.startDocument("UTF-8", true);
      xml.startTag("", "hierarchy");
      xml.attribute("", "snapshot-mode", "no-idle");
      xml.attribute("", "started-uptime-ms", Long.toString(snapshotStarted));
      xml.attribute("", "root-wait-ms", Long.toString(snapshotStarted - started));
      dump(xml, root, 0, 0);
      xml.endTag("", "hierarchy");
      xml.endDocument();
      xml.flush();
      Files.write(Paths.get(args[0]), bytes.toByteArray(), StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE);
      System.out.println("UI_DUMP_WRITTEN nodes=" + nodes + " durationMs=" + (SystemClock.uptimeMillis() - started));
      exit = 0;
    } catch (Throwable error) {
      Throwable cause = error instanceof InvocationTargetException ? ((InvocationTargetException) error).getTargetException() : error;
      System.err.println("UI_DUMP_FAILED " + cause.getClass().getName() + ": " + cause.getMessage());
    } finally {
      if (root != null) root.recycle();
      if (automation != null) {
        try { UiAutomation.class.getMethod("disconnect").invoke(automation); }
        catch (Throwable error) { System.err.println("UI_DUMP_DISCONNECT_FAILED " + error.getClass().getName()); exit = 1; }
      }
      if (thread != null) thread.quitSafely();
    }
    System.exit(exit);
  }

  static boolean validOutput(String value) {
    return value.matches("/data/local/tmp/brailleme-ui-[a-zA-Z0-9_-]{1,96}\\.xml");
  }

  static String safeText(CharSequence value) {
    if (value == null) return "";
    String text = value.toString();
    StringBuilder output = new StringBuilder(text.length());
    for (int offset = 0; offset < text.length();) {
      int code = text.codePointAt(offset);
      offset += Character.charCount(code);
      if (code == 9 || code == 10 || code == 13 || (code >= 0x20 && code <= 0xD7FF)
          || (code >= 0xE000 && code <= 0xFFFD) || (code >= 0x10000 && code <= 0x10FFFF)) output.appendCodePoint(code);
    }
    return output.toString();
  }

  private static void attr(XmlSerializer xml, String name, Object value) throws Exception {
    xml.attribute("", name, safeText(value == null ? "" : value.toString()));
  }

  private static void dump(XmlSerializer xml, AccessibilityNodeInfo node, int index, int depth) throws Exception {
    if (++nodes > 10000 || depth > 128 || SystemClock.uptimeMillis() > deadline)
      throw new IllegalStateException("Accessibility tree exceeded snapshot budget");
    Rect bounds = new Rect();
    node.getBoundsInScreen(bounds);
    xml.startTag("", "node");
    attr(xml, "index", index);
    attr(xml, "text", node.isPassword() ? "" : node.getText());
    attr(xml, "resource-id", node.getViewIdResourceName());
    attr(xml, "class", node.getClassName());
    attr(xml, "package", node.getPackageName());
    attr(xml, "content-desc", node.isPassword() ? "" : node.getContentDescription());
    attr(xml, "checkable", node.isCheckable());
    attr(xml, "checked", node.isChecked());
    attr(xml, "clickable", node.isClickable());
    attr(xml, "enabled", node.isEnabled());
    attr(xml, "focusable", node.isFocusable());
    attr(xml, "focused", node.isFocused());
    attr(xml, "scrollable", node.isScrollable());
    attr(xml, "long-clickable", node.isLongClickable());
    attr(xml, "password", node.isPassword());
    attr(xml, "selected", node.isSelected());
    attr(xml, "visible-to-user", node.isVisibleToUser());
    attr(xml, "bounds", "[" + bounds.left + "," + bounds.top + "][" + bounds.right + "," + bounds.bottom + "]");
    int count = node.getChildCount();
    for (int childIndex = 0; childIndex < count; childIndex++) {
      AccessibilityNodeInfo child = node.getChild(childIndex);
      if (child == null) throw new IllegalStateException("Accessibility child changed during snapshot");
      try { dump(xml, child, childIndex, depth + 1); }
      finally { child.recycle(); }
    }
    xml.endTag("", "node");
  }
}
