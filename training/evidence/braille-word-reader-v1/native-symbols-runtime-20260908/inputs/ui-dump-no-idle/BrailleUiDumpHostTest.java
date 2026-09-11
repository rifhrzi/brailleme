public final class BrailleUiDumpHostTest {
  public static void main(String[] args) {
    check(BrailleUiDump.validOutput("/data/local/tmp/brailleme-ui-v6_001.xml"));
    check(!BrailleUiDump.validOutput("/data/local/tmp/../other.xml"));
    check(!BrailleUiDump.validOutput("/sdcard/brailleme-ui-test.xml"));
    check(!BrailleUiDump.validOutput("/data/local/tmp/brailleme-ui-a.xml;id"));
    check(BrailleUiDump.safeText(null).equals(""));
    check(BrailleUiDump.safeText("Pindai <&> \u2801\ud83d\udd0d").equals("Pindai <&> \u2801\ud83d\udd0d"));
    check(BrailleUiDump.safeText("a\u0000\u0008\ud800b\udc00\ufffe\uffff\t\r\nc").equals("ab\t\r\nc"));
    System.out.println("HOST_TESTS_PASSED 7 (path and XML character guards only; no Android runtime claim)");
  }
  private static void check(boolean result) { if (!result) throw new AssertionError(); }
}
