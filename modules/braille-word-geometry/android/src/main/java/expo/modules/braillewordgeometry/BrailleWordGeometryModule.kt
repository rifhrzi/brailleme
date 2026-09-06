package expo.modules.braillewordgeometry

import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition

class BrailleWordGeometryModule : Module() {
  override fun definition() = ModuleDefinition {
    Name("BrailleWordGeometry")
    Function("initialize") {
      System.loadLibrary("BrailleWordGeometry")
    }
  }
}
