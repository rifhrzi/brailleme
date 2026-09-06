package com.margelo.nitro.braillewordgeometry

import androidx.annotation.Keep
import com.margelo.nitro.camera.HybridPhotoSpec
import com.margelo.nitro.camera.hybrids.instances.HybridPhoto

@Keep
class HybridWordPhotoGeometry : HybridWordPhotoGeometrySpec() {
  override fun getGeometry(photo: HybridPhotoSpec): PhotoGeometry {
    val captured = photo as? HybridPhoto
      ?: throw IllegalArgumentException("Unsupported photo implementation")
    val image = captured.image
    val values = FloatArray(9)
    image.imageInfo.sensorToBufferTransformMatrix.getValues(values)
    val crop = image.cropRect
    return PhotoGeometry(
      width = image.width.toDouble(),
      height = image.height.toDouble(),
      sensorToBuffer = values.map { it.toDouble() }.toDoubleArray(),
      rotationDegrees = image.imageInfo.rotationDegrees.toDouble(),
      isMirrored = captured.isMirrored,
      cropRect = doubleArrayOf(crop.left.toDouble(), crop.top.toDouble(), crop.right.toDouble(), crop.bottom.toDouble()),
    )
  }
}
