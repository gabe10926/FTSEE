package sigmacorns.GlobalShutterVision

import android.graphics.Canvas
import org.firstinspires.ftc.robotcore.internal.camera.calibration.CameraCalibration
import org.firstinspires.ftc.vision.VisionProcessor
import org.opencv.core.*
import org.opencv.imgproc.Imgproc

/**
 * AprilTag-anchored ROI ball detector for FTC Decode 2025-2026.
 * Detects GREEN and PURPLE balls in the blue goal ramp area.
 *
 * Port of Python OpenCV pipeline to FTC VisionProcessor.
 * AprilTag detection is handled separately (by Limelight or AprilTagProcessor).
 * Call [setTagCorners] each frame with the detected tag corners to anchor the ROI.
 */
class EyeBrain : VisionProcessor {

    // ========================================================================
    // COLOR THRESHOLDS (from vision.py)
    // ========================================================================

    // BLUE / RED / YELLOW (HSV)
    private val blueHsvLower = Scalar(100.0, 80.0, 60.0)
    private val blueHsvUpper = Scalar(130.0, 255.0, 255.0)

    private val redHsvLower1 = Scalar(0.0, 80.0, 60.0)
    private val redHsvUpper1 = Scalar(12.0, 255.0, 255.0)
    private val redHsvLower2 = Scalar(170.0, 80.0, 60.0)
    private val redHsvUpper2 = Scalar(180.0, 255.0, 255.0)

    private val yellowHsvLower = Scalar(18.0, 80.0, 60.0)
    private val yellowHsvUpper = Scalar(38.0, 255.0, 255.0)

    // Detection area bounds
    // 3.6 in red/blue balls and 2.8 in yellow balls; yellow is roughly 60% of the area.
    private val minBallAreaBlueRed = 300.0
    private val minBallAreaYellow = 180.0
    private val maxBallArea = 20000.0

    // ROI multipliers (of tag size) - ramp is LEFT and BELOW the tag
    var leftMult = 4.0
    var rightMult = 1.0
    var upMult = 0.5
    var downMult = 3.0

    // ========================================================================
    // STATE
    // ========================================================================

    /** Latest detection results, read from OpMode. */
    @Volatile
    var detections: List<BallDetection> = emptyList()
        private set

    /** Motif string e.g. "GGPG" */
    @Volatile
    var motif: String = ""
        private set

    // Tag corners fed externally (e.g. from Limelight or AprilTagProcessor)
    // Array of 4 Points: TL, TR, BR, BL
    @Volatile
    private var tagCorners: Array<Point>? = null

    private var lastRoi: MatOfPoint? = null

    // Reusable Mats to avoid per-frame allocation
    private val hsv = Mat()
    private val blueMask = Mat()
    private val redMask1 = Mat()
    private val redMask2 = Mat()
    private val redMask = Mat()
    private val yellowMask = Mat()
    private val roiMask = Mat()
    private val hierarchy = Mat()

    // ========================================================================
    // PUBLIC API
    // ========================================================================

    /**
     * Feed AprilTag corner data each frame.
     * @param corners 4 points in order: top-left, top-right, bottom-right, bottom-left
     *                (pixel coordinates in the camera frame). Pass null if tag not visible.
     */
    fun setTagCorners(corners: Array<Point>?) {
        tagCorners = corners
    }

    // ========================================================================
    // VisionProcessor interface
    // ========================================================================

    override fun init(width: Int, height: Int, calibration: CameraCalibration) {
        // nothing needed
    }

    override fun processFrame(frame: Mat, captureTimeNanos: Long): Any? {
        val corners = tagCorners
        val roi: MatOfPoint

        if (corners != null && corners.size == 4) {
            roi = computeRoi(corners, frame.size())
            lastRoi = roi
        } else if (lastRoi != null) {
            roi = lastRoi!!
        } else {
            detections = emptyList()
            motif = ""
            return null
        }

        // Build ROI mask
        if (roiMask.empty() || roiMask.rows() != frame.rows() || roiMask.cols() != frame.cols()) {
            roiMask.create(frame.size(), CvType.CV_8UC1)
        }
        roiMask.setTo(Scalar(0.0))
        Imgproc.fillPoly(roiMask, listOf(roi), Scalar(255.0))

        // Color conversions
        // Note: FTC VisionPortal delivers frames as RGB, not BGR
        Imgproc.cvtColor(frame, hsv, Imgproc.COLOR_RGB2HSV)

        Core.inRange(hsv, blueHsvLower, blueHsvUpper, blueMask)
        Core.inRange(hsv, redHsvLower1, redHsvUpper1, redMask1)
        Core.inRange(hsv, redHsvLower2, redHsvUpper2, redMask2)
        Core.bitwise_or(redMask1, redMask2, redMask)
        Core.inRange(hsv, yellowHsvLower, yellowHsvUpper, yellowMask)

        Core.bitwise_and(blueMask, roiMask, blueMask)
        Core.bitwise_and(redMask, roiMask, redMask)
        Core.bitwise_and(yellowMask, roiMask, yellowMask)

        // Find contours and build detections
        val result = mutableListOf<BallDetection>()
        val candidates = listOf(
            blueMask to ("B" to minBallAreaBlueRed),
            redMask to ("R" to minBallAreaBlueRed),
            yellowMask to ("Y" to minBallAreaYellow)
        )

        for ((mask, pair) in candidates) {
            val contours = mutableListOf<MatOfPoint>()
            Imgproc.findContours(mask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
            val (color, minArea) = pair
            for (cnt in contours) {
                val area = Imgproc.contourArea(cnt)
                if (area >= minArea && area <= maxBallArea) {
                    val rect = Imgproc.boundingRect(cnt)
                    val cx = rect.x + rect.width / 2
                    val cy = rect.y + rect.height / 2
                    result.add(BallDetection(cx, cy, area, color))
                }
                cnt.release()
            }
        }

        // Sort left-to-right
        result.sortBy { it.centerX }

        detections = result
        motif = result.joinToString("") { it.color }

        // Draw ROI and detections on frame for camera stream preview
        Imgproc.polylines(frame, listOf(roi), true, Scalar(255.0, 0.0, 0.0), 2)
        var bCount = 0; var rCount = 0; var yCount = 0
        for (det in result) {
            val drawColor: Scalar
            val label: String
            when (det.color) {
                "B" -> { bCount++; drawColor = Scalar(255.0, 0.0, 0.0); label = "B$bCount" }
                "R" -> { rCount++; drawColor = Scalar(0.0, 0.0, 255.0); label = "R$rCount" }
                else -> { yCount++; drawColor = Scalar(0.0, 255.0, 255.0); label = "Y$yCount" }
            }
            Imgproc.circle(frame, Point(det.centerX.toDouble(), det.centerY.toDouble()), 10, drawColor, 2)
            Imgproc.putText(frame, label, Point(det.centerX - 15.0, det.centerY - 20.0),
                Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, drawColor, 2)
        }
        Imgproc.putText(frame, "Motif: $motif (${result.size})", Point(10.0, 30.0),
            Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255.0, 255.0, 0.0), 2)

        return detections
    }

    override fun onDrawFrame(canvas: Canvas, onscreenWidth: Int, onscreenHeight: Int,
                             scaleBmpPxToCanvasPx: Float, scaleCanvasDensity: Float, userContext: Any?) {
        // Drawing is done in processFrame via OpenCV on the Mat directly
    }

    // ========================================================================
    // ROI COMPUTATION
    // ========================================================================

    private fun computeRoi(tagCorners: Array<Point>, frameSize: Size): MatOfPoint {
        val (tl, tr, br, bl) = tagCorners

        // Direction vectors (averaged across tag edges)
        val rightVecX = ((tr.x - tl.x) + (br.x - bl.x)) / 2
        val rightVecY = ((tr.y - tl.y) + (br.y - bl.y)) / 2
        val downVecX = ((bl.x - tl.x) + (br.x - tr.x)) / 2
        val downVecY = ((bl.y - tl.y) + (br.y - tr.y)) / 2

        val rightLen = Math.sqrt(rightVecX * rightVecX + rightVecY * rightVecY)
        val downLen = Math.sqrt(downVecX * downVecX + downVecY * downVecY)

        val rdx = if (rightLen > 0) rightVecX / rightLen else 1.0
        val rdy = if (rightLen > 0) rightVecY / rightLen else 0.0
        val ddx = if (downLen > 0) downVecX / downLen else 0.0
        val ddy = if (downLen > 0) downVecY / downLen else 1.0

        val tagW = rightLen
        val tagH = downLen

        // Offset vectors
        val leftOx = rdx * tagW * leftMult;  val leftOy = rdy * tagW * leftMult
        val rightOx = rdx * tagW * rightMult; val rightOy = rdy * tagW * rightMult
        val upOx = ddx * tagH * upMult;   val upOy = ddy * tagH * upMult
        val downOx = ddx * tagH * downMult; val downOy = ddy * tagH * downMult

        val w = frameSize.width - 1
        val h = frameSize.height - 1

        fun clamp(p: Point) = Point(p.x.coerceIn(0.0, w), p.y.coerceIn(0.0, h))

        val roiTL = clamp(Point(tl.x - leftOx - upOx, tl.y - leftOy - upOy))
        val roiTR = clamp(Point(tr.x + rightOx - upOx, tr.y + rightOy - upOy))
        val roiBR = clamp(Point(br.x + rightOx + downOx, br.y + rightOy + downOy))
        val roiBL = clamp(Point(bl.x - leftOx + downOx, bl.y - leftOy + downOy))

        return MatOfPoint(roiTL, roiTR, roiBR, roiBL)
    }
}

// ============================================================================
// DATA CLASSES
// ============================================================================

data class BallDetection(
    val centerX: Int,
    val centerY: Int,
    val area: Double,
    val color: String  // "G" or "P"
)
