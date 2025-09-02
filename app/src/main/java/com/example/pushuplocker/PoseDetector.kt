package com.example.pushuplocker

import android.graphics.Bitmap

data class KeyPoint(val x: Float, val y: Float, val score: Float)

class PoseDetector {

    /**
     * Processes an image and returns a list of keypoints.
     *
     * @param image The input image.
     * @return A list of keypoints.
     */
    fun processImage(image: Bitmap): List<KeyPoint> {
        // TODO: Implement pose detection logic here.
        return emptyList()
    }
}
